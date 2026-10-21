"""A small load test for the tinyvit model on Triton.

Sends many requests in parallel and reports throughput, latency, and how many
requests Triton batched together.

Run from the lab07 directory (with Triton running):
    python triton/benchmark.py --concurrency 8 --requests 200
"""
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tritonclient.http as httpclient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tinyvit")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of requests in flight at the same time.")
    parser.add_argument("--requests", type=int, default=200, help="Total number of requests to send.")
    args = parser.parse_args()

    url = os.environ.get("TRITON_URL", "localhost:8000")
    client = httpclient.InferenceServerClient(url=url)

    # Every request contains a single (random) image. We only care about the speed here.
    batch = np.random.rand(1, 3, 224, 224).astype(np.float32)
    inputs = [httpclient.InferInput("input", list(batch.shape), "FP32")]
    inputs[0].set_data_from_numpy(batch)
    outputs = [httpclient.InferRequestedOutput("probabilities", class_count=1)]

    # Warm-up, so that the first (slow) requests do not distort the results.
    for _ in range(5):
        client.infer(args.model, inputs, outputs=outputs)
    stats_before = client.get_inference_statistics(args.model)["model_stats"][0]

    # Each thread acts as one user who sends a request, waits for the answer, and sends the next one.
    thread_local = threading.local()

    def send_request(_):
        if not hasattr(thread_local, "client"):
            thread_local.client = httpclient.InferenceServerClient(url=url)
        sent = time.perf_counter()
        thread_local.client.infer(args.model, inputs, outputs=outputs)
        return time.perf_counter() - sent

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        latencies = list(executor.map(send_request, range(args.requests)))
    duration = time.perf_counter() - start

    stats_after = client.get_inference_statistics(args.model)["model_stats"][0]
    n_inferences = stats_after["inference_count"] - stats_before["inference_count"]
    n_executions = stats_after["execution_count"] - stats_before["execution_count"]

    print(f"Model: {args.model}, concurrency: {args.concurrency}, requests: {args.requests}")
    print(f"Throughput:      {args.requests / duration:.1f} requests/s")
    print(f"Latency p50/p95: {np.percentile(latencies, 50) * 1000:.0f} / {np.percentile(latencies, 95) * 1000:.0f} ms")
    print(f"Average batch size on the server: {n_inferences / n_executions:.2f}")
    client.close()


if __name__ == "__main__":
    main()
