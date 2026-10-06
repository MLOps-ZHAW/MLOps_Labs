# Deploying models with NVIDIA Triton Inference Server

## Why do I even need a deployment server?

You might be wondering what speaks against loading your model with PyTorch, calling `.eval()`, and
wrapping everything in a FastAPI or Flask service.

This is undoubtedly a common approach, but there are a few considerations and potential drawbacks:

1. **Performance**: PyTorch is a powerful deep learning library, but it is not the most efficient choice for serving predictions in a production environment, especially if the service needs to handle a large number of requests concurrently. Python's global interpreter lock, the overhead of the framework, and one request at a time all limit how much you get out of your hardware.

2. **Resource Management**: Serving deep learning models requires careful resource management, especially memory usage and GPU utilization. In a production setting, it's essential to monitor resource usage and implement strategies such as batching to optimize performance and scalability. Purpose-built inference servers come with such tools built in.

3. **Deployment complexity**: While deploying a model with PyTorch and wrapping it in a web service using FastAPI or Flask is relatively straightforward, managing the deployment pipeline, versioning, and scaling can become complex, especially in large-scale production environments. Inference servers come with dedicated model management solutions.

## Triton in a nutshell

[NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) is an open-source inference server and one of the industry standards for serving models in production. It is available on all major clouds (AWS SageMaker, Google Vertex AI, Azure ML) and is used as a serving runtime by Kubernetes platforms such as KServe.

The key ideas:

- **Triton serves models, not code.** You give it a model file, e.g. in ONNX, TorchScript or TensorRT format, plus a small configuration file. Triton runs the model with the matching _backend_ (ONNX Runtime, PyTorch, TensorRT, OpenVINO, Python, ...). You don't write a web service at all.
- **Models live in a _model repository_**, a folder with a fixed structure:

  ```text
  model_repository/
  └── tinyvit/              <- model name
      ├── config.pbtxt      <- model configuration
      ├── labels.txt        <- (optional) class labels
      └── 1/                <- model version
          └── model.onnx    <- the model itself
  ```

- **Standardized APIs.** Triton implements the [Open Inference Protocol](https://github.com/kserve/open-inference-protocol) (also known as the KServe v2 protocol) over HTTP and gRPC. The same protocol is spoken by KServe, Seldon and others, so clients are portable between servers.
- **Built-in scaling and monitoring**: multiple model instances, dynamic batching, model pipelines (ensembles), and Prometheus metrics.

Triton runs as a Docker container. By default it listens on three ports:

|Port|Purpose|
|:---|:------|
|8000|HTTP / REST API|
|8001|gRPC API|
|8002|Prometheus metrics|

We use the image `nvcr.io/nvidia/tritonserver:26.09-py3` (Triton 2.73.0). If you haven't pulled it yet (see the [README](./README_lab07.md#setup)), do it now.

In this part, we will deploy a [TinyViT](https://huggingface.co/timm/tiny_vit_5m_224.dist_in22k) vision transformer that was trained on `ImageNet-22k` (21,841 classes). It comes from [timm](https://huggingface.co/docs/timm) (PyTorch Image Models), a library of pretrained computer vision models. `timm.create_model("tiny_vit_5m_224.dist_in22k")` returns a regular PyTorch `nn.Module`. The name encodes the architecture (TinyViT, ~5M parameters, 224x224 input) and the weights (distilled on ImageNet-22k). The weights are hosted on the Hugging Face Hub, and each model also ships its preprocessing configuration (resize, crop, normalization), which we will need in step 4.

The steps are:

1. Export the model to ONNX.
2. Write the model configuration.
3. Start the server.
4. Send requests to it.
5. Move the preprocessing to the server with the Python backend and an ensemble.
6. Scale the service with model instances and dynamic batching.
7. Monitor it.

All commands below are run from the `lab07` directory with the lab environment activated (`conda activate mlops-lab-07`).

## Step 1: Export the model to ONNX

Triton can't run arbitrary Python code that builds a `timm` model. Instead, we export the model to [ONNX](https://onnx.ai/), an open, framework-independent format for neural networks. An ONNX file contains the computation graph and the weights, and can be executed without PyTorch, e.g. by [ONNX Runtime](https://onnxruntime.ai/). This decoupling of training framework and serving runtime is one of the most common steps when deploying a model.

Have a look at [`triton/export_tinyvit.py`](./triton/export_tinyvit.py), then run it:

```shell
python triton/export_tinyvit.py
```

It writes `model_repository/tinyvit/1/model.onnx` (about 50 MB). A few things worth noticing in the script:

- We append a `Softmax` to the model, so the exported graph returns probabilities instead of logits.
- `dynamic_shapes` marks the first dimension (the batch size) as dynamic. Without this, the ONNX model would only accept exactly the batch size of the dummy input - and Triton could not batch requests.
- After exporting, we run the ONNX model with ONNX Runtime and check that it produces the same output as PyTorch. Always do this: exporters do have bugs, and you don't want to find them in production.

## Step 2: The model configuration

Every model needs a `config.pbtxt` (a [protobuf text file](https://protobuf.dev/reference/protobuf/textformat-spec/)). Ours is in [`model_repository/tinyvit/config.pbtxt`](./model_repository/tinyvit/config.pbtxt):

```protobuf
name: "tinyvit"
backend: "onnxruntime"
max_batch_size: 8

input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]

output [
  {
    name: "probabilities"
    data_type: TYPE_FP32
    dims: [ 21841 ]
    label_filename: "labels.txt"
  }
]

instance_group [
  {
    count: 1
    kind: KIND_CPU
  }
]

parameters { key: "intra_op_thread_count" value: { string_value: "4" } }
```

- `backend` selects how the model is executed, here with ONNX Runtime.
- `max_batch_size` is the largest batch the model accepts. Triton then adds the batch dimension to `dims` itself: the actual input shape is `[batch, 3, 224, 224]`.
- `input` / `output` must match the names we gave the tensors in the export script.
- `label_filename` points to [`labels.txt`](./model_repository/tinyvit/labels.txt), one label per line, in the order of the model's classes. It was generated with `timm.data.ImageNetInfo("imagenet-22k")`. We will use it in step 4.
- `instance_group` defines how many copies of the model run in parallel and on which device. We start with one copy on the CPU.
- `intra_op_thread_count` limits ONNX Runtime to 4 CPU threads per model instance. We will see in step 6 why this matters.

The full reference of all options is in the [model configuration docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html).

## Step 3: Start the server

Start Triton and mount the model repository into the container:

```shell
docker run --rm -it -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g \
  -v "$(pwd)/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:26.09-py3 \
  tritonserver --model-repository=/models --model-control-mode=explicit --load-model=tinyvit
```

_On Windows (PowerShell), write the command on one line and use `-v "${PWD}/model_repository:/models"`._

The repository contains two more models (`preprocess` and `tinyvit_pipeline`) that we will use in step 5. They need an extra Python package that is not part of the official image, so for now we tell Triton to load only `tinyvit` (`--model-control-mode=explicit --load-model=tinyvit`).

After a few seconds, you should see a table with the model status, and that the servers are running:

```text
+---------+---------+--------+
| Model   | Version | Status |
+---------+---------+--------+
| tinyvit | 1       | READY  |
+---------+---------+--------+
...
Started GRPCInferenceService at 0.0.0.0:8001
Started HTTPService at 0.0.0.0:8000
Started Metrics Service at 0.0.0.0:8002
```

Keep this terminal open (stop the server with `Ctrl+C`) and open a second terminal for the next steps. Let's check that everything works:

```shell
curl localhost:8000/v2/health/ready -v       # HTTP 200 if the server is ready
curl localhost:8000/v2/models/tinyvit        # model metadata: inputs and outputs
curl localhost:8000/v2/models/tinyvit/config # the full configuration
```

_No `curl`? Open the URLs in your browser instead (except the first one, which returns an empty body)._

The metadata should look like this:

```json
{"name":"tinyvit","versions":["1"],"platform":"onnxruntime_onnx",
 "inputs":[{"name":"input","datatype":"FP32","shape":[-1,3,224,224]}],
 "outputs":[{"name":"probabilities","datatype":"FP32","shape":[-1,21841]}]}
```

`-1` is the dynamic batch dimension. Also have a look at the full configuration: Triton filled in many defaults that we did not specify.

## Step 4: Send requests

### The Open Inference Protocol

An inference request is a `POST` to `/v2/models/<model name>/infer`. Its body lists the input tensors with their name, shape, datatype and data. You can send it with any HTTP client, e.g. with `requests`:

```python
import numpy as np
import requests

x = np.zeros((1, 3, 224, 224), dtype=np.float32)  # a (very boring) black image
request = {
    "inputs": [
        {"name": "input", "shape": list(x.shape), "datatype": "FP32", "data": x.flatten().tolist()}
    ]
}
response = requests.post("http://localhost:8000/v2/models/tinyvit/infer", json=request)
print(response.json()["outputs"][0]["shape"])  # [1, 21841]
```

Sending 150,528 floats as JSON text is slow, though. That's why Triton supports a binary extension of the protocol, and the `tritonclient` package uses it for you:

```python
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = [httpclient.InferInput("input", list(x.shape), "FP32")]
inputs[0].set_data_from_numpy(x)

result = client.infer(model_name="tinyvit", inputs=inputs)
probabilities = result.as_numpy("probabilities")  # NumPy array of shape (1, 21841)
```

`tritonclient.grpc` offers the same interface over gRPC (port 8001), which is even more efficient.

#### Your turn

Send the cat image [`imgs/cat.jpg`](./imgs/cat.jpg) to the server and print the index and probability of the most likely class. Then look up the label in `model_repository/tinyvit/labels.txt` (line `index + 1`).

_Hint: The model expects a preprocessed image: resized, center-cropped to 224x224, normalized, in the layout `(batch, channels, height, width)`. Use the same preprocessing as during training. `timm` gives it to you:_

```python
import timm

model = timm.create_model("tiny_vit_5m_224.dist_in22k", pretrained=False)
data_config = timm.data.resolve_data_config({}, model=model)
transforms = timm.data.create_transform(**data_config, is_training=False)
```

<details>
    <summary>Solution</summary>

```python
import numpy as np
import timm
import tritonclient.http as httpclient
from PIL import Image

model = timm.create_model("tiny_vit_5m_224.dist_in22k", pretrained=False)
data_config = timm.data.resolve_data_config({}, model=model)
transforms = timm.data.create_transform(**data_config, is_training=False)

image = Image.open("imgs/cat.jpg").convert("RGB")
batch = transforms(image).unsqueeze(0).numpy()  # shape (1, 3, 224, 224)

client = httpclient.InferenceServerClient(url="localhost:8000")
inputs = [httpclient.InferInput("input", list(batch.shape), "FP32")]
inputs[0].set_data_from_numpy(batch)
probabilities = client.infer(model_name="tinyvit", inputs=inputs).as_numpy("probabilities")[0]

index = int(np.argmax(probabilities))
with open("model_repository/tinyvit/labels.txt") as f:
    labels = f.read().splitlines()
print(f"{probabilities[index]:.3f}  {labels[index]} (class {index})")  # 0.508  Burmese cat (class 2404)
```

</details>

#### Another turn

Every client now has to download the labels and pick the top classes itself. Triton can do this for you: since the output has a `label_filename`, you can ask for the top `k` classes with the [classification extension](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_classification.md). Instead of 21,841 probabilities, Triton then returns `k` strings of the form `"<probability>:<class index>:<label>"`.

Modify your client to print the top 5 classes with their probabilities.

_Hint: Request the output with `httpclient.InferRequestedOutput("probabilities", class_count=5)` and pass it to `client.infer(..., outputs=[...])`. The result contains bytes, so `.decode()` them._

<details>
    <summary>Solution</summary>

The solution is also available as [`triton/client.py`](./triton/client.py) (`python triton/client.py`).

```python
outputs = [httpclient.InferRequestedOutput("probabilities", class_count=5)]
result = client.infer(model_name="tinyvit", inputs=inputs, outputs=outputs)

for entry in result.as_numpy("probabilities")[0]:
    probability, index, label = entry.decode().split(":", 2)
    print(f"{float(probability):.3f}  {label} (class {index})")
```

Output:

```text
0.508  Burmese cat (class 2404)
0.051  domestic cat, house cat, Felis domesticus, Felis catus (class 2388)
0.032  alley cat (class 2391)
0.023  cat, true cat (class 2387)
0.022  feline, felid (class 2386)
```

In the plain JSON protocol, the same is requested with `"outputs": [{"name": "probabilities", "parameters": {"classification": 5}}]`.

</details>

## Step 5: Preprocessing on the server - Python backend and ensembles

Our client still has to install `timm` and `torch` just to preprocess the image. That's not only inconvenient, it is also risky: if a client preprocesses images slightly differently than during training (a different resize method, a forgotten normalization, ...), the model silently gets worse. This is known as **training-serving skew**, and it is one of the most common bugs in deployed ML systems. It is much safer to do the preprocessing on the server, and let clients send plain JPEG files.

Triton offers two tools for this:

- The **Python backend** runs a Python file as a "model". This is the escape hatch for anything that is not a neural network: pre- and postprocessing, business logic, or models that can't be exported.
- An **ensemble** chains several models into a pipeline that clients call like a single model. Triton passes the tensors between the steps without a round trip to the client.

Have a look at the two new models in the repository:

- [`model_repository/preprocess/1/model.py`](./model_repository/preprocess/1/model.py): a `TritonPythonModel` class whose `execute()` method receives a batch of requests. It decodes the image bytes with Pillow and replicates the `timm` preprocessing step by step: resize the shorter side to 235 pixels (bicubic), center crop to 224x224, scale to `[0, 1]`, normalize, and transpose to channels-first. The [`config.pbtxt`](./model_repository/preprocess/config.pbtxt) declares a `TYPE_STRING` (bytes) input `image` and the `input` tensor as output.
- [`model_repository/tinyvit_pipeline/config.pbtxt`](./model_repository/tinyvit_pipeline/config.pbtxt): an ensemble (`platform: "ensemble"`) with two steps. The `input_map` / `output_map` entries wire the tensors: the ensemble input `image` goes into `preprocess`, its output (called `preprocessed_image` inside the ensemble) goes into `tinyvit`, whose `probabilities` become the ensemble output. An ensemble has no model file, but Triton still needs an (empty) version folder `1/`.

### Packaging: our own Triton image

The Python backend runs inside the Triton container, and the official image does not contain Pillow. So we need our own image that adds the dependencies of our models. This is what the [`triton/Dockerfile`](./triton/Dockerfile) does:

```dockerfile
FROM nvcr.io/nvidia/tritonserver:26.09-py3

RUN pip install --no-cache-dir --break-system-packages pillow==12.3.0
```

(`--break-system-packages` is needed because the image's Python is managed by the operating system. Inside a container that only serves models, this is fine.)

Build the image:

```shell
docker build -t tritonserver-lab07 triton/
```

Stop the running server (`Ctrl+C`) and start the new image. This time without `--model-control-mode`, so Triton loads all models in the repository:

```shell
docker run --rm -it -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size=1g \
  -v "$(pwd)/model_repository:/models" \
  tritonserver-lab07 \
  tritonserver --model-repository=/models
```

All three models should be `READY` now. (`--shm-size=1g` gives the Python backend enough shared memory to exchange tensors with the server.)

The client only has to send the raw bytes of the file now. Have a look at [`triton/client_pipeline.py`](./triton/client_pipeline.py) and run it:

```shell
python triton/client_pipeline.py
```

You should get exactly the same top 5 classes and probabilities as in step 4: the server-side preprocessing produces bit-identical inputs. Note that this client needs neither `timm` nor `torch` - only `tritonclient` and `numpy`.

_Question: Why did we have to be so careful in `model.py` with rounding the resize and crop sizes exactly like `torchvision` does? What would happen if we hadn't?_

<details>
    <summary>Answer</summary>

Any difference in preprocessing is training-serving skew. A one-pixel shift in the center crop already changes the probability for "Burmese cat" from 0.508 to 0.461 for our cat image. Here, the effect is small, but you would never notice it without comparing against the reference implementation - which is exactly what you should do (and what we did while writing this lab).

</details>

The image we just built is also how you would ship the service: push it (together with the model repository, or with the models baked into the image) to a container registry and run it on any machine or Kubernetes cluster.

## Step 6: Scaling the service

Triton offers two main knobs to get more throughput out of your hardware. Both are set in `config.pbtxt`.

### Model instances

The `instance_group` setting controls how many copies of the model Triton runs in parallel. With `count: 4`, four requests can be processed at the same time:

```protobuf
instance_group [
  {
    count: 4
    kind: KIND_CPU
  }
]
```

On a CPU, each instance uses `intra_op_thread_count` threads. Keep _instances x threads_ at or below the number of CPU cores of your machine - otherwise the threads compete for the cores and everything gets slower. (This is why we set `intra_op_thread_count` at all. Without it, ONNX Runtime uses all cores for every single request, and while developing this lab, a single instance was 8x slower under load than with 4 threads!)

### Dynamic batching

The dynamic batcher collects individual requests that arrive close together and runs them through the model as one batch:

```protobuf
dynamic_batching {
  max_queue_delay_microseconds: 5000
}
```

`max_queue_delay_microseconds` is how long Triton may wait for more requests to fill a batch (at most `max_batch_size`). Why is batching useful? GPUs (and to a lesser extent CPUs) process a batch of inputs much more efficiently than the same inputs one by one, and the per-request overhead is shared. The price is a bit of extra latency for waiting.

Unlike a hand-written service, the model doesn't need any changes for batching: Triton concatenates the requests along the batch dimension and splits the results again. This only works because we exported the model with a dynamic batch dimension.

### Your turn: measure it

[`triton/benchmark.py`](./triton/benchmark.py) simulates several users who send requests at the same time. It reports the throughput, the latency, and the average batch size that Triton actually executed:

```shell
python triton/benchmark.py --concurrency 8 --requests 200
```

Measure the following configurations of `tinyvit`. After every change to `config.pbtxt`, restart the server (`Ctrl+C` and start it again). Choose the instance count and thread count so that they fit your machine (e.g. 2 instances x 2 threads on a 4-core laptop).

|Configuration|Throughput (req/s)|Latency p50 / p95 (ms)|Avg. batch size|
|:------------|:-----------------|:---------------------|:--------------|
|1 instance (baseline)| | | |
|1 instance + dynamic batching| | | |
|N instances| | | |
|N instances + dynamic batching| | | |

What do you observe? Which configuration would you choose if you cared most about throughput, and which one if you cared most about latency?

<details>
    <summary>Example results and discussion</summary>

Your numbers will differ depending on your machine. On a server CPU with 4 threads per instance and `--concurrency 8`, we measured:

|Configuration|Throughput (req/s)|Latency p50 / p95 (ms)|Avg. batch size|
|:------------|:-----------------|:---------------------|:--------------|
|1 instance (baseline)|52|152 / 159|1.0|
|1 instance + dynamic batching|60|129 / 149|3.9|
|4 instances|147|55 / 61|1.0|
|4 instances + dynamic batching|119|69 / 100|2.5|

- On a CPU, **more instances** helped most, as long as there were enough cores. With 8 concurrent users and only one instance, most of the latency is time spent waiting in the queue.
- **Dynamic batching** worked (the server executed batches of about 4), but helped only a little on the CPU, and with 4 instances it even hurt: the instances were busy anyway, so waiting for fuller batches only added latency.
- On a **GPU**, the picture changes: batching is where GPUs shine. With `perf_analyzer` (see below) on an A100, dynamic batching doubled the throughput of a single instance, from 475 to 955 requests/s.

There is no free lunch, and no universally best configuration: it depends on the model, the hardware and the traffic pattern. Always measure.

</details>

### Running on a GPU

If you have an NVIDIA GPU (and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed), add `--gpus all` to the `docker run` command and change `kind: KIND_CPU` to `kind: KIND_GPU`. When you combine a GPU with dynamic batching, also add

```protobuf
parameters { key: "cudnn_conv_algo_search" value: { string_value: "1" } }
```

By default, ONNX Runtime benchmarks all convolution algorithms whenever the input shape changes. With dynamic batching, the batch size - and therefore the shape - changes all the time, and this made batching 7x _slower_ while developing this lab. The value `1` selects a fast heuristic instead.

On a GPU, the model is so fast that our Python benchmark script becomes the bottleneck. For serious measurements, use NVIDIA's [`perf_analyzer`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/README.html). On Linux, you can install it with `pip install perf-analyzer`; on other systems, it is part of the `nvcr.io/nvidia/tritonserver:26.09-py3-sdk` image. Example:

```shell
perf_analyzer -m tinyvit --concurrency-range 1:16:4
```

## Step 7: Monitoring

A model in production needs monitoring. Triton exposes metrics in the [Prometheus / OpenMetrics format](https://openmetrics.io/) on port 8002. While your server is running, open <http://localhost:8002/metrics>. Among many others, you will find:

- `nv_inference_request_success` / `nv_inference_request_failure`: number of successful and failed requests per model.
- `nv_inference_count` and `nv_inference_exec_count`: number of inferences and number of model executions. Their ratio is the average batch size - this is how `benchmark.py` computes it.
- `nv_inference_queue_duration_us` and `nv_inference_compute_infer_duration_us`: cumulative time spent waiting in the queue and computing. If the queue time grows, you need more instances.
- `nv_cpu_utilization`, `nv_gpu_utilization`, `nv_gpu_memory_used_bytes`: resource usage.

In production, Prometheus scrapes this endpoint regularly and Grafana visualizes the metrics and alerts you. We won't set this up in this lab, but it's a great thing to explore in your own projects. More detailed per-model statistics, including a breakdown per batch size, are available via the API at `http://localhost:8000/v2/models/tinyvit/stats`.

## What about large language models?

We served a vision model, but the same concepts apply to most "classic" models: tabular, vision, and smaller NLP models. Large language models are a special case: generating text token by token needs dedicated techniques (KV caching, continuous batching, paged attention), so they are typically served with specialized engines such as [vLLM](https://docs.vllm.ai/), [SGLang](https://docs.sglang.ai/) or NVIDIA's [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). Triton can run vLLM and TensorRT-LLM as backends, too.

## Outlook: Kubernetes and KServe

So far, we have one server on one machine. To run models at scale, you need to deploy them on many machines, scale them with the traffic, roll out new model versions safely (canary deployments), and so on. This is what [KServe](https://kserve.github.io/), a CNCF project for model serving on Kubernetes, does. KServe can use Triton as its serving runtime: you give it the model repository and it takes care of the rest. Because both speak the Open Inference Protocol, your clients stay exactly the same.

## Troubleshooting

- **`failed to load all models` / `No module named 'PIL'`**: you started the official image without `--model-control-mode=explicit --load-model=tinyvit`. Either add these flags or use the image `tritonserver-lab07` from step 5. By default, Triton refuses to start if any model in the repository fails to load - the log tells you which one and why.
- **A model is `UNAVAILABLE`**: read the error in the model status table. Typical causes are typos in `config.pbtxt`, names that don't match the ONNX model, or a missing `model.onnx` (did you run the export script?).
- **`address already in use`**: another server (or an old Triton container) is still running on port 8000. Stop it, e.g. with `docker ps` and `docker stop <container id>`.
- **The model directory is empty inside the container (Windows)**: use `-v "${PWD}/model_repository:/models"` in PowerShell and make sure Docker Desktop has access to the drive.
- **Changes to `config.pbtxt` or `model.py` have no effect**: restart the server. Triton only reads the model repository when it starts (unless you enable [model control](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html)).
- **Podman instead of Docker**: all commands work with `podman` as well.
