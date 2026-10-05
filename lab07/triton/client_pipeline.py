"""Send the raw bytes of the cat image to the tinyvit_pipeline ensemble and print the top 5 classes.

The preprocessing now happens on the server, so the client needs neither timm nor torch.

Run from the lab07 directory (with Triton running):
    python triton/client_pipeline.py
"""
import numpy as np
import tritonclient.http as httpclient

with open("imgs/cat.jpg", "rb") as f:
    image_bytes = f.read()

# A batch with one image. Triton's BYTES/STRING tensors are NumPy arrays of dtype object.
batch = np.array([[image_bytes]], dtype=object)  # shape (1, 1)

client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = [httpclient.InferInput("image", list(batch.shape), "BYTES")]
inputs[0].set_data_from_numpy(batch)
outputs = [httpclient.InferRequestedOutput("probabilities", class_count=5)]

result = client.infer(model_name="tinyvit_pipeline", inputs=inputs, outputs=outputs)

for entry in result.as_numpy("probabilities")[0]:
    probability, index, label = entry.decode().split(":", 2)
    print(f"{float(probability):.3f}  {label} (class {index})")
