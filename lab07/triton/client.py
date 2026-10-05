"""Send the cat image to the tinyvit model on Triton and print the top 5 classes.

Run from the lab07 directory (with Triton running):
    python triton/client.py
"""
import timm
import tritonclient.http as httpclient
from PIL import Image

# The same preprocessing that was used to train the model (resize, center crop, normalize).
model = timm.create_model("tiny_vit_5m_224.dist_in22k", pretrained=False)
data_config = timm.data.resolve_data_config({}, model=model)
transforms = timm.data.create_transform(**data_config, is_training=False)

image = Image.open("imgs/cat.jpg").convert("RGB")
batch = transforms(image).unsqueeze(0).numpy()  # shape (1, 3, 224, 224), float32

client = httpclient.InferenceServerClient(url="localhost:8000")

inputs = [httpclient.InferInput("input", list(batch.shape), "FP32")]
inputs[0].set_data_from_numpy(batch)
# class_count=5 asks Triton for the top 5 classes instead of all 21841 probabilities.
outputs = [httpclient.InferRequestedOutput("probabilities", class_count=5)]

result = client.infer(model_name="tinyvit", inputs=inputs, outputs=outputs)

# Each entry has the format "<probability>:<class index>:<label>".
for entry in result.as_numpy("probabilities")[0]:
    probability, index, label = entry.decode().split(":", 2)
    print(f"{float(probability):.3f}  {label} (class {index})")
