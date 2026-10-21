import io

import numpy as np
import triton_python_backend_utils as pb_utils
from PIL import Image

# Same preprocessing as timm uses for tiny_vit_5m_224 (see timm.data.resolve_data_config).
RESIZE = 235  # int(224 / crop_pct), with crop_pct = 0.95
CROP = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Resize the shorter side to RESIZE pixels, keeping the aspect ratio.
    width, height = image.size
    if width <= height:
        new_size = (RESIZE, int(RESIZE * height / width))
    else:
        new_size = (int(RESIZE * width / height), RESIZE)
    image = image.resize(new_size, Image.BICUBIC)

    # Crop the CROP x CROP pixels in the center.
    width, height = image.size
    left, top = round((width - CROP) / 2), round((height - CROP) / 2)
    image = image.crop((left, top, left + CROP, top + CROP))

    # Scale to [0, 1], normalize, and change the layout from HWC to CHW.
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD
    return array.transpose(2, 0, 1)


class TritonPythonModel:
    """Decodes JPEG/PNG images and turns them into the input tensor of the tinyvit model."""

    def execute(self, requests):
        responses = []
        # With dynamic batching, Triton can pass several requests at once.
        for request in requests:
            images = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            # images has shape (batch_size, 1) and contains the raw bytes of each image.
            batch = np.stack([preprocess(image[0]) for image in images])
            output = pb_utils.Tensor("input", batch)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
        return responses
