"""Export the TinyViT classifier to ONNX and place it in the Triton model repository.

Run from the lab07 directory:
    python triton/export_tinyvit.py
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import timm
import torch

MODEL_NAME = "tiny_vit_5m_224.dist_in22k"
OUTPUT_PATH = Path(__file__).parent.parent / "model_repository" / "tinyvit" / "1" / "model.onnx"


def main():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.eval()
    # Add the softmax to the exported graph, so that Triton returns probabilities.
    model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))

    dummy_input = torch.randn(2, 3, 224, 224)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input,),
        OUTPUT_PATH,
        input_names=["input"],
        output_names=["probabilities"],
        # The batch dimension is dynamic, so Triton can batch requests.
        dynamic_shapes={"input": {0: torch.export.Dim("batch", min=1, max=64)}},
        external_data=False,  # Store the weights inside model.onnx.
    )
    print(f"Exported {MODEL_NAME} to {OUTPUT_PATH}")

    # Sanity check: ONNX Runtime and PyTorch must produce the same output.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4  # Avoids thread-affinity warnings on some machines.
    session = ort.InferenceSession(OUTPUT_PATH, options, providers=["CPUExecutionProvider"])
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        expected = model(x).numpy()
    actual = session.run(["probabilities"], {"input": x.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
    print("ONNX Runtime output matches PyTorch.")


if __name__ == "__main__":
    main()
