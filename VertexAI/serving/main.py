import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import joblib
import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from google.cloud import storage
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set by the Agent Platform (formerly Vertex AI) prediction service to a gs:// URI
# pointing at the artifact_uri passed to aiplatform.Model.upload(). The platform does
# NOT mount these files locally - the container is responsible for downloading them.
AIP_STORAGE_URI = os.environ.get("AIP_STORAGE_URI")
# Fallback for local testing without a real deployment.
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "/tmp/model")

MODEL_STATE = {}


def download_model_artifacts(storage_uri: str, local_dir: str) -> None:
    parsed = urlparse(storage_uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.lstrip("/")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        raise FileNotFoundError(f"No model artifacts found under {storage_uri}")

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative_path = os.path.relpath(blob.name, prefix)
        destination = Path(local_dir) / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination))
        logger.info("Downloaded %s to %s", blob.name, destination)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = LOCAL_MODEL_DIR
    if AIP_STORAGE_URI:
        logger.info("Downloading model artifacts from %s", AIP_STORAGE_URI)
        download_model_artifacts(AIP_STORAGE_URI, model_dir)
    else:
        logger.info("AIP_STORAGE_URI not set, loading model from %s", model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_STATE["device"] = device
    MODEL_STATE["tokenizer"] = AutoTokenizer.from_pretrained(model_dir)
    MODEL_STATE["model"] = (
        AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    )
    MODEL_STATE["label_encoder"] = joblib.load(Path(model_dir) / "label_encoder.joblib")
    logger.info("Model loaded, ready to serve predictions on device %s", device)

    yield
    MODEL_STATE.clear()


app = FastAPI(lifespan=lifespan)

AIP_HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
AIP_PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")


@app.get(AIP_HEALTH_ROUTE)
async def health():
    return {"status": "ok" if "model" in MODEL_STATE else "loading"}


@app.post(AIP_PREDICT_ROUTE)
async def predict(request: Request):
    body = await request.json()
    instances = body["instances"]
    texts = [instance["text"] for instance in instances]

    tokenizer = MODEL_STATE["tokenizer"]
    model = MODEL_STATE["model"]
    label_encoder = MODEL_STATE["label_encoder"]
    device = MODEL_STATE["device"]

    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**encodings).logits
        probabilities = F.softmax(logits, dim=-1).cpu().numpy()

    class_names = label_encoder.classes_
    predictions = []
    for probs in probabilities:
        predicted_class = class_names[probs.argmax()]
        predictions.append(
            {
                "probabilities": {
                    class_name: float(prob)
                    for class_name, prob in zip(class_names, probs)
                },
                "predicted_class": predicted_class,
            }
        )

    return {"predictions": predictions}
