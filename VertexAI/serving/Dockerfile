FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only torch build: the model is deployed on a CPU machine type
# (see deploy_model's machine_type in ../main.py), and the default PyPI
# wheel would otherwise pull in several GB of unused CUDA libraries.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# AIP_HTTP_PORT/AIP_HEALTH_ROUTE/AIP_PREDICT_ROUTE are injected by the
# Agent Platform (formerly Vertex AI) prediction service at deploy time,
# matching the serving_container_* arguments passed to Model.upload() in
# ../main.py.
ENV AIP_HTTP_PORT=8080

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${AIP_HTTP_PORT}"]
