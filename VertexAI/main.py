import json

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from kfp import compiler, dsl
from kfp.dsl import (
    Artifact,
    component,
    Dataset,
    Input,
    Metrics,
    Model,
    Output,
)


@component(
    packages_to_install=[
        "db-dtypes==1.7.1",
        "google-cloud-bigquery==3.45.0",
        "pandas==3.0.5",
        "pyarrow==25.0.1",
    ],
    base_image="python:3.13",
)
def fetch_data_from_bigquery(
    project_id: str,
    dataset_id: str,
    table_id: str,
    output_dataset: Output[Dataset],
):
    import datetime
    from google.cloud import bigquery

    # NOTE: The timestamp filter is for testing purposes only
    start_timestamp = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime("%Y-%m-%d %H:%M:%S")
    query = f"""
        SELECT text, label
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE timestamp >= '{start_timestamp}'
    """

    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    print(f"Number of rows: {len(df)}")
    df.to_parquet(output_dataset.path)


@component(
    packages_to_install=[
        "accelerate==1.15.0",
        "pandas==3.0.5",
        "pyarrow==25.0.1",
        "scikit-learn==1.9.0",
        "torch==2.14.0",
        "transformers==5.17.0",
    ],
    base_image="python:3.13",
)
def train_model(
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    output_metrics: Output[Metrics],
):
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    from sklearn.preprocessing import LabelEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("PyTorch is using the GPU")
        print("Current device: %s", torch.cuda.current_device())
        print("Device name: %s", torch.cuda.get_device_name(device))
    else:
        print("PyTorch is using the CPU")

    df = pd.read_parquet(input_dataset.path)
    print("Class Distribution in the Dataset:")
    print(df["label"].value_counts())

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["label"])
    num_labels = len(np.unique(labels))
    print(f"Number of Labels: {num_labels}")
    print("LabelEncoder Classes:")
    print(label_encoder.classes_)
    print("Label Mapping:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name}: {i}")

    texts = df["text"].tolist()
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, stratify=labels)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    num_labels = len(label_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

    max_length = 128
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = TextDataset(train_encodings, train_labels)
    val_dataset = TextDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir=str(Path(output_model.path) / "results"),
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(Path(output_model.path) / "logs"),
        eval_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    trainer.train()

    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, preds, average="macro")
    accuracy = accuracy_score(val_labels, preds)

    output_metrics.log_metric("accuracy", accuracy)
    output_metrics.log_metric("precision", precision)
    output_metrics.log_metric("recall", recall)
    output_metrics.log_metric("f1", f1)

    cm = confusion_matrix(val_labels, preds)
    print("Confusion Matrix:")
    print(cm)

    model.save_pretrained(output_model.path)
    tokenizer.save_pretrained(output_model.path)
    joblib.dump(label_encoder, f"{output_model.path}/label_encoder.joblib")


@component(
    packages_to_install=["google-cloud-aiplatform==2.1.0"],
    base_image="python:3.13",
)
def deploy_model(
    project_id: str,
    region: str,
    service_account: str,
    repository_name: str,
    image_name: str,
    model_tag: str,
    model_display_name: str,
    endpoint_display_name: str,
    model: Input[Model],
    deployed_model: Output[Model],
    vertex_endpoint: Output[Artifact],
):
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=region,
        service_account=service_account,
    )

    # List and undeploy models from endpoints
    endpoints = aiplatform.Endpoint.list(
        project=project_id,
        location=region,
        filter=f'display_name="{endpoint_display_name}"',
    )
    for _endpoint in endpoints:
        deployed_models = _endpoint.list_models()
        for deployed_model in deployed_models:
            print(f"Undeploying model: {deployed_model.model} from endpoint: {_endpoint.name}")
            _endpoint.undeploy(deployed_model_id=deployed_model.id)
        print(f"Deleting endpoint: {_endpoint.name}")
        _endpoint.delete()

    # List and delete existing models with the same display name
    models = aiplatform.Model.list(
        project=project_id,
        location=region,
        filter=f'display_name="{model_display_name}"',
    )
    for _model in models:
        print(f"Deleting model: {_model.name}")
        _model.delete()

    uploaded_model = aiplatform.Model.upload(
        display_name=model_display_name,
        model_id=model_display_name,
        serving_container_image_uri=f"{region}-docker.pkg.dev/{project_id}/{repository_name}/{image_name}:{model_tag}",
        artifact_uri=model.uri,
        serving_container_ports=[8080],
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
    )

    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_display_name,
        endpoint_id=endpoint_display_name,
    )

    deployed_model = uploaded_model.deploy(
        endpoint=endpoint,
        traffic_percentage=100,
        machine_type="n1-standard-4",
        service_account=service_account,
        min_replica_count=1,
        max_replica_count=1,
    )

    deployed_model.uri = uploaded_model.resource_name
    vertex_endpoint.uri = deployed_model.resource_name


@dsl.pipeline(
    name="text-classification-pipeline",
    description="A pipeline that trains and deploys a text classification model",
)
def text_classification_pipeline():
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    project_id = config["project_id"]
    region = config["region"]
    dataset_id = config["dataset_id"]
    table_id = config["table_id"]
    service_account = config["service_account"]
    repository_name = config["repository_name"]
    image_name = config["image_name"]
    model_tag = config["model_tag"]
    model_display_name = config["model_display_name"]
    endpoint_display_name = config["endpoint_display_name"]

    fetch_data_task = fetch_data_from_bigquery(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
    )

    train_task = train_model(
        input_dataset=fetch_data_task.outputs["output_dataset"],
    )

    deploy_task = deploy_model(
        project_id=project_id,
        region=region,
        service_account=service_account,
        repository_name=repository_name,
        image_name=image_name,
        model_tag=model_tag,
        model_display_name=model_display_name,
        endpoint_display_name=endpoint_display_name,
        model=train_task.outputs["output_model"],
    )

    print(deploy_task.outputs)


if __name__ == "__main__":
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    aiplatform.init(
        project=config["project_id"],
        location=config["region"],
        service_account=config["service_account"],
    )

    compiler.Compiler().compile(
        pipeline_func=text_classification_pipeline,
        package_path="text_classification_pipeline.json",
    )

    job = pipeline_jobs.PipelineJob(
        display_name="text-classification-pipeline",
        template_path="text_classification_pipeline.json",
        pipeline_root=f"gs://{config['bucket_name']}/pipeline_root",
        enable_caching=False,
    )

    job.run(service_account=config["service_account"])
