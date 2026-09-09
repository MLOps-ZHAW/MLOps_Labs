# MLOps Pipeline by Vertex AI

## Description

This project demonstrates an end-to-end MLOps pipeline using Google Cloud's Vertex AI. The pipeline automates data fetching, model training and evaluation, and model deployment on Vertex AI for serving predictions.

## Prerequisites

- **Google Cloud Project**: A GCP project with billing enabled.
- **Vertex AI API**: Ensure that the Vertex AI API and other necessary APIs are enabled in your project.
- **Service Account**: A service account with the following permissions:
  - Access to BigQuery datasets.
  - Permissions to create and manage Vertex AI resources.
  - Permissions to read and write to Google Cloud Storage.
- **Local Environment**:
  - Docker and Docker Compose installed.
  - Google Cloud SDK (`gcloud` CLI) installed and authenticated.
  - `jq` command-line JSON processor installed.
- **Configuration Files**:
  - `config.json`: Copy from `config.json.sample` and fill in your project-specific details.
  - `credential.json`: Downloaded service account key file placed in the project root.

## Scope

The project focuses on automating the lifecycle of a machine learning model for text classification using Vertex AI. It includes:

- Fetching data from BigQuery.
- Training a DistilBERT-based text classification model using Hugging Face Transformers.
- Deploying the trained model to Vertex AI for real-time predictions.
- Building a custom Docker image for model serving.
- Orchestrating the entire pipeline using Kubeflow Pipelines (KFP).

## Data

- **Dataset**: AG News dataset, a collection of news articles labeled into four categories.
- **Data Preparation**:
  - The `prepare_dataset.py` script loads and preprocesses the data.
  - Data is uploaded to BigQuery with an added `timestamp` field to simulate recent data entries.
- **Data Fetching**:
  - The pipeline fetches data from BigQuery based on a timestamp filter (last two days). We do not need to use BigQuery for this dataset, and it is just for demonstration purposes towards a real-world scenario.

## Model

- **Architecture**: Based on `distilbert-base-uncased` from Hugging Face Transformers.
- **Purpose**: Fine-tuned for text classification to categorize news articles.
- **Classes**: Classifies texts into four categories:
  - World
  - Sports
  - Business
  - Sci/Tech
- **Training Details**:
  - Tokenization using `AutoTokenizer`.
  - Training with `Trainer` API.
  - Evaluation metrics: accuracy, precision, recall, F1 score.
- **Artifacts**: The trained model, tokenizer, and label encoder are saved for deployment.


## Building and pushing the container image for inference

```
cd serving
CONFIG=$(cat ../config.json | jq '.model_tag, .region, .repository_name, .image_name' | sed 's/"//g' | tr '\n' ',')
gcloud builds submit \
    --config cloudbuild.yaml \
    --substitutions _MODEL_VERSION=$(echo $CONFIG | cut -d, -f1),_LOCATION=$(echo $CONFIG | cut -d, -f2),_REPOSITORY_NAME=$(echo $CONFIG | cut -d, -f3),_IMAGE_NAME=$(echo $CONFIG | cut -d, -f4)
```

For the first time, you need to authenticate with your Google Cloud service account:
```
gcloud auth activate-service-account <service-account> --key-file=./credential.json
gcloud auth configure-docker <location>-docker.pkg.dev
```

## Building docker image
```
docker compose build
```

### Preparing data for training
Here, I will use IMDb dataset for training. The script will register only part of it to Google BigQuery.
```
docker compose run mlops-v1 python prepare_dataset.py
```

### Starting pipeline
```
docker compose run mlops-v1
```
In my case, the pipeline completed in 1 hour.

## Requesting predictions
I have prepared a sample request script `sample-request.sh` to test the deployed model.
```
chmod +x sample-request.sh
./sample-request.sh
```

The inference output will look like:
```
{
  "predictions": [
    {
      "probabilities": {
        "Sci/Tech": 0.0032518664374947548,
        "World": 0.98935961723327637,
        "Sports": 0.0034750890918076038,
        "Business": 0.0039134090766310692
      },
      "predicted_class": "World"
    },
    {
      "probabilities": {
        "World": 0.013002258725464341,
        "Sports": 0.97295308113098145,
        "Business": 0.0041450257413089284,
        "Sci/Tech": 0.00989964697510004
      },
      "predicted_class": "Sports"
    },
    {
      "probabilities": {
        "World": 0.0058785984292626381,
        "Sports": 0.0053764986805617809,
        "Sci/Tech": 0.029164601117372509,
        "Business": 0.95958030223846436
      },
      "predicted_class": "Business"
    },
    {
      "predicted_class": "Sci/Tech",
      "probabilities": {
        "Business": 0.018011067062616348,
        "Sports": 0.002777427202090621,
        "Sci/Tech": 0.97608381509780884,
        "World": 0.003127649892121553
      }
    }
  ],
  "deployedModelId": "xxx",
  "model": "projects/xxx/locations/xxx/models/text-classification-model",
  "modelDisplayName": "text-classification-model",
  "modelVersionId": "1"
}
```
