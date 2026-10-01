import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer


def preprocess_reviews(reviews_raw: pd.DataFrame) -> pd.DataFrame:
    """Rename the quality criteria of the review data to match the CQI data."""
    return reviews_raw.rename(
        columns={
            "aroma": "Aroma",
            "acid": "Acidity",
            "body": "Body",
            "flavor": "Flavor",
            "aftertaste": "Aftertaste",
        }
    )


def fit_imputer(cqi_raw: pd.DataFrame, feature_columns: list[str], missing_columns: list[str]) -> xgb.XGBRegressor:
    """Learn to predict the quality criteria that are missing in the review data from the CQI data."""
    imputer = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, multi_strategy="one_output_per_tree")
    # Look ma, no train-test split!
    imputer.fit(cqi_raw[feature_columns], cqi_raw[missing_columns])
    return imputer


def impute_missing(
    reviews: pd.DataFrame, imputer: xgb.XGBRegressor, feature_columns: list[str], missing_columns: list[str]
) -> pd.DataFrame:
    reviews_imputed = reviews.copy()
    reviews_imputed[missing_columns] = imputer.predict(reviews[feature_columns])
    return reviews_imputed


def _mean_pooling(model_output, attention_mask):
    """Mean pooling - take the attention mask into account for correct averaging."""
    token_embeddings = model_output[0]  # first element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def embed_descriptions(reviews: pd.DataFrame, embedding: dict) -> np.ndarray:
    """Embed the text descriptions and reduce their dimensionality with a PCA."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(embedding["model_name"])
    model = AutoModel.from_pretrained(embedding["model_name"]).to(device)
    batch_size = embedding["batch_size"]

    embeddings = []
    for desc_col in embedding["description_columns"]:
        texts = reviews[desc_col].fillna("").to_list()
        col_embeddings = []
        for i in range(0, len(texts), batch_size):
            encoded_input = tokenizer(
                texts[i : i + batch_size], padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)
            col_embeddings.append(_mean_pooling(model_output, encoded_input["attention_mask"]).cpu())
        embeddings.append(torch.vstack(col_embeddings))
    stacked_embeddings = torch.hstack(embeddings).numpy()
    return PCA(n_components=embedding["pca_components"]).fit_transform(stacked_embeddings)


def combine_features(
    reviews_imputed: pd.DataFrame,
    description_embeddings: np.ndarray,
    feature_columns: list[str],
    missing_columns: list[str],
) -> np.ndarray:
    """Combine quality criteria and text embeddings into a shared latent space."""
    return np.hstack([reviews_imputed[feature_columns + missing_columns].to_numpy(), description_embeddings])
