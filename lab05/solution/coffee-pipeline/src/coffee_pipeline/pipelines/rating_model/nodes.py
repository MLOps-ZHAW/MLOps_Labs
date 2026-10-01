import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


def train_rating_model(
    shared_latent_space: np.ndarray, reviews: pd.DataFrame, rating_model: dict
) -> tuple[xgb.XGBRegressor, pd.DataFrame]:
    """Some downstream application: predict the review rating from the shared latent space."""
    X_train, X_test, y_train, y_test = train_test_split(
        shared_latent_space,
        reviews["rating"],
        test_size=rating_model["test_size"],
        random_state=rating_model["random_state"],
    )
    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(X_train, y_train)
    metrics = pd.DataFrame({"r2_train": [model.score(X_train, y_train)], "r2_test": [model.score(X_test, y_test)]})
    return model, metrics
