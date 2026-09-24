import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def clean(covid_raw: pd.DataFrame) -> pd.DataFrame:
    data = covid_raw.copy()
    for col in ["Cases", "Tests"]:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", ""))
    return data


def split(covid_clean: pd.DataFrame, test_size: float, random_state: int):
    X = covid_clean["Tests"].to_numpy().reshape(-1, 1)
    y = covid_clean["Cases"].to_numpy().reshape(-1, 1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    return LinearRegression().fit(X_train, y_train)


def report(regressor: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "coef": regressor.coef_.ravel(),
            "intercept": regressor.intercept_.ravel(),
            "r2_test": regressor.score(X_test, y_test),
        }
    )
