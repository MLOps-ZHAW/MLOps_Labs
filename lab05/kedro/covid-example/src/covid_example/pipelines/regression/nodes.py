import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def clean(covid_raw: pd.DataFrame) -> pd.DataFrame:
    """Convert the `Cases` and `Tests` columns to numbers.

    The raw CSV writes large numbers with thousands separators (e.g. "2,431"),
    so pandas reads them as strings. The commas are removed and the columns
    are parsed as numeric; all other columns are left unchanged.
    """
    data = covid_raw.copy()
    for col in ["Cases", "Tests"]:
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", ""))
    return data


def split(covid_clean: pd.DataFrame, test_size: float, random_state: int):
    """Split the data into a training and a test set.

    The feature `X` is the daily number of `Tests`, the target `y` the daily
    number of positive `Cases`, both as column vectors (shape `(n, 1)`) as
    scikit-learn expects. Returns `X_train, X_test, y_train, y_test`.
    """
    X = covid_clean["Tests"].to_numpy().reshape(-1, 1)
    y = covid_clean["Cases"].to_numpy().reshape(-1, 1)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """Fit a linear regression `Cases = coef * Tests + intercept` on the training set."""
    return LinearRegression().fit(X_train, y_train)


def report(regressor: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    """Summarise the fitted model in a one-row table.

    Contains the slope (`coef`), the `intercept` and the coefficient of
    determination R² of the model on the test set (`r2_test`).
    """
    return pd.DataFrame(
        {
            "coef": regressor.coef_.ravel(),
            "intercept": regressor.intercept_.ravel(),
            "r2_test": regressor.score(X_test, y_test),
        }
    )
