from kedro.pipeline import Node, Pipeline

from .nodes import clean, report, split, train


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(clean, inputs="covid_raw", outputs="covid_clean", name="clean"),
            Node(
                split,
                inputs=["covid_clean", "params:test_size", "params:random_state"],
                # train_test_split returns 4 values -> 4 outputs
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split",
            ),
            Node(train, inputs=["X_train", "y_train"], outputs="regressor", name="train"),
            Node(report, inputs=["regressor", "X_test", "y_test"], outputs="coefficients", name="report"),
        ]
    )
