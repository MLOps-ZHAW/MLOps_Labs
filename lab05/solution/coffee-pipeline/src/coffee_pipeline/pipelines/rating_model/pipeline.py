from kedro.pipeline import Node, Pipeline

from .nodes import train_rating_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                train_rating_model,
                inputs=["shared_latent_space", "reviews", "params:rating_model"],
                outputs=["rating_model", "rating_metrics"],
                name="train_rating_model",
            ),
        ]
    )
