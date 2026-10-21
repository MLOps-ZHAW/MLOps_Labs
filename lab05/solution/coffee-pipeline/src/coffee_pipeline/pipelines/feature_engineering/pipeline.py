from kedro.pipeline import Node, Pipeline

from .nodes import combine_features, embed_descriptions, fit_imputer, impute_missing, preprocess_reviews


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(preprocess_reviews, inputs="reviews_raw", outputs="reviews", name="preprocess_reviews"),
            # Predict missing fields
            Node(
                fit_imputer,
                inputs=["cqi_raw", "params:feature_columns", "params:missing_columns"],
                outputs="imputer",
                name="fit_imputer",
            ),
            Node(
                impute_missing,
                inputs=["reviews", "imputer", "params:feature_columns", "params:missing_columns"],
                outputs="reviews_imputed",
                name="impute_missing",
            ),
            # Embeddings
            Node(
                embed_descriptions,
                inputs=["reviews", "params:embedding"],
                outputs="description_embeddings",
                name="embed_descriptions",
            ),
            # Shared latent space
            Node(
                combine_features,
                inputs=["reviews_imputed", "description_embeddings", "params:feature_columns", "params:missing_columns"],
                outputs="shared_latent_space",
                name="combine_features",
            ),
        ]
    )
