from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, task

from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project

# Kedro settings required to run your pipeline
env = "local"
pipeline_name = "__default__"
project_path = Path.cwd()
package_name = "coffee_pipeline"
conf_source = "" or Path.cwd() / "conf"


def _run_kedro_node(node_names=None, namespaces=None):
    """Run a Kedro pipeline slice. Pass node_names for regular node groups or namespaces for namespace groups."""
    configure_project(package_name)
    with KedroSession.create(project_path=project_path, env=env, conf_source=conf_source) as session:
        if namespaces is not None:
            session.run(pipeline_names=[pipeline_name], namespaces=namespaces)
        else:
            session.run(pipeline_names=[pipeline_name], node_names=node_names)


@dag(
    dag_id="coffee-pipeline",
    start_date=datetime(2026,1,1),
    max_active_runs=3,
    schedule="@daily",
    catchup=False,
    default_args=dict(
        owner="airflow",
        depends_on_past=False,
        email_on_failure=False,
        email_on_retry=False,
        retries=1,
        retry_delay=timedelta(minutes=1)
    ),
)
def coffee_pipeline():

    @task(task_id="fit-imputer")
    def fit_imputer():
        _run_kedro_node(node_names=["fit_imputer"])

    @task(task_id="preprocess-reviews")
    def preprocess_reviews():
        _run_kedro_node(node_names=["preprocess_reviews"])

    @task(task_id="embed-descriptions")
    def embed_descriptions():
        _run_kedro_node(node_names=["embed_descriptions"])

    @task(task_id="impute-missing")
    def impute_missing():
        _run_kedro_node(node_names=["impute_missing"])

    @task(task_id="combine-features")
    def combine_features():
        _run_kedro_node(node_names=["combine_features"])

    @task(task_id="train-rating-model")
    def train_rating_model():
        _run_kedro_node(node_names=["train_rating_model"])

    tasks = {
        "fit-imputer": fit_imputer(),
        "preprocess-reviews": preprocess_reviews(),
        "embed-descriptions": embed_descriptions(),
        "impute-missing": impute_missing(),
        "combine-features": combine_features(),
        "train-rating-model": train_rating_model(),
    }
    tasks["preprocess-reviews"] >> tasks["embed-descriptions"]
    tasks["fit-imputer"] >> tasks["impute-missing"]
    tasks["preprocess-reviews"] >> tasks["impute-missing"]
    tasks["embed-descriptions"] >> tasks["combine-features"]
    tasks["impute-missing"] >> tasks["combine-features"]
    tasks["combine-features"] >> tasks["train-rating-model"]
    tasks["preprocess-reviews"] >> tasks["train-rating-model"]


coffee_pipeline()