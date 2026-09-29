from datetime import datetime
from pathlib import Path

# The DAG object and the @task decorator; we'll need these to define our workflow.
from airflow.sdk import DAG, task

# Operators; predefined tasks, e.g. to run a bash command
from airflow.providers.standard.operators.bash import BashOperator

# Both tasks also write their output to a file named after the start time of the DAG run (UTC):
# lab05/airflow/output/demo_output_<YYYY-MM-DD_HH-MM-SS>.txt
# Using the start time of the *run* (not the current time) makes sure both tasks use the same file.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"

# A DAG represents a workflow, a collection of tasks
# catchup=False: only run the latest interval, don't backfill every day since start_date
with DAG(dag_id="demo", start_date=datetime(2025, 1, 1), schedule="0 0 * * *", catchup=False) as dag:
    # Tasks are represented as operators
    # `tee` prints "hello" (-> task log) and writes it to the file.
    # {{ dag_run.start_date }} is a Jinja template, filled in by Airflow when the task runs.
    hello = BashOperator(
        task_id="hello",
        bash_command=(
            f"mkdir -p {OUTPUT_DIR} && echo hello | tee "
            f'"{OUTPUT_DIR}/demo_output_{{{{ dag_run.start_date.strftime("{TIME_FORMAT}") }}}}.txt"'
        ),
    )

    # Tasks can also be declared using decorated python functions.
    # This is known as "taskflow".
    # Arguments named like a context variable (here: dag_run) are filled in by Airflow.
    @task()
    def say_airflow(dag_run=None):
        print("airflow")
        # append to the file created by `hello`
        with open(OUTPUT_DIR / f"demo_output_{dag_run.start_date.strftime(TIME_FORMAT)}.txt", "a") as f:
            f.write("airflow\n")

    # Set dependencies between tasks
    hello >> say_airflow()
