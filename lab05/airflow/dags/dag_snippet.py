from datetime import datetime

# The DAG object and the @task decorator; we'll need these to define our workflow.
from airflow.sdk import DAG, task

# Operators; predefined tasks, e.g. to run a bash command
from airflow.providers.standard.operators.bash import BashOperator

# A DAG represents a workflow, a collection of tasks
# catchup=False: only run the latest interval, don't backfill every day since start_date
with DAG(dag_id="demo", start_date=datetime(2022, 1, 1), schedule="0 0 * * *", catchup=False) as dag:
    # Tasks are represented as operators
    hello = BashOperator(task_id="hello", bash_command="echo hello")

    # Tasks can also be declared using decorated python functions.
    # This is known as "taskflow".
    @task()
    def say_airflow():
        print("airflow")

    # Set dependencies between tasks
    hello >> say_airflow()
