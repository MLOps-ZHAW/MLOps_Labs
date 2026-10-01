# covid-example

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

A minimal Kedro project used in [Kedro 101](../kedro_101.md): a linear regression that predicts the number of positive COVID cases from the number of tests ([data source](https://github.com/Ayushijain09/Regression-on-COVID-dataset)).

```shell
conda activate mlops-lab-05
cd lab05/kedro/covid-example

kedro run          # run the pipeline, results end up in data/
kedro viz run      # visualise the pipeline

# export to Airflow and run it once
pip install -e .
kedro airflow create --target-dir ../../airflow/dags
cd ../../airflow && export AIRFLOW_HOME=$(pwd) && cd ../kedro/covid-example
airflow db migrate
airflow dags test covid-example
```

| File | Content |
|:--|:--|
| `conf/base/catalog.yml` | Data Catalog: all datasets of the pipeline |
| `conf/base/parameters.yml` | Parameters (`params:test_size`, `params:random_state`) |
| `conf/base/airflow.yml` | Settings for `kedro airflow create` |
| `src/covid_example/pipelines/regression/` | The nodes (`nodes.py`) and the pipeline (`pipeline.py`) |
