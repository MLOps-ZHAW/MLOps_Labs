import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_gbq
from datasets import load_dataset
from google.cloud import bigquery
from google.oauth2 import service_account


if __name__ == "__main__":
    size = 2000
    dataset = load_dataset("ag_news")
    subset = dataset["train"].select(range(size))
    df = pd.DataFrame(subset)

    label_mapping = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    df["label"] = df["label"].map(label_mapping)

    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=3)
    total_seconds = int((end_date - start_date).total_seconds())
    random_seconds = np.random.randint(0, total_seconds, size=size)
    timestamps = [start_date + datetime.timedelta(seconds=int(s)) for s in random_seconds]

    df["timestamp"] = timestamps
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(df.head())

    config_path = Path(__file__).parent / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    project_id = config["project_id"]

    location = config["region"]
    service_account_file = "/app/credential.json"
    credentials = service_account.Credentials.from_service_account_file(service_account_file)
    client = bigquery.Client(project=project_id, credentials=credentials, location=location)
    pandas_gbq.to_gbq(
        df,
        project_id=project_id,
        destination_table=f"{config['dataset_id']}.{config['table_id']}",
        location=location,
        if_exists="replace",
        credentials=credentials,
        bigquery_client=client,
    )
