from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH      = Path("/usr/local/airflow/include/data/raw/ett/ETTh1.csv")
PROCESSED_DIR = Path("/usr/local/airflow/include/data/processed/ett")


def load_ett():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])
    print(f"Shape      : {df.shape}")
    print(f"Date range : {df['date'].min()} to {df['date'].max()}")
    print(f"Nulls      : {df.isnull().sum().sum()}")
    print(f"Columns    : {df.columns.tolist()}")


def clean_ett():
    df = pd.read_csv(RAW_PATH, parse_dates=["date"])

    df = df.sort_values("date").reset_index(drop=True)
    df = df.set_index("date").asfreq("h", method="ffill").reset_index()

    assert df.isnull().sum().sum() == 0, "Nulls remain after cleaning"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "ett_clean.csv", index=False)

    print(f"Clean shape : {df.shape}")
    print(f"Saved to    : {PROCESSED_DIR}/ett_clean.csv")


def split_ett():
    df = pd.read_csv(PROCESSED_DIR / "ett_clean.csv", parse_dates=["date"])

    n         = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end].reset_index(drop=True)
    val   = df.iloc[train_end:val_end].reset_index(drop=True)
    test  = df.iloc[val_end:].reset_index(drop=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR   / "val.csv",   index=False)
    test.to_csv(PROCESSED_DIR  / "test.csv",  index=False)

    print(f"Train : {len(train)} rows | {train['date'].min()} to {train['date'].max()}")
    print(f"Val   : {len(val)} rows   | {val['date'].min()} to {val['date'].max()}")
    print(f"Test  : {len(test)} rows  | {test['date'].min()} to {test['date'].max()}")


with DAG(
    dag_id="ett_etl",
    start_date=datetime(2024, 2, 16),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "etl", "ett"]
) as dag:

    t1 = PythonOperator(task_id="load_ett",  python_callable=load_ett)
    t2 = PythonOperator(task_id="clean_ett", python_callable=clean_ett)
    t3 = PythonOperator(task_id="split_ett", python_callable=split_ett)

    t1 >> t2 >> t3