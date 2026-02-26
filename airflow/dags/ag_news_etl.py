from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from pathlib import Path


RAW_DIR       = Path("/usr/local/airflow/include/data/raw/ag_news")
PROCESSED_DIR = Path("/usr/local/airflow/include/data/processed/ag_news")


def load_ag_news():
    from datasets import load_from_disk

    dataset = load_from_disk(str(RAW_DIR))

    train = dataset["train"].to_pandas()
    test  = dataset["test"].to_pandas()

    print(f"Train samples : {len(train)}")
    print(f"Test samples  : {len(test)}")
    print(f"Columns       : {train.columns.tolist()}")
    print(f"Label dist    :\n{train['label'].value_counts().sort_index()}")


def split_ag_news():
    from datasets import load_from_disk

    dataset   = load_from_disk(str(RAW_DIR))
    train_full = dataset["train"].to_pandas()
    test       = dataset["test"].to_pandas()

    n       = len(train_full)
    val_end = int(n * 0.85)

    train = train_full.iloc[:val_end].reset_index(drop=True)
    val   = train_full.iloc[val_end:].reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR   / "val.csv",   index=False)
    test.to_csv(PROCESSED_DIR  / "test.csv",  index=False)

    print(f"Train : {len(train)} rows")
    print(f"Val   : {len(val)} rows")
    print(f"Test  : {len(test)} rows")
    print(f"Train label dist:\n{train['label'].value_counts().sort_index()}")


def verify_ag_news():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val   = pd.read_csv(PROCESSED_DIR / "val.csv")
    test  = pd.read_csv(PROCESSED_DIR / "test.csv")

    assert train.isnull().sum().sum() == 0, "Nulls in train"
    assert val.isnull().sum().sum() == 0,   "Nulls in val"
    assert test.isnull().sum().sum() == 0,  "Nulls in test"

    print(f"Train : {len(train)} rows | nulls: 0")
    print(f"Val   : {len(val)} rows   | nulls: 0")
    print(f"Test  : {len(test)} rows  | nulls: 0")
    print(f"All splits verified clean")


with DAG(
    dag_id="ag_news_etl",
    start_date=datetime(2024, 2, 16),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "etl", "ag_news"]
) as dag:

    t1 = PythonOperator(task_id="load_ag_news",   python_callable=load_ag_news)
    t2 = PythonOperator(task_id="split_ag_news",  python_callable=split_ag_news)
    t3 = PythonOperator(task_id="verify_ag_news", python_callable=verify_ag_news)

    t1 >> t2 >> t3