from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path("/usr/local/airflow/include/data/raw/fraud/creditcard.csv")
PROCESSED_DIR = Path("/usr/local/airflow/include/data/processed/fraud")


def load_fraud():
    df = pd.read_csv(RAW_PATH)
    print(f"Shape      : {df.shape}")
    print(f"Columns    : {df.columns.tolist()}")
    print(f"Nulls      : {df.isnull().sum().sum()}")
    print(f"Fraud rate : {df['Class'].mean() * 100:.4f}%")


def clean_fraud():
    df = pd.read_csv(RAW_PATH)

    df = df.drop(columns=["Time"])
    df["Amount"] = np.log1p(df["Amount"])
    df = df.reset_index(drop=True)

    assert df.isnull().sum().sum() == 0, "Nulls remain after cleaning"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "creditcard_clean.csv", index=False)

    print(f"Clean shape : {df.shape}")
    print(f"Saved to    : {PROCESSED_DIR}/creditcard_clean.csv")


def split_fraud():
    df = pd.read_csv(PROCESSED_DIR / "creditcard_clean.csv")

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    train = df.iloc[:train_end].reset_index(drop=True)
    val   = df.iloc[train_end:val_end].reset_index(drop=True)
    test  = df.iloc[val_end:].reset_index(drop=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR   / "val.csv",   index=False)
    test.to_csv(PROCESSED_DIR  / "test.csv",  index=False)

    print(f"Train : {len(train)} rows | Fraud: {train['Class'].sum()}")
    print(f"Val   : {len(val)} rows   | Fraud: {val['Class'].sum()}")
    print(f"Test  : {len(test)} rows  | Fraud: {test['Class'].sum()}")


with DAG(
    dag_id="fraud_etl",
    start_date=datetime(2024, 2, 16),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "etl", "fraud"]
) as dag:

    t1 = PythonOperator(task_id="load_fraud",  python_callable=load_fraud)
    t2 = PythonOperator(task_id="clean_fraud", python_callable=clean_fraud)
    t3 = PythonOperator(task_id="split_fraud", python_callable=split_fraud)

    t1 >> t2 >> t3