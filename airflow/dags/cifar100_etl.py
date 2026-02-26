from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision


RAW_DIR       = Path("/usr/local/airflow/include/data/raw/cifar100")
PROCESSED_DIR = Path("/usr/local/airflow/include/data/processed/cifar100")


def load_cifar100():
    train_dataset = torchvision.datasets.CIFAR100(
        root=str(RAW_DIR), train=True, download=False
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=str(RAW_DIR), train=False, download=False
    )

    print(f"Train samples : {len(train_dataset)}")
    print(f"Test samples  : {len(test_dataset)}")
    print(f"Num classes   : {len(train_dataset.classes)}")
    print(f"Image size    : {train_dataset[0][0].size}")


def save_cifar100_metadata():
    train_dataset = torchvision.datasets.CIFAR100(
        root=str(RAW_DIR), train=True, download=False
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=str(RAW_DIR), train=False, download=False
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    class_df = pd.DataFrame({
        "class_idx"  : range(100),
        "class_name" : train_dataset.classes
    })
    class_df.to_csv(PROCESSED_DIR / "classes.csv", index=False)

    train_labels = [label for _, label in train_dataset]
    test_labels  = [label for _, label in test_dataset]

    np.save(PROCESSED_DIR / "train_labels.npy", np.array(train_labels))
    np.save(PROCESSED_DIR / "test_labels.npy",  np.array(test_labels))

    print(f"Saved classes.csv       : 100 classes")
    print(f"Saved train_labels.npy  : {len(train_labels)} labels")
    print(f"Saved test_labels.npy   : {len(test_labels)} labels")


def verify_cifar100():
    classes   = pd.read_csv(PROCESSED_DIR / "classes.csv")
    train_lbl = np.load(PROCESSED_DIR / "train_labels.npy")
    test_lbl  = np.load(PROCESSED_DIR / "test_labels.npy")

    counts = np.bincount(train_lbl)

    print(f"Classes loaded    : {len(classes)}")
    print(f"Train labels      : {len(train_lbl)}")
    print(f"Test labels       : {len(test_lbl)}")
    print(f"Min per class     : {counts.min()}")
    print(f"Max per class     : {counts.max()}")
    print(f"Perfectly balanced: {counts.min() == counts.max()}")


with DAG(
    dag_id="cifar100_etl",
    start_date=datetime(2024, 2, 16),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "etl", "cifar100"]
) as dag:

    t1 = PythonOperator(task_id="load_cifar100",          python_callable=load_cifar100)
    t2 = PythonOperator(task_id="save_cifar100_metadata", python_callable=save_cifar100_metadata)
    t3 = PythonOperator(task_id="verify_cifar100",        python_callable=verify_cifar100)

    t1 >> t2 >> t3