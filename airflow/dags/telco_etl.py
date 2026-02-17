from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def load_telco():
    df = pd.read_csv('/usr/local/airflow/include/telco_churn_cleaned.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return len(df)

def check_quality():
    df = pd.read_csv('/usr/local/airflow/include/telco_churn_cleaned.csv')
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")


def detect_drift():
    df = pd.read_csv('/usr/local/airflow/include/telco_churn_cleaned.csv')
    split = int(len(df) * 0.7)
    reference = df[:split]
    current = df[split:]
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    
    result = report.as_dict()
    drift_detected = result['metrics'][0]['result']['dataset_drift']
    drift_share = result['metrics'][0]['result']['share_of_drifted_columns']
    
    print(f" Drift detected: {drift_detected}")
    print(f" Share of drifted columns: {drift_share:.2%}")





with DAG(
    'telco_etl_pipeline',
    start_date=datetime(2024, 2, 16),
    schedule=None,
    catchup=False
) as dag:
    
    task1 = PythonOperator(task_id='load_data', python_callable=load_telco)
    task2 = PythonOperator(task_id='check_quality', python_callable=check_quality)
    task3 = PythonOperator(task_id='detect_drift', python_callable=detect_drift)
    
    task1 >> task2 >> task3