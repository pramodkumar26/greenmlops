from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset



def load_fraud():
    df = pd.read_csv('/usr/local/airflow/include/creditcard.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return len(df)


def check_fraud_quality():
    df = pd.read_csv('/usr/local/airflow/include/creditcard.csv')
    missing = df.isnull().sum().sum()
    total_transactions = len(df)
    class_counts = df['Class'].value_counts()
    legit_count = class_counts.get(0, 0)
    fraud_count = class_counts.get(1, 0)
    fraud_rate = (fraud_count / total_transactions) * 100

    print(f"Missing values: {missing}")
    print(f"Total transactions: {total_transactions}")
    print(f"Fraud distribution:\n{class_counts}")
    print(f'Fraud rate: {fraud_rate:.4f}%')


def detect_fraud_drift():
    df = pd.read_csv('/usr/local/airflow/include/creditcard.csv')
    
    # Temporal split - important for fraud data
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
    print(f" HIGH urgency dataset - fraud patterns evolve fast")

def save_fraud():
    df = pd.read_csv('/usr/local/airflow/include/creditcard.csv')
    df.to_csv('/usr/local/airflow/include/creditcard_clean.csv', index=False)
    print(f" Saved {len(df)} rows to creditcard_clean.csv")


with DAG(
    dag_id="fraud_etl",
    start_date=datetime(2024, 2, 16),
    schedule=None,  
    catchup=False
) as dag:
    

    task1 = PythonOperator(task_id='load_fraud', python_callable=load_fraud)
    task2 = PythonOperator(task_id='check_fraud_quality', python_callable=check_fraud_quality)
    task3 = PythonOperator(task_id='detect_drift', python_callable=detect_fraud_drift)
    task4 = PythonOperator(task_id='save_clean_data', python_callable=save_fraud)


    task1 >> task2 >> task3 >> task4