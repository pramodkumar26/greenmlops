from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def load_nycdata():
    df = pd.read_csv('/usr/local/airflow/include/nyc_taxi.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def check_nyc_quality():
    df = pd.read_csv('/usr/local/airflow/include/nyc_taxi.csv')
    missing = df.isnull().sum().sum()
    total_trips = len(df)

    start_date = df['tpep_pickup_datetime'].min()  
    end_date = df['tpep_pickup_datetime'].max()
    avg_duration = df['trip_duration'].mean()


    print(f"Missing values: {missing}")
    print(f"Total trips: {total_trips}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Average trip duration: {avg_duration:.2f} seconds")

def detect_nyc_drift():
    df = pd.read_csv('/usr/local/airflow/include/nyc_taxi.csv')
    
    # Drop datetime - not needed for drift detection
    df = df.drop(columns=['tpep_pickup_datetime'])
    
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
    print(f" LOW urgency - can wait up to 24hrs for clean energy window")

def save_nyc():
    df = pd.read_csv('/usr/local/airflow/include/nyc_taxi.csv')
    df.to_csv('/usr/local/airflow/include/nyc_taxi_clean.csv', index=False)
    print(f" Saved {len(df)} rows to nyc_taxi_clean.csv")

with DAG(
    dag_id="nyctaxi_etl",
    start_date=datetime(2026, 2, 16),
    schedule=None,  
    catchup=False
) as dag:
    

    task1 = PythonOperator(task_id='load_nycdata', python_callable=load_nycdata)
    task2 = PythonOperator(task_id='check_nyc_quality', python_callable=check_nyc_quality)
    task3 = PythonOperator(task_id='detect_drift', python_callable=detect_nyc_drift)
    task4 = PythonOperator(task_id='save_clean_data', python_callable=save_nyc)

    task1 >> task2 >> task3 >> task4