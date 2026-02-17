from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def load_airquality_data():
    df = pd.read_csv('/usr/local/airflow/include/airquality.csv')
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def check_airquality_data():
    df = pd.read_csv('/usr/local/airflow/include/airquality.csv')
    missing = df.isnull().sum().sum()
    total_readings = len(df)
    pm25_stats = df['PM2.5'].agg(['min', 'max', 'mean'])  
    date_min = df['datetime'].min()
    date_max = df['datetime'].max()
    date_range = f"{date_min} to {date_max}"  

    print(f"Missing values: {missing}")
    print(f"Total readings: {total_readings}")
    print(f"PM2.5 stats: {pm25_stats.to_dict()}")
    print(f"Date range: {date_range}")


def detect_airquality_drift():
    df = pd.read_csv('/usr/local/airflow/include/airquality.csv')
    df = df.drop(columns=['datetime'])
    
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
    print(f" LOW-MEDIUM urgency - wildfire events cause sudden drift")

    
def save_airquality():
    df = pd.read_csv('/usr/local/airflow/include/airquality.csv')
    df.to_csv('/usr/local/airflow/include/airquality_clean.csv', index=False)
    print(f" Saved {len(df)} rows to airquality_clean.csv")



with DAG(
    'airquality_etl',
    start_date=datetime(2026, 2, 16),
    schedule='@daily',
    catchup=False
) as dag:
    
    load = PythonOperator(task_id='load_airquality_data', python_callable=load_airquality_data)
    check = PythonOperator(task_id='check_airquality_data', python_callable=check_airquality_data)
    detect = PythonOperator(task_id='detect_drift', python_callable=detect_airquality_drift)
    save = PythonOperator(task_id='save_clean_data', python_callable=save_airquality)


    load >> check >> detect >> save