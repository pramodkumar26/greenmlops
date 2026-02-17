from airflow import DAG
from datetime import datetime   
from airflow.operators.python import PythonOperator


def hello():
    print("Hello, Airflow!")


with DAG('my_first_dag', start_date=datetime(2024, 2, 16), schedule='@daily') as dag:
    task1 = PythonOperator(
        task_id='hello_task',
        python_callable=hello
    )