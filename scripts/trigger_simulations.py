import time
import requests
from datetime import datetime

AIRFLOW_BASE    = "http://localhost:8080/api/v1"
AIRFLOW_USER    = "admin"
AIRFLOW_PASS    = "admin"

# Delays in seconds before triggering each DAG
# ETT is already running -- these start from now
SCHEDULE = [
    {"dag_id": "fraud_simulation",    "delay_seconds":  30 * 60},
    {"dag_id": "cifar100_simulation", "delay_seconds": 140 * 60},
    {"dag_id": "ag_news_simulation",  "delay_seconds": 230 * 60},
]


def trigger_dag(dag_id):
    url      = f"{AIRFLOW_BASE}/dags/{dag_id}/dagRuns"
    payload  = {"conf": {}}
    response = requests.post(
        url,
        json=payload,
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        headers={"Content-Type": "application/json"},
    )
    if response.status_code in (200, 201):
        run_id = response.json().get("dag_run_id", "unknown")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] triggered {dag_id} -- run_id: {run_id}")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ERROR triggering {dag_id}: "
              f"{response.status_code} {response.text}")


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] scheduler started")
    print(f"  fraud_simulation   triggers in 30 min")
    print(f"  cifar100_simulation triggers in 140 min")
    print(f"  ag_news_simulation  triggers in 0 min")
    print()

    start_time = time.time()

    for entry in SCHEDULE:
        dag_id        = entry["dag_id"]
        delay_seconds = entry["delay_seconds"]
        elapsed       = time.time() - start_time
        remaining     = delay_seconds - elapsed

        if remaining > 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] waiting {remaining/60:.1f} min for {dag_id}...")
            time.sleep(remaining)

        trigger_dag(dag_id)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] all DAGs triggered")


if __name__ == "__main__":
    main()