from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timezone
from pathlib import Path
import sys
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "pramodkumar26"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "158512e86a965be54f8cd213e33bb5d0cb54a0c3"

sys.path.insert(0, "/usr/local/airflow/include/src")
sys.path.insert(0, "/usr/local/airflow/include/src/carbon")

import mlflow
import pandas as pd

from carbon.experiment_schema import DriftCheckRecord, RetrainingEventRecord

FRAUD_PROCESSED  = Path("/usr/local/airflow/include/data/processed/fraud")
CAISO_CSV        = "/usr/local/airflow/include/data/raw/carbon/caiso_2024_hourly.csv"
MLFLOW_URI       = "https://dagshub.com/pramodkumar26/greenmlops.mlflow"
DATASET          = "fraud"
APPROACH         = "carbon_aware"
SEED             = 0
REFERENCE_SIZE   = 595
CURRENT_SIZE     = 256

FRAUD_INJECTION_DAYS = {
    4, 7, 10, 14, 17, 20, 23, 26, 29, 32,
    35, 38, 41, 44, 47, 50, 52, 55, 57, 59
}


def check_drift(**context):
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric
    from dataclasses import asdict

    sim_day = context["dag_run"].conf.get("sim_day", 0)

    train_df = pd.read_csv(FRAUD_PROCESSED / "train.csv")
    feature_cols = [c for c in train_df.columns if c != "Class"]

    reference = train_df.iloc[:REFERENCE_SIZE][feature_cols]

    window_start = REFERENCE_SIZE + sim_day * CURRENT_SIZE
    current = train_df.iloc[window_start: window_start + CURRENT_SIZE][feature_cols].copy()

    if sim_day in FRAUD_INJECTION_DAYS:
        ref_std = reference.std()
        current = current + 1.5 * ref_std

    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference, current_data=current)
    report_result = report.as_dict()

    drift_result = report_result["metrics"][0]["result"]
    drift_detected = drift_result["dataset_drift"]
    drift_score = float(drift_result["share_of_drifted_columns"])

    caiso_df = pd.read_csv(CAISO_CSV)
    carbon_at_check = float(
        caiso_df.iloc[sim_day % len(caiso_df)]["Carbon intensity gCO\u2082eq/kWh (direct)"]
    )

    retraining_triggered = drift_detected

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("drift_detection")
    with mlflow.start_run(
        run_name=f"drift_check_{DATASET}_day{sim_day}",
        nested=True,
    ):
        record = DriftCheckRecord(
            timestamp_day=sim_day,
            dataset=DATASET,
            drift_score=round(drift_score, 4),
            drift_detected=drift_detected,
            days_since_last_retrain=None,
            retraining_triggered=retraining_triggered,
            accuracy_on_new_distribution=-1.0,
            carbon_intensity_at_check=carbon_at_check,
        )
        mlflow.log_params({k: str(v) for k, v in asdict(record).items()})

    context["ti"].xcom_push(key="drift_result", value={
        "sim_day": sim_day,
        "drift_detected": drift_detected,
        "drift_score": round(drift_score, 4),
        "retraining_triggered": retraining_triggered,
        "carbon_at_check": carbon_at_check,
    })


def execute_retraining(**context):
    from dataclasses import asdict
    from codecarbon import EmissionsTracker

    drift_result = context["ti"].xcom_pull(
        key="drift_result", task_ids="check_drift"
    )

    if not drift_result["retraining_triggered"]:
        return

    sim_day = drift_result["sim_day"]
    t0 = datetime(2024, 1, 1 + (sim_day % 28), tzinfo=timezone.utc)

    os.makedirs("/usr/local/airflow/include/codecarbon", exist_ok=True)
    emissions_tracker = EmissionsTracker(
        project_name="greenmlops",
        output_dir="/usr/local/airflow/include/codecarbon",
        save_to_file=True,
    )
    emissions_tracker.start()

    import time
    time.sleep(1)

    emissions = emissions_tracker.stop()
    energy_kwh = float(getattr(emissions, "energy_consumed", 0.0) or 0.0)

    carbon_immediate = drift_result["carbon_at_check"] * energy_kwh
    carbon_scheduled = carbon_immediate

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("drift_detection")
    with mlflow.start_run(
        run_name=f"retrain_{DATASET}_day{sim_day}",
        nested=True,
    ):
        event = RetrainingEventRecord(
            t0=t0,
            t_star=t0,
            dataset=DATASET,
            urgency_class="CRITICAL",
            carbon_intensity_at_t0=drift_result["carbon_at_check"],
            carbon_intensity_at_t_star=drift_result["carbon_at_check"],
            energy_kwh=energy_kwh,
            carbon_immediate=carbon_immediate,
            carbon_scheduled=carbon_scheduled,
            carbon_saved_pct=0.0,
            wait_hours=0.0,
            accuracy_during_wait=[],
            accuracy_post_retrain=-1.0,
            policy_applied="immediate_critical",
            delta_max_pct=0.0,
            seed=SEED,
            approach=APPROACH,
        )
        mlflow.log_params({k: str(v) for k, v in event.to_flat_dict().items()})
        mlflow.log_metric("carbon_saved_pct", 0.0)
        mlflow.log_metric("energy_kwh", energy_kwh)


with DAG(
    dag_id="drift_check_fraud",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "drift", "fraud"],
) as dag:

    t1 = PythonOperator(task_id="check_drift",        python_callable=check_drift)
    t2 = PythonOperator(task_id="execute_retraining", python_callable=execute_retraining)

    t1 >> t2