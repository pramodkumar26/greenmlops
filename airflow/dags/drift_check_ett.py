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

from carbon.carbon_scheduler import CarbonScheduler
from carbon.cooldown_tracker import CooldownTracker
from carbon.experiment_schema import DriftCheckRecord, RetrainingEventRecord

ETT_PROCESSED   = Path("/usr/local/airflow/include/data/processed/ett")
CAISO_CSV       = "/usr/local/airflow/include/data/raw/carbon/caiso_2024_hourly.csv"
COOLDOWN_DIR    = "/usr/local/airflow/include/cooldown"
MLFLOW_URI      = "https://dagshub.com/pramodkumar26/greenmlops.mlflow"
DATASET         = "ett"
APPROACH        = "carbon_aware"
SEED            = 0
SAMPLES_PER_DAY = 85

ETT_INJECTION_DAYS = {
    4, 7, 10, 14, 17, 20, 23, 26, 29, 32,
    35, 38, 41, 44, 47, 50, 52, 55, 57, 59
}


def check_drift(**context):
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric
    from dataclasses import asdict

    sim_day = context["dag_run"].conf.get("sim_day", 0)

    train_df = pd.read_csv(ETT_PROCESSED / "train.csv")
    feature_cols = [c for c in train_df.columns if c not in ["OT", "date"]]
    reference = train_df.iloc[: 7 * SAMPLES_PER_DAY][feature_cols]

    window_start = sim_day * SAMPLES_PER_DAY
    current = train_df.iloc[window_start: window_start + 3 * SAMPLES_PER_DAY][feature_cols].copy()

    if sim_day in ETT_INJECTION_DAYS:
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

    tracker = CooldownTracker(state_dir=COOLDOWN_DIR, run_id=f"seed_{SEED}")
    days_since = tracker.days_since_retrain(DATASET, sim_day)
    eligible = tracker.is_eligible(DATASET, sim_day)
    retraining_triggered = drift_detected and eligible

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
            days_since_last_retrain=days_since,
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


def schedule_retraining(**context):
    drift_result = context["ti"].xcom_pull(
        key="drift_result", task_ids="check_drift"
    )

    if not drift_result["retraining_triggered"]:
        context["ti"].xcom_push(key="schedule_result", value={"skip": True})
        return

    sim_day = drift_result["sim_day"]
    t0 = datetime(2024, 1, 1 + (sim_day % 28), tzinfo=timezone.utc)

    scheduler = CarbonScheduler(caiso_csv_path=CAISO_CSV)
    result = scheduler.schedule_for_dataset(
        t0=t0,
        dataset_name=DATASET,
    )

    context["ti"].xcom_push(key="schedule_result", value={
        "skip": False,
        "sim_day": sim_day,
        "t0": result["t0"].isoformat(),
        "t_star": result["t_star"].isoformat(),
        "carbon_at_t0": result["carbon_intensity_at_t0"],
        "carbon_at_t_star": result["carbon_intensity_at_t_star"],
        "carbon_saved_pct": result["carbon_saved_pct"],
        "wait_hours": result["wait_hours"],
        "policy": result["policy"],
        "delta_max_pct": result["delta_max_pct"],
    })


def execute_retraining(**context):
    from dataclasses import asdict
    from codecarbon import EmissionsTracker

    sched = context["ti"].xcom_pull(
        key="schedule_result", task_ids="schedule_retraining"
    )

    if sched.get("skip"):
        return

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

    carbon_immediate = sched["carbon_at_t0"] * energy_kwh
    carbon_scheduled = sched["carbon_at_t_star"] * energy_kwh

    cooldown_tracker = CooldownTracker(
        state_dir=COOLDOWN_DIR, run_id=f"seed_{SEED}"
    )
    cooldown_tracker.record_retrain(DATASET, sched["sim_day"])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("drift_detection")
    with mlflow.start_run(
        run_name=f"retrain_{DATASET}_day{sched['sim_day']}",
        nested=True,
    ):
        event = RetrainingEventRecord(
            t0=datetime.fromisoformat(sched["t0"]),
            t_star=datetime.fromisoformat(sched["t_star"]),
            dataset=DATASET,
            urgency_class="LOW",
            carbon_intensity_at_t0=sched["carbon_at_t0"],
            carbon_intensity_at_t_star=sched["carbon_at_t_star"],
            energy_kwh=energy_kwh,
            carbon_immediate=carbon_immediate,
            carbon_scheduled=carbon_scheduled,
            carbon_saved_pct=sched["carbon_saved_pct"],
            wait_hours=sched["wait_hours"],
            accuracy_during_wait=[],
            accuracy_post_retrain=-1.0,
            policy_applied=sched["policy"],
            delta_max_pct=sched["delta_max_pct"],
            seed=SEED,
            approach=APPROACH,
        )
        mlflow.log_params({k: str(v) for k, v in event.to_flat_dict().items()})
        mlflow.log_metric("carbon_saved_pct", sched["carbon_saved_pct"])
        mlflow.log_metric("energy_kwh", energy_kwh)


with DAG(
    dag_id="drift_check_ett",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "drift", "ett"],
) as dag:

    t1 = PythonOperator(task_id="check_drift",         python_callable=check_drift)
    t2 = PythonOperator(task_id="schedule_retraining", python_callable=schedule_retraining)
    t3 = PythonOperator(task_id="execute_retraining",  python_callable=execute_retraining)

    t1 >> t2 >> t3