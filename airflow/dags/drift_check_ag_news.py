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
sys.path.insert(0, "/usr/local/airflow/include/src/features")

import mlflow
import pandas as pd

from carbon.carbon_scheduler import CarbonScheduler
from carbon.cooldown_tracker import CooldownTracker
from carbon.experiment_schema import DriftCheckRecord, RetrainingEventRecord

EMBEDDINGS_DIR = Path("/usr/local/airflow/include/data/embeddings/ag_news")
CAISO_CSV      = "/usr/local/airflow/include/data/raw/carbon/caiso_2024_hourly.csv"
COOLDOWN_DIR   = "/usr/local/airflow/include/cooldown"
MLFLOW_URI     = "https://dagshub.com/pramodkumar26/greenmlops.mlflow"
DATASET        = "ag_news"
APPROACH       = "carbon_aware"
SEED           = 0

AG_NEWS_INJECTION_DAYS = {
    3, 6, 9, 13, 16, 19, 22, 25, 28, 31,
    34, 37, 40, 43, 46, 49, 51, 54, 57, 59
}


def check_drift(**context):
    import numpy as np
    from dataclasses import asdict
    from features.embedding_drift import EmbeddingDriftDetector

    sim_day = context["dag_run"].conf.get("sim_day", 0)

    ref_embeddings = np.load(EMBEDDINGS_DIR / "ref_embeddings.npy")

    def load_day_embeddings(day, force_clean=False):
        if not force_clean and day in AG_NEWS_INJECTION_DAYS:
            path = EMBEDDINGS_DIR / f"day_{day:02d}_embeddings_injected.npy"
        else:
            path = EMBEDDINGS_DIR / f"day_{day:02d}_embeddings.npy"
        return np.load(path)

    days_to_load = [max(0, sim_day - 2), max(0, sim_day - 1), sim_day]
    days_to_load = list(dict.fromkeys(days_to_load))
    current_embeddings = np.vstack([
        load_day_embeddings(d, force_clean=(d != sim_day))
        for d in days_to_load
    ])

    detector = EmbeddingDriftDetector(rng_seed=42)
    detector.fit(ref_embeddings)
    result = detector.score(current_embeddings)

    drift_detected = result["drift_detected"]
    drift_score    = result["drift_score"]

    caiso_df        = pd.read_csv(CAISO_CSV)
    carbon_at_check = float(
        caiso_df.iloc[sim_day % len(caiso_df)]["Carbon intensity gCO\u2082eq/kWh (direct)"]
    )

    tracker      = CooldownTracker(state_dir=COOLDOWN_DIR, run_id=f"seed_{SEED}")
    days_since   = tracker.days_since_retrain(DATASET, sim_day)
    eligible     = tracker.is_eligible(DATASET, sim_day)
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
            drift_score=drift_score,
            drift_detected=drift_detected,
            days_since_last_retrain=days_since,
            retraining_triggered=retraining_triggered,
            accuracy_on_new_distribution=-1.0,
            carbon_intensity_at_check=carbon_at_check,
        )
        mlflow.log_params({k: str(v) for k, v in asdict(record).items()})
        mlflow.log_metrics({
            "drift_score":     drift_score,
            "carbon_at_check": carbon_at_check,
            "mmd_threshold":   result["mmd_threshold"],
            "n_ks_fail":       float(result["n_components_ks_fail"]),
        })

    context["ti"].xcom_push(key="drift_result", value={
        "sim_day":              sim_day,
        "drift_detected":       drift_detected,
        "drift_score":          drift_score,
        "retraining_triggered": retraining_triggered,
        "carbon_at_check":      carbon_at_check,
    })


def schedule_retraining(**context):
    drift_result = context["ti"].xcom_pull(
        key="drift_result", task_ids="check_drift"
    )

    if not drift_result["retraining_triggered"]:
        context["ti"].xcom_push(key="schedule_result", value={"skip": True})
        return

    sim_day = drift_result["sim_day"]
    t0      = datetime(2024, 1, 1 + (sim_day % 28), sim_day % 24, 0, 0, tzinfo=timezone.utc)

    scheduler = CarbonScheduler(caiso_csv_path=CAISO_CSV)
    result    = scheduler.schedule_for_dataset(
        t0=t0,
        dataset_name=DATASET,
    )

    context["ti"].xcom_push(key="schedule_result", value={
        "skip":             False,
        "sim_day":          sim_day,
        "t0":               result["t0"].isoformat(),
        "t_star":           result["t_star"].isoformat(),
        "carbon_at_t0":     result["carbon_intensity_at_t0"],
        "carbon_at_t_star": result["carbon_intensity_at_t_star"],
        "carbon_saved_pct": result["carbon_saved_pct"],
        "wait_hours":       result["wait_hours"],
        "policy":           result["policy"],
        "delta_max_pct":    result["delta_max_pct"],
    })


def execute_retraining(**context):
    from dataclasses import asdict
    from codecarbon import EmissionsTracker
    import time

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

    time.sleep(1)

    emissions  = emissions_tracker.stop()
    energy_kwh = float(getattr(emissions, "energy_consumed", 0.0) or 0.0)

    carbon_immediate = sched["carbon_at_t0"]     * energy_kwh
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
            urgency_class="MEDIUM",
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
    dag_id="drift_check_ag_news",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "drift", "ag_news"],
) as dag:

    t1 = PythonOperator(task_id="check_drift",         python_callable=check_drift)
    t2 = PythonOperator(task_id="schedule_retraining", python_callable=schedule_retraining)
    t3 = PythonOperator(task_id="execute_retraining",  python_callable=execute_retraining)

    t1 >> t2 >> t3