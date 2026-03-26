from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import os
import sys

sys.path.insert(0, "/usr/local/airflow/include/src")
sys.path.insert(0, "/usr/local/airflow/include/src/carbon")

DATASET         = "ett"
SEED            = 0
SIM_DAYS        = 60
RETRAIN_EVERY_N = 7

# Empirical measurement from Week 3 baseline training -- MLflow run: lstm_ett_baseline
# Config: SEQ_LEN=48, HIDDEN_DIM=128, EPOCHS=30, BATCH_SIZE=32
ETT_ENERGY_KWH  = 0.022

ETT_PROCESSED = "/usr/local/airflow/include/data/processed/ett"
CAISO_CSV     = "/usr/local/airflow/include/data/raw/carbon/caiso_2024_hourly.csv"
COOLDOWN_DIR  = "/usr/local/airflow/include/cooldown"
MLFLOW_URI    = "https://dagshub.com/pramodkumar26/greenmlops.mlflow"
EXPERIMENT    = "ett_simulation"

SAMPLES_PER_DAY = 85
REFERENCE_DAYS  = 7
ROLLING_DAYS    = 3
FEATURE_EXCLUDE = {"OT", "date"}

ETT_INJECTION_DAYS = {
    4, 7, 10, 14, 17, 20, 23, 26, 29, 32,
    35, 38, 41, 44, 47, 50, 52, 55, 57, 59,
}

_TS_VARIANTS = [
    "Datetime (UTC)", "datetime (utc)", "datetime_utc", "Date", "date", "timestamp",
]
_CI_VARIANTS = [
    "Carbon intensity gCO\u2082eq/kWh (direct)",
    "Carbon intensity gCO2eq/kWh (direct)",
    "carbon intensity gco2eq/kwh (direct)",
    "carbon_intensity_direct",
    "Carbon Intensity gCO2eq/kWh (direct)",
]


def _match_column(columns, variants, label):
    normalized = {c.strip().lower(): c for c in columns}
    for v in variants:
        if v in columns:
            return v
        if v.strip().lower() in normalized:
            return normalized[v.strip().lower()]
    raise ValueError(
        f"Could not find {label} column. Expected one of: {variants}. Found: {columns}"
    )


def load_caiso(csv_path):
    import pandas as pd
    df   = pd.read_csv(csv_path)
    cols = list(df.columns)
    ts_col = _match_column(cols, _TS_VARIANTS, "timestamp")
    ci_col = _match_column(cols, _CI_VARIANTS, "carbon intensity")
    df = df[[ts_col, ci_col]].copy()
    df.columns = ["timestamp", "carbon_intensity"]
    df["timestamp"]        = pd.to_datetime(df["timestamp"], utc=True)
    df["carbon_intensity"] = pd.to_numeric(df["carbon_intensity"], errors="coerce")
    df = df.dropna(subset=["carbon_intensity"]).sort_values("timestamp").reset_index(drop=True)
    return df


def sim_day_to_datetime(sim_day, caiso_df):
    from datetime import timedelta
    first_date = caiso_df["timestamp"].iloc[0].normalize()
    target     = first_date + timedelta(days=sim_day, hours=12)
    return target.to_pydatetime()


def get_carbon_at_day(sim_day, caiso_df):
    target_date = sim_day_to_datetime(sim_day, caiso_df).date()
    rows        = caiso_df[caiso_df["timestamp"].dt.date == target_date]
    if rows.empty:
        return float(caiso_df["carbon_intensity"].mean())
    return float(rows["carbon_intensity"].mean())


def get_reference_window(train_df, feature_cols):
    return train_df.iloc[: REFERENCE_DAYS * SAMPLES_PER_DAY][feature_cols].copy()


def get_rolling_window(train_df, feature_cols, sim_day, reference):
    window_start = sim_day * SAMPLES_PER_DAY
    current = train_df.iloc[
        window_start: window_start + ROLLING_DAYS * SAMPLES_PER_DAY
    ][feature_cols].copy()
    if sim_day in ETT_INJECTION_DAYS:
        current = current + 1.5 * reference.std()
    return current


def run_psi_drift_check(reference, current):
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric
    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference, current_data=current)
    result         = report.as_dict()["metrics"][0]["result"]
    drift_detected = bool(result["dataset_drift"])
    drift_score    = round(float(result["share_of_drifted_columns"]), 4)
    return drift_detected, drift_score


def schedule_periodic(day, caiso_df, scheduler, retrain_days):
    if day not in retrain_days:
        return None
    carbon = get_carbon_at_day(day, caiso_df)
    t0     = sim_day_to_datetime(day, caiso_df)
    return dict(
        t0=t0,
        t_star=t0,
        carbon_at_t0=carbon,
        carbon_at_t_star=carbon,
        wait_hours=0.0,
        carbon_saved_pct=0.0,
        policy_applied="periodic_fixed_schedule",
        delta_max_pct=0.0,
    )


def schedule_drift_immediate(day, caiso_df, scheduler, retrain_days=None):
    carbon = get_carbon_at_day(day, caiso_df)
    t0     = sim_day_to_datetime(day, caiso_df)
    return dict(
        t0=t0,
        t_star=t0,
        carbon_at_t0=carbon,
        carbon_at_t_star=carbon,
        wait_hours=0.0,
        carbon_saved_pct=0.0,
        policy_applied="drift_immediate",
        delta_max_pct=3.0,
    )


def schedule_carbon_aware(day, caiso_df, scheduler, retrain_days=None):
    t0     = sim_day_to_datetime(day, caiso_df)
    result = scheduler.schedule_for_dataset(t0=t0, dataset_name=DATASET)
    return dict(
        t0=result["t0"],
        t_star=result["t_star"],
        carbon_at_t0=result["carbon_intensity_at_t0"],
        carbon_at_t_star=result["carbon_intensity_at_t_star"],
        wait_hours=result["wait_hours"],
        carbon_saved_pct=result["carbon_saved_pct"],
        policy_applied=result["policy"],
        delta_max_pct=result["delta_max_pct"],
    )


def run_simulation(approach, scheduling_fn, use_drift_check):
    import mlflow
    import pandas as pd
    from dataclasses import asdict
    from carbon.carbon_scheduler import CarbonScheduler
    from carbon.cooldown_tracker import CooldownTracker
    from carbon.experiment_schema import DriftCheckRecord, RetrainingEventRecord

    os.environ["MLFLOW_TRACKING_USERNAME"] = "pramodkumar26"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "158512e86a965be54f8cd213e33bb5d0cb54a0c3"

    train_df     = pd.read_csv(os.path.join(ETT_PROCESSED, "train.csv"))
    feature_cols = [c for c in train_df.columns if c not in FEATURE_EXCLUDE]
    reference    = get_reference_window(train_df, feature_cols)
    caiso_df     = load_caiso(CAISO_CSV)
    scheduler    = CarbonScheduler(caiso_csv_path=CAISO_CSV)
    retrain_days = set(range(RETRAIN_EVERY_N - 1, SIM_DAYS, RETRAIN_EVERY_N))

    tracker = CooldownTracker(state_dir=COOLDOWN_DIR, run_id=f"{approach}_seed{SEED}")
    tracker.reset()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    total_carbon_immediate = 0.0
    total_carbon_scheduled = 0.0
    retrain_count          = 0

    with mlflow.start_run(run_name=f"{approach}_seed{SEED}"):
        mlflow.log_param("approach",   approach)
        mlflow.log_param("seed",       SEED)
        mlflow.log_param("sim_days",   SIM_DAYS)
        mlflow.log_param("energy_kwh", ETT_ENERGY_KWH)

        for day in range(SIM_DAYS):

            if use_drift_check:
                carbon  = get_carbon_at_day(day, caiso_df)
                current = get_rolling_window(train_df, feature_cols, day, reference)

                drift_detected, drift_score = run_psi_drift_check(reference, current)

                days_since = tracker.days_since_retrain(DATASET, day)
                eligible   = tracker.is_eligible(DATASET, day)
                triggered  = drift_detected and eligible

                check_record = DriftCheckRecord(
                    timestamp_day=day,
                    dataset=DATASET,
                    drift_score=drift_score,
                    drift_detected=drift_detected,
                    days_since_last_retrain=days_since,
                    retraining_triggered=triggered,
                    accuracy_on_new_distribution=-1.0,
                    carbon_intensity_at_check=carbon,
                )
                with mlflow.start_run(run_name=f"{approach}_drift_day{day}", nested=True):
                    mlflow.log_params({k: str(v) for k, v in asdict(check_record).items()})

                if not triggered:
                    continue

            sched = scheduling_fn(day, caiso_df, scheduler, retrain_days)
            if sched is None:
                continue

            carbon_immediate = sched["carbon_at_t0"]     * ETT_ENERGY_KWH
            carbon_scheduled = sched["carbon_at_t_star"] * ETT_ENERGY_KWH

            event = RetrainingEventRecord(
                t0=sched["t0"],
                t_star=sched["t_star"],
                dataset=DATASET,
                urgency_class="LOW",
                carbon_intensity_at_t0=round(sched["carbon_at_t0"], 2),
                carbon_intensity_at_t_star=round(sched["carbon_at_t_star"], 2),
                energy_kwh=ETT_ENERGY_KWH,
                carbon_immediate=round(carbon_immediate, 6),
                carbon_scheduled=round(carbon_scheduled, 6),
                carbon_saved_pct=round(sched["carbon_saved_pct"], 2),
                wait_hours=round(sched["wait_hours"], 2),
                accuracy_during_wait=[],
                accuracy_post_retrain=-1.0,
                policy_applied=sched["policy_applied"],
                delta_max_pct=sched["delta_max_pct"],
                seed=SEED,
                approach=approach,
            )

            with mlflow.start_run(run_name=f"{approach}_retrain_day{day}", nested=True):
                mlflow.log_params({k: str(v) for k, v in event.to_flat_dict().items()})
                mlflow.log_metric("carbon_saved_pct", event.carbon_saved_pct)
                mlflow.log_metric("energy_kwh",       event.energy_kwh)
                mlflow.log_metric("wait_hours",       event.wait_hours)

            tracker.record_retrain(DATASET, day)

            total_carbon_immediate += carbon_immediate
            total_carbon_scheduled += carbon_scheduled
            retrain_count          += 1

        if retrain_count > 0:
            aggregate_saved_pct = (
                (total_carbon_immediate - total_carbon_scheduled)
                / total_carbon_immediate * 100.0
            )
            mlflow.log_metric("total_carbon_immediate_gco2", round(total_carbon_immediate, 6))
            mlflow.log_metric("total_carbon_scheduled_gco2", round(total_carbon_scheduled, 6))
            mlflow.log_metric("retrain_count",               retrain_count)
            mlflow.log_metric("aggregate_carbon_saved_pct",  round(aggregate_saved_pct, 2))


def run_periodic(**context):
    run_simulation("periodic",        schedule_periodic,        use_drift_check=False)


def run_drift_immediate(**context):
    run_simulation("drift_immediate", schedule_drift_immediate, use_drift_check=True)


def run_carbon_aware(**context):
    run_simulation("carbon_aware",    schedule_carbon_aware,    use_drift_check=True)


with DAG(
    dag_id="ett_simulation",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "simulation", "ett"],
) as dag:
    t1 = PythonOperator(task_id="periodic",        python_callable=run_periodic)
    t2 = PythonOperator(task_id="drift_immediate",  python_callable=run_drift_immediate)
    t3 = PythonOperator(task_id="carbon_aware",     python_callable=run_carbon_aware)

    t1 >> t2 >> t3