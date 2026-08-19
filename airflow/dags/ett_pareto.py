from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import os
import sys
os.environ["MLFLOW_TRACKING_USERNAME"] = "pramodkumar26"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

sys.path.insert(0, "/usr/local/airflow/include/src")
sys.path.insert(0, "/usr/local/airflow/include/src/carbon")

DATASET         = "ett"
SIM_DAYS        = 60
RETRAIN_EVERY_N = 7
ETT_ENERGY_KWH  = 0.022

ETT_PROCESSED = "/usr/local/airflow/include/data/processed/ett"
CAISO_CSV     = "/usr/local/airflow/include/data/raw/carbon/caiso_2024_hourly.csv"
COOLDOWN_DIR  = "/usr/local/airflow/include/cooldown"
MLFLOW_URI    = "https://dagshub.com/pramodkumar26/greenmlops.mlflow"
EXPERIMENT    = "ett_pareto"

SAMPLES_PER_DAY = 85
REFERENCE_DAYS  = 7
ROLLING_DAYS    = 3
FEATURE_EXCLUDE = {"OT", "date"}

WINDOW_CONFIGS = {
    0: "2024-01-01",
    1: "2024-05-01",
    2: "2024-10-15",
}
SEEDS        = [0, 1, 2]
DMAX_VALUES  = [3, 6, 12]

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
    df     = pd.read_csv(csv_path)
    cols   = list(df.columns)
    ts_col = _match_column(cols, _TS_VARIANTS, "timestamp")
    ci_col = _match_column(cols, _CI_VARIANTS, "carbon intensity")
    df     = df[[ts_col, ci_col]].copy()
    df.columns = ["timestamp", "carbon_intensity"]
    df["timestamp"]        = pd.to_datetime(df["timestamp"], utc=True)
    df["carbon_intensity"] = pd.to_numeric(df["carbon_intensity"], errors="coerce")
    df = df.dropna(subset=["carbon_intensity"]).sort_values("timestamp").reset_index(drop=True)
    return df


def sim_day_to_datetime(sim_day, caiso_df, anchor_date):
    import pandas as pd
    from datetime import timedelta
    anchor = pd.Timestamp(anchor_date, tz="UTC")
    target = anchor + timedelta(days=sim_day, hours=12)
    return target.to_pydatetime()


def get_carbon_at_day(sim_day, caiso_df, anchor_date):
    target_date = sim_day_to_datetime(sim_day, caiso_df, anchor_date).date()
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


def run_simulation(window, seed, anchor_date, d_max_hours):
    import mlflow
    import pandas as pd
    from dataclasses import asdict
    from carbon.carbon_scheduler import CarbonScheduler
    from carbon.cooldown_tracker import CooldownTracker
    from carbon.experiment_schema import DriftCheckRecord, RetrainingEventRecord

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

    train_df     = pd.read_csv(os.path.join(ETT_PROCESSED, "train.csv"))
    feature_cols = [c for c in train_df.columns if c not in FEATURE_EXCLUDE]
    reference    = get_reference_window(train_df, feature_cols)
    caiso_df     = load_caiso(CAISO_CSV)
    scheduler    = CarbonScheduler(caiso_csv_path=CAISO_CSV)

    run_id  = f"carbon_aware_dmax{d_max_hours}_window{window}_seed{seed}"
    tracker = CooldownTracker(state_dir=COOLDOWN_DIR, run_id=f"ett_pareto_{run_id}")
    tracker.reset()

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    total_carbon_immediate = 0.0
    total_carbon_scheduled = 0.0
    retrain_count          = 0

    with mlflow.start_run(run_name=run_id):
        mlflow.log_param("approach",    "carbon_aware")
        mlflow.log_param("window",      window)
        mlflow.log_param("anchor_date", anchor_date)
        mlflow.log_param("seed",        seed)
        mlflow.log_param("sim_days",    SIM_DAYS)
        mlflow.log_param("energy_kwh",  ETT_ENERGY_KWH)
        mlflow.log_param("d_max_hours", d_max_hours)
        mlflow.log_param("dataset",     DATASET)

        for day in range(SIM_DAYS):
            carbon  = get_carbon_at_day(day, caiso_df, anchor_date)
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
            with mlflow.start_run(run_name=f"{run_id}_drift_day{day}", nested=True):
                mlflow.log_params({k: str(v) for k, v in asdict(check_record).items()})

            if not triggered:
                continue

            t0     = sim_day_to_datetime(day, caiso_df, anchor_date)
            result = scheduler.schedule(
                t0=t0,
                urgency_class="LOW",
                d_max_hours=d_max_hours,
                dataset_name=DATASET,
            )

            carbon_immediate = result["carbon_intensity_at_t0"]     * ETT_ENERGY_KWH
            carbon_scheduled = result["carbon_intensity_at_t_star"] * ETT_ENERGY_KWH

            event = RetrainingEventRecord(
                t0=result["t0"],
                t_star=result["t_star"],
                dataset=DATASET,
                urgency_class="LOW",
                carbon_intensity_at_t0=round(result["carbon_intensity_at_t0"], 2),
                carbon_intensity_at_t_star=round(result["carbon_intensity_at_t_star"], 2),
                energy_kwh=ETT_ENERGY_KWH,
                carbon_immediate=round(carbon_immediate, 6),
                carbon_scheduled=round(carbon_scheduled, 6),
                carbon_saved_pct=round(result["carbon_saved_pct"], 2),
                wait_hours=round(result["wait_hours"], 2),
                accuracy_during_wait=[],
                accuracy_post_retrain=-1.0,
                policy_applied=result["policy"],
                delta_max_pct=result["delta_max_pct"],
                seed=seed,
                approach="carbon_aware",
            )

            with mlflow.start_run(run_name=f"{run_id}_retrain_day{day}", nested=True):
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
            mlflow.log_metric("d_max_hours",                 d_max_hours)


def run_pareto(**context):
    for d_max_hours in DMAX_VALUES:
        for window, anchor_date in WINDOW_CONFIGS.items():
            for seed in SEEDS:
                run_simulation(window, seed, anchor_date, d_max_hours)


with DAG(
    dag_id="ett_pareto",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "pareto", "ett"],
) as dag:
    t1 = PythonOperator(task_id="run_pareto", python_callable=run_pareto)
