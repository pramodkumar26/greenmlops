from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import os
import sys
import numpy as np
os.environ["MLFLOW_TRACKING_USERNAME"] = "pramodkumar26"
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

sys.path.insert(0, "/usr/local/airflow/include/src")
sys.path.insert(0, "/usr/local/airflow/include/src/carbon")

DATASET          = "cifar100"
SIM_DAYS         = 60
CIFAR100_ENERGY_KWH = 0.007853

EMBEDDINGS_DIR = "/usr/local/airflow/include/data/embeddings/cifar100"
CAISO_CSV      = "/usr/local/airflow/include/data/raw/carbon/caiso_2024_hourly.csv"
COOLDOWN_DIR   = "/usr/local/airflow/include/cooldown"
MLFLOW_URI     = "https://dagshub.com/pramodkumar26/greenmlops.mlflow"
EXPERIMENT     = "cifar100_pareto"

REFERENCE_DAYS  = 7
ROLLING_DAYS    = 3
SAMPLES_PER_DAY = 85
N_SAMPLES_MMD   = 256

WINDOW_CONFIGS = {
    0: "2024-01-01",
    1: "2024-05-01",
    2: "2024-10-15",
}
SEEDS       = [0, 1, 2]
DMAX_VALUES = [3, 6, 24]

CIFAR100_INJECTION_DAYS = {
    3, 6, 9, 13, 16, 19, 22, 25, 28, 31,
    34, 37, 40, 43, 46, 49, 51, 54, 57, 59,
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


def load_embeddings(embeddings_dir, sim_day):
    path = os.path.join(embeddings_dir, f"day_{sim_day:02d}_embeddings.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def compute_mmd(ref_pca, cur_pca, bandwidth, rng):
    n  = min(N_SAMPLES_MMD, len(ref_pca), len(cur_pca))
    ri = rng.choice(len(ref_pca), size=n, replace=False)
    ci = rng.choice(len(cur_pca), size=n, replace=False)
    x  = ref_pca[ri]
    y  = cur_pca[ci]

    def rbf(a, b):
        diff = a[:, None, :] - b[None, :, :]
        return np.exp(-np.sum(diff ** 2, axis=-1) / (2 * bandwidth ** 2))

    return float(rbf(x, x).mean() + rbf(y, y).mean() - 2 * rbf(x, y).mean())


def run_mmd_drift_check(embeddings_dir, sim_day, ref_pca, pca_model, null_mean, null_std, rng, injection_days):
    raw = load_embeddings(embeddings_dir, sim_day)
    if raw is None:
        return False, 0.0

    if sim_day in injection_days:
        noise = rng.standard_normal(raw.shape) * 1.5 * raw.std()
        raw   = raw + noise

    cur_pca   = pca_model.transform(raw)
    bandwidth = np.sqrt(np.median(np.sum((ref_pca[:100] - ref_pca[:100].mean(axis=0)) ** 2, axis=1)))
    bandwidth = max(bandwidth, 1e-6)

    mmd_score     = compute_mmd(ref_pca, cur_pca, bandwidth, rng)
    threshold     = null_mean + 2 * null_std
    drift_detected = mmd_score > threshold

    return drift_detected, round(mmd_score, 6)


def run_simulation(window, seed, anchor_date, d_max_hours):
    import mlflow
    import pickle
    import pandas as pd
    from dataclasses import asdict
    from carbon.carbon_scheduler import CarbonScheduler
    from carbon.cooldown_tracker import CooldownTracker
    from carbon.experiment_schema import DriftCheckRecord, RetrainingEventRecord

    os.environ["MLFLOW_TRACKING_USERNAME"] = "pramodkumar26"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

    rng = np.random.default_rng(seed)

    pca_path  = os.path.join(EMBEDDINGS_DIR, "pca_model.pkl")
    null_path = os.path.join(EMBEDDINGS_DIR, "mmd_null_stats.npy")

    with open(pca_path, "rb") as f:
        pca_model = pickle.load(f)

    null_stats = np.load(null_path)
    null_mean  = float(null_stats[0])
    null_std   = float(null_stats[1])

    ref_raw = np.load(os.path.join(EMBEDDINGS_DIR, "ref_embeddings.npy"))
    ref_pca = pca_model.transform(ref_raw)

    caiso_df  = load_caiso(CAISO_CSV)
    scheduler = CarbonScheduler(caiso_csv_path=CAISO_CSV)

    run_id  = f"carbon_aware_dmax{d_max_hours}_window{window}_seed{seed}"
    tracker = CooldownTracker(state_dir=COOLDOWN_DIR, run_id=f"cifar100_pareto_{run_id}")
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
        mlflow.log_param("energy_kwh",  CIFAR100_ENERGY_KWH)
        mlflow.log_param("d_max_hours", d_max_hours)
        mlflow.log_param("dataset",     DATASET)

        for day in range(SIM_DAYS):
            carbon = get_carbon_at_day(day, caiso_df, anchor_date)

            drift_detected, drift_score = run_mmd_drift_check(
                EMBEDDINGS_DIR, day, ref_pca, pca_model,
                null_mean, null_std, rng, CIFAR100_INJECTION_DAYS,
            )

            days_since = tracker.days_since_retrain(DATASET, day)
            eligible   = tracker.is_eligible(DATASET, day)
            triggered  = drift_detected and eligible

            if not triggered:
                continue

            t0     = sim_day_to_datetime(day, caiso_df, anchor_date)
            result = scheduler.schedule(
                t0=t0,
                urgency_class="MEDIUM",
                d_max_hours=d_max_hours,
                dataset_name=DATASET,
            )

            carbon_immediate = result["carbon_intensity_at_t0"]     * CIFAR100_ENERGY_KWH
            carbon_scheduled = result["carbon_intensity_at_t_star"] * CIFAR100_ENERGY_KWH

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
    dag_id="cifar100_pareto",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["greenmlops", "pareto", "cifar100"],
) as dag:
    t1 = PythonOperator(task_id="run_pareto", python_callable=run_pareto)