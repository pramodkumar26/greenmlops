"""
ett_simulation_core.py

Shared logic for the ETT three-approach comparison simulation.
Imported by ett_periodic.py, ett_drift_immediate.py, ett_carbon_aware.py.

Each DAG owns only its scheduling decision. Everything else lives here.
"""

import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd

from carbon.experiment_schema import DriftCheckRecord, RetrainingEventRecord

logger = logging.getLogger(__name__)

# Empirical measurement from Week 3 baseline training -- MLflow run: baseline_training
ETT_ENERGY_KWH = 0.022

SAMPLES_PER_DAY = 85
REFERENCE_DAYS  = 7
ROLLING_DAYS    = 3

ETT_INJECTION_DAYS = {
    4, 7, 10, 14, 17, 20, 23, 26, 29, 32,
    35, 38, 41, 44, 47, 50, 52, 55, 57, 59,
}

FEATURE_COLS_EXCLUDE = {"OT", "date"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ett_train(processed_dir: str) -> pd.DataFrame:
    path = Path(processed_dir) / "train.csv"
    df = pd.read_csv(path)
    logger.info("Loaded ETT train: %d rows from %s", len(df), path)
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]


def get_reference_window(train_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Fixed reference window: days 0-6 of training data.
    Computed once before simulation begins, never updated.
    """
    return train_df.iloc[: REFERENCE_DAYS * SAMPLES_PER_DAY][feature_cols].copy()


def get_rolling_window(
    train_df: pd.DataFrame,
    feature_cols: list,
    sim_day: int,
) -> pd.DataFrame:
    """
    Rolling current window: most recent 3 days ending at sim_day.
    Drift injection is applied here if sim_day is in the injection schedule.
    """
    window_start = sim_day * SAMPLES_PER_DAY
    current = train_df.iloc[
        window_start: window_start + ROLLING_DAYS * SAMPLES_PER_DAY
    ][feature_cols].copy()

    if sim_day in ETT_INJECTION_DAYS:
        reference = get_reference_window(train_df, feature_cols)
        ref_std = reference.std()
        current = current + 1.5 * ref_std
        logger.debug("Drift injection applied on day %d (1.5 * ref_std)", sim_day)

    return current


# ---------------------------------------------------------------------------
# CAISO carbon data
# ---------------------------------------------------------------------------

_TS_VARIANTS = [
    "Datetime (UTC)",
    "datetime (utc)",
    "datetime_utc",
    "Date",
    "date",
    "timestamp",
]

_CI_VARIANTS = [
    "Carbon intensity gCO\u2082eq/kWh (direct)",
    "Carbon intensity gCO2eq/kWh (direct)",
    "carbon intensity gco2eq/kwh (direct)",
    "carbon_intensity_direct",
    "Carbon Intensity gCO2eq/kWh (direct)",
]


def _match_column(columns: list, variants: list, label: str) -> str:
    normalized = {c.strip().lower(): c for c in columns}
    for v in variants:
        if v in columns:
            return v
        if v.strip().lower() in normalized:
            return normalized[v.strip().lower()]
    raise ValueError(
        f"Could not find {label} column. Expected one of: {variants}. Found: {columns}"
    )


def load_caiso(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    ts_col = _match_column(cols, _TS_VARIANTS, "timestamp")
    ci_col = _match_column(cols, _CI_VARIANTS, "carbon intensity")

    df = df[[ts_col, ci_col]].copy()
    df.columns = ["timestamp", "carbon_intensity"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["carbon_intensity"] = pd.to_numeric(df["carbon_intensity"], errors="coerce")
    df = df.dropna(subset=["carbon_intensity"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Loaded CAISO: %d rows | CI range %.1f-%.1f | first date %s",
        len(df),
        df["carbon_intensity"].min(),
        df["carbon_intensity"].max(),
        df["timestamp"].iloc[0].date(),
    )
    return df


def sim_day_to_datetime(sim_day: int, caiso_df: pd.DataFrame) -> datetime:
    """
    Maps a simulation day index to noon UTC on the corresponding real date.
    sim_day 0 = first date in the CSV at 12:00 UTC, sim_day 1 = next date, etc.

    Noon UTC is used as the hourly anchor so CarbonScheduler can scan forward
    within the D_max window using actual CAISO hourly rows. Using midnight risks
    falling before the first row of a given date in the CSV.
    """
    first_date = caiso_df["timestamp"].iloc[0].normalize()
    target = first_date + timedelta(days=sim_day, hours=12)
    return target.to_pydatetime()


def get_carbon_at_day(sim_day: int, caiso_df: pd.DataFrame) -> float:
    """
    Returns the mean carbon intensity (gCO2/kWh) for the date corresponding
    to sim_day. Averages all hourly rows for that calendar date.

    Averaging across the full day is appropriate here because the simulation
    runs at daily granularity -- we are not picking a specific hour within the
    day for the drift check itself.
    """
    target_dt = sim_day_to_datetime(sim_day, caiso_df)
    target_date = target_dt.date()

    mask = caiso_df["timestamp"].dt.date == target_date
    rows = caiso_df[mask]

    if rows.empty:
        logger.warning(
            "No CAISO rows found for date %s (sim_day=%d). "
            "Check that the CAISO CSV covers the full simulation date range.",
            target_date, sim_day,
        )
        return float(caiso_df["carbon_intensity"].mean())

    return float(rows["carbon_intensity"].mean())


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------

def run_psi_drift_check(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[bool, float]:
    """
    Runs Evidently PSI drift check.
    Returns (drift_detected, drift_score).
    drift_score = share of drifted columns (0.0 to 1.0).
    """
    from evidently.report import Report
    from evidently.metrics import DatasetDriftMetric

    report = Report(metrics=[DatasetDriftMetric()])
    report.run(reference_data=reference, current_data=current)
    result = report.as_dict()["metrics"][0]["result"]

    drift_detected = bool(result["dataset_drift"])
    drift_score    = float(result["share_of_drifted_columns"])
    return drift_detected, round(drift_score, 4)


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def setup_mlflow(mlflow_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)


def log_drift_check(
    record: DriftCheckRecord,
    run_name: str,
):
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params({k: str(v) for k, v in asdict(record).items()})


def log_retraining_event(
    event: RetrainingEventRecord,
    run_name: str,
):
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params({k: str(v) for k, v in event.to_flat_dict().items()})
        mlflow.log_metric("carbon_saved_pct", event.carbon_saved_pct)
        mlflow.log_metric("energy_kwh", event.energy_kwh)
        mlflow.log_metric("wait_hours", event.wait_hours)


# ---------------------------------------------------------------------------
# Retraining event construction
# ---------------------------------------------------------------------------

def build_retraining_event(
    sim_day: int,
    t0: datetime,
    t_star: datetime,
    carbon_at_t0: float,
    carbon_at_t_star: float,
    wait_hours: float,
    carbon_saved_pct: float,
    policy_applied: str,
    delta_max_pct: float,
    seed: int,
    approach: str,
) -> RetrainingEventRecord:
    carbon_immediate = carbon_at_t0   * ETT_ENERGY_KWH
    carbon_scheduled = carbon_at_t_star * ETT_ENERGY_KWH

    return RetrainingEventRecord(
        t0=t0,
        t_star=t_star,
        dataset="ett",
        urgency_class="LOW",
        carbon_intensity_at_t0=round(carbon_at_t0, 2),
        carbon_intensity_at_t_star=round(carbon_at_t_star, 2),
        energy_kwh=ETT_ENERGY_KWH,
        carbon_immediate=round(carbon_immediate, 6),
        carbon_scheduled=round(carbon_scheduled, 6),
        carbon_saved_pct=round(carbon_saved_pct, 2),
        wait_hours=round(wait_hours, 2),
        accuracy_during_wait=[],
        accuracy_post_retrain=-1.0,
        policy_applied=policy_applied,
        delta_max_pct=delta_max_pct,
        seed=seed,
        approach=approach,
    )