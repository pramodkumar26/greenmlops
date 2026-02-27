import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

URGENCY_CONFIG = {
    "CRITICAL": {
        "datasets": ["fraud"],
        "d_max_hours": 0,
        "delta_max_pct": 0.0,
        "clean_threshold": None,
    },
    "MEDIUM": {
        "datasets": ["cifar100", "ag_news"],
        "d_max_hours": 12,
        "delta_max_pct": 2.0,
        "clean_threshold": 150.0,   
    },
    "LOW": {
        "datasets": ["ett"],
        "d_max_hours": 24,
        "delta_max_pct": 3.0,
        "clean_threshold": 150.0,   
    },
}

DATASET_TO_URGENCY = {
    dataset: urgency
    for urgency, cfg in URGENCY_CONFIG.items()
    for dataset in cfg["datasets"]
}

POLICIES = {
    "immediate_critical":          "CRITICAL urgency - always train at t0, no carbon check",
    "immediate_accuracy_exceeded": "Accuracy drop >= delta_max_pct - forced immediate retrain",
    "immediate_already_clean":     "Carbon at t0 already below clean_threshold",
    "immediate_optimal":           "t0 is the lowest-carbon point within the D_max window",
    "scheduled_clean_window":      "Delayed to a lower-carbon window within D_max",
    "maxdelay_fallback":           "No clean window found within D_max - training at t0 + D_max",
    "fallback_no_data":            "No CAISO data found in window - defaulted to t0",
}

# Known column name variants for the timestamp and carbon intensity columns.
# Electricity Maps exports can differ by locale, encoding, or export version.
_TS_VARIANTS = [
    "Datetime (UTC)",
    "datetime (utc)",
    "datetime_utc",
    "Date",
    "date",
    "timestamp",
]

_CI_VARIANTS = [
    "Carbon intensity gCO\u2082eq/kWh (direct)",   # unicode subscript
    "Carbon intensity gCO2eq/kWh (direct)",        # ascii fallback
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
        f"Could not find {label} column in CSV. "
        f"Expected one of: {variants}. Found: {columns}"
    )


def load_caiso_data(csv_path: str) -> pd.DataFrame:
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

    return df


def _floor_to_hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


class CarbonScheduler:


    def __init__(self, caiso_csv_path: str):
        self.caiso = load_caiso_data(caiso_csv_path)
        self._validate_data()

    def _validate_data(self):
        if self.caiso.empty:
            raise ValueError("CAISO data is empty after loading")
        if len(self.caiso) < 8784:
            logger.warning(
                "CAISO data has %d rows, expected ~8784 for full year 2024",
                len(self.caiso),
            )
        ci = self.caiso["carbon_intensity"]
        logger.info(
            "CAISO loaded: %d rows | CI range %.1f-%.1f gCO2/kWh | mean %.1f",
            len(self.caiso), ci.min(), ci.max(), ci.mean(),
        )

    def _get_carbon_at(self, ts: datetime) -> float:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        idx = (self.caiso["timestamp"] - ts).abs().idxmin()
        nearest = self.caiso.loc[idx, "timestamp"]
        gap_hours = abs((nearest - ts).total_seconds()) / 3600.0
        if gap_hours > 1.0:
            logger.warning(
                "Nearest CAISO row is %.1fh away from requested ts %s - "
                "t0 may be outside the CSV date range",
                gap_hours, ts.isoformat(),
            )
        return float(self.caiso.loc[idx, "carbon_intensity"])

    def _get_window(self, t0: datetime, d_max_hours: int) -> pd.DataFrame:
        # Floor t0 to the hour so we don't accidentally skip the first CAISO row
        # when t0 has minutes/seconds (e.g. 20:03 would miss the 20:00 row otherwise).
        t0_floored = _floor_to_hour(t0)
        t_end = t0_floored + timedelta(hours=d_max_hours)
        mask = (
            (self.caiso["timestamp"] >= t0_floored)
            & (self.caiso["timestamp"] <= t_end)
        )
        return self.caiso[mask].copy()

    def schedule(
        self,
        t0: datetime,
        urgency_class: str,
        d_max_hours: int = None,
        dataset_name: str = None,
        current_accuracy_drop_pct: float = 0.0,
    ) -> dict:
        
        urgency_class = urgency_class.upper()

        if urgency_class not in URGENCY_CONFIG:
            raise ValueError(
                f"Unknown urgency class '{urgency_class}'. "
                f"Must be one of {list(URGENCY_CONFIG.keys())}"
            )

        cfg = URGENCY_CONFIG[urgency_class]

        if d_max_hours is None:
            d_max_hours = cfg["d_max_hours"]

        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)

        t0 = _floor_to_hour(t0)
        carbon_at_t0 = self._get_carbon_at(t0)

        def result(t_star, carbon_t_star, policy):
            return self._build_result(
                t0=t0,
                t_star=t_star,
                carbon_at_t0=carbon_at_t0,
                carbon_at_t_star=carbon_t_star,
                urgency_class=urgency_class,
                d_max_hours=d_max_hours,
                policy=policy,
                dataset_name=dataset_name,
                delta_max_pct=cfg["delta_max_pct"],
            )

        if urgency_class == "CRITICAL" or d_max_hours == 0:
            return result(t0, carbon_at_t0, "immediate_critical")

        if current_accuracy_drop_pct >= cfg["delta_max_pct"]:
            logger.warning(
                "Accuracy drop %.2f%% >= delta_max %.2f%% for %s - forcing immediate retrain",
                current_accuracy_drop_pct, cfg["delta_max_pct"], urgency_class,
            )
            return result(t0, carbon_at_t0, "immediate_accuracy_exceeded")

        window = self._get_window(t0, d_max_hours)

        if window.empty:
            logger.warning(
                "No CAISO data in window [%s, +%dh] - falling back to t0",
                t0.isoformat(), d_max_hours,
            )
            return result(t0, carbon_at_t0, "fallback_no_data")

        if carbon_at_t0 <= cfg["clean_threshold"]:
            return result(t0, carbon_at_t0, "immediate_already_clean")

        min_idx = window["carbon_intensity"].idxmin()
        t_star = window.loc[min_idx, "timestamp"]
        carbon_at_t_star = float(window.loc[min_idx, "carbon_intensity"])

        # Check t_star == t0 before the clean_threshold check.
        # Handles the case where t0 is already the lowest point in a fully dirty window
        
        if t_star == t0:
            return result(t0, carbon_at_t_star, "immediate_optimal")

        # No hour in the window is below clean_threshold - enforce MaxDelay.
        
        if carbon_at_t_star > cfg["clean_threshold"]:
            t_maxdelay = t0 + timedelta(hours=d_max_hours)
            carbon_at_maxdelay = self._get_carbon_at(t_maxdelay)
            return result(t_maxdelay, carbon_at_maxdelay, "maxdelay_fallback")

        return result(t_star, carbon_at_t_star, "scheduled_clean_window")

    def _build_result(
        self, t0, t_star, carbon_at_t0, carbon_at_t_star,
        urgency_class, d_max_hours, policy, dataset_name, delta_max_pct,
    ) -> dict:
        # Normalize to plain python datetimes so MLflow and JSON serialization
        
        if hasattr(t0, "to_pydatetime"):
            t0 = t0.to_pydatetime()
        if hasattr(t_star, "to_pydatetime"):
            t_star = t_star.to_pydatetime()

        wait_hours = (t_star - t0).total_seconds() / 3600.0
        carbon_saved_pct = (
            ((carbon_at_t0 - carbon_at_t_star) / carbon_at_t0) * 100.0
            if carbon_at_t0 > 0 else 0.0
        )
        return {
            "t0": t0,
            "t_star": t_star,
            "urgency_class": urgency_class,
            "d_max_hours": d_max_hours,
            "delta_max_pct": delta_max_pct,
            "dataset_name": dataset_name,
            "carbon_intensity_at_t0": round(carbon_at_t0, 2),
            "carbon_intensity_at_t_star": round(carbon_at_t_star, 2),
            "carbon_saved_pct": round(carbon_saved_pct, 2),
            "wait_hours": round(wait_hours, 2),
            "policy": policy,
        }

    def get_urgency_for_dataset(self, dataset_name: str) -> str:
        name = dataset_name.lower()
        if name not in DATASET_TO_URGENCY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Known datasets: {list(DATASET_TO_URGENCY.keys())}"
            )
        return DATASET_TO_URGENCY[name]

    def schedule_for_dataset(
        self,
        t0: datetime,
        dataset_name: str,
        current_accuracy_drop_pct: float = 0.0,
    ) -> dict:
        urgency_class = self.get_urgency_for_dataset(dataset_name)
        return self.schedule(
            t0=t0,
            urgency_class=urgency_class,
            dataset_name=dataset_name,
            current_accuracy_drop_pct=current_accuracy_drop_pct,
        )

    def carbon_stats(self) -> dict:
        ci = self.caiso["carbon_intensity"]
        p10, p25, p75, p90 = np.percentile(ci, [10, 25, 75, 90])
        return {
            "min": round(float(ci.min()), 2),
            "p10": round(float(p10), 2),
            "p25": round(float(p25), 2),
            "mean": round(float(ci.mean()), 2),
            "median": round(float(ci.median()), 2),
            "p75": round(float(p75), 2),
            "p90": round(float(p90), 2),
            "max": round(float(ci.max()), 2),
        }