from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional


@dataclass
class DriftCheckRecord:
    timestamp_day: int
    dataset: str
    drift_score: float
    drift_detected: bool
    days_since_last_retrain: Optional[int]
    retraining_triggered: bool
    accuracy_on_new_distribution: float
    carbon_intensity_at_check: float


@dataclass
class RetrainingEventRecord:
    t0: datetime
    t_star: datetime
    dataset: str
    urgency_class: str
    carbon_intensity_at_t0: float
    carbon_intensity_at_t_star: float
    energy_kwh: float
    carbon_immediate: float
    carbon_scheduled: float
    carbon_saved_pct: float
    wait_hours: float
    accuracy_during_wait: List[float]
    accuracy_post_retrain: float
    policy_applied: str
    delta_max_pct: float
    seed: int
    approach: str

    def to_flat_dict(self) -> dict:
        d = asdict(self)
        d["t0"] = self.t0.isoformat()
        d["t_star"] = self.t_star.isoformat()
        d["accuracy_during_wait"] = str(d["accuracy_during_wait"])
        return d