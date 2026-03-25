
"""
urgency_classifier.py

Single source of truth for dataset urgency classification in GreenMLOps.
Both CarbonScheduler and Airflow DAGs import from here.
CarbonScheduler no longer owns URGENCY_CONFIG - it imports this module instead.

Urgency classes per DRIFT_PROTOCOL.md:
    CRITICAL  - Fraud: D_max=0, always retrain immediately, no cooldown
    MEDIUM    - CIFAR-100, AG News: D_max=12h, max accuracy drop 2%
    LOW       - ETT: D_max=24h, max accuracy drop 3%
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class UrgencyConfig:
    urgency_class: str
    d_max_hours: int
    delta_max_pct: float
    clean_threshold_gco2: Optional[float]
    cooldown_days: int


_URGENCY_TABLE = {
    "CRITICAL": UrgencyConfig(
        urgency_class="CRITICAL",
        d_max_hours=0,
        delta_max_pct=0.0,
        clean_threshold_gco2=None,
        cooldown_days=0,
    ),
    "MEDIUM": UrgencyConfig(
        urgency_class="MEDIUM",
        d_max_hours=12,
        delta_max_pct=2.0,
        clean_threshold_gco2=180.0,
        cooldown_days=3,
    ),
    "LOW": UrgencyConfig(
        urgency_class="LOW",
        d_max_hours=24,
        delta_max_pct=3.0,
        clean_threshold_gco2=180.0,
        cooldown_days=3,
    ),
}

_DATASET_TO_URGENCY = {
    "fraud":    "CRITICAL",
    "cifar100": "MEDIUM",
    "ag_news":  "MEDIUM",
    "ett":      "LOW",
}


class UrgencyClassifier:

    def get_config(self, urgency_class: str) -> UrgencyConfig:
        key = urgency_class.upper()
        if key not in _URGENCY_TABLE:
            raise ValueError(
                f"Unknown urgency class '{urgency_class}'. "
                f"Valid classes: {list(_URGENCY_TABLE.keys())}"
            )
        return _URGENCY_TABLE[key]

    def classify(self, dataset_name: str) -> UrgencyConfig:
        key = dataset_name.lower()
        if key not in _DATASET_TO_URGENCY:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Known datasets: {list(_DATASET_TO_URGENCY.keys())}"
            )
        urgency_class = _DATASET_TO_URGENCY[key]
        return _URGENCY_TABLE[urgency_class]

    def urgency_class_for(self, dataset_name: str) -> str:
        return self.classify(dataset_name).urgency_class

    def all_datasets(self) -> dict:
        return {ds: _DATASET_TO_URGENCY[ds] for ds in _DATASET_TO_URGENCY}