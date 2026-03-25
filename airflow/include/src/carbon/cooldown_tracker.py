"""
cooldown_tracker.py

Tracks cooldown state per dataset across a simulation run.
State is persisted to a JSON file so it survives across Airflow task
boundaries and simulation loop iterations.

Rules per DRIFT_PROTOCOL.md:
    - MEDIUM and LOW urgency: 3-day minimum between retraining events
    - CRITICAL (Fraud): cooldown does not apply, always eligible
    - Cooldown is per dataset, not per urgency class

One JSON file per simulation run (keyed by run_id). Each seed gets its
own file so runs never share state.

State file schema:
    {
        "run_id": "seed_0",
        "last_retrain_day": {
            "fraud":    -999,
            "cifar100": -999,
            "ag_news":  -999,
            "ett":      -999
        }
    }

-999 means the dataset has never been retrained in this run.
"""

import json
import os
import logging
from typing import Optional

from urgency_classifier import UrgencyClassifier

logger = logging.getLogger(__name__)

_KNOWN_DATASETS = ["fraud", "cifar100", "ag_news", "ett"]
_NEVER_RETRAINED = -999


class CooldownTracker:

    def __init__(self, state_dir: str, run_id: str):
        """
        state_dir : directory where JSON state files are written
        run_id    : unique identifier for this simulation run (e.g. "seed_0")
                    each seed should use a distinct run_id
        """
        self.state_path = os.path.join(state_dir, f"cooldown_{run_id}.json")
        self.run_id = run_id
        self._clf = UrgencyClassifier()
        self._state = self._load_or_init()

    def _load_or_init(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            logger.info("Loaded cooldown state from %s", self.state_path)
            return state

        state = {
            "run_id": self.run_id,
            "last_retrain_day": {ds: _NEVER_RETRAINED for ds in _KNOWN_DATASETS},
        }
        self._write(state)
        logger.info("Initialized new cooldown state at %s", self.state_path)
        return state

    def _write(self, state: dict):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def is_eligible(self, dataset_name: str, current_day: int) -> bool:
        """
        Returns True if the dataset is eligible for retraining on current_day.

        CRITICAL datasets are always eligible regardless of current_day.
        MEDIUM/LOW datasets must have current_day - last_retrain_day >= cooldown_days.
        A dataset that has never been retrained (_NEVER_RETRAINED) is always eligible.
        """
        key = dataset_name.lower()
        cfg = self._clf.classify(key)

        if cfg.cooldown_days == 0:
            return True

        last_day = self._state["last_retrain_day"].get(key, _NEVER_RETRAINED)

        if last_day == _NEVER_RETRAINED:
            return True

        days_since = current_day - last_day
        eligible = days_since >= cfg.cooldown_days

        if not eligible:
            logger.debug(
                "%s cooldown active: last retrain day %d, current day %d, "
                "need %d days gap, have %d",
                key, last_day, current_day, cfg.cooldown_days, days_since,
            )

        return eligible

    def record_retrain(self, dataset_name: str, current_day: int):
        """
        Call this after a retraining event executes (at t_star, not t0).
        Updates last_retrain_day and persists state.
        """
        key = dataset_name.lower()
        if key not in _KNOWN_DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Known: {_KNOWN_DATASETS}"
            )
        self._state["last_retrain_day"][key] = current_day
        self._write(self._state)
        logger.info("Recorded retrain for %s on day %d", key, current_day)

    def days_since_retrain(self, dataset_name: str, current_day: int) -> Optional[int]:
        """
        Returns days since last retrain, or None if never retrained in this run.
        Used for MLflow logging.
        """
        key = dataset_name.lower()
        last_day = self._state["last_retrain_day"].get(key, _NEVER_RETRAINED)
        if last_day == _NEVER_RETRAINED:
            return None
        return current_day - last_day

    def reset(self):
        """
        Resets all cooldown state for this run. Overwrites the JSON file.
        Use at the start of a new simulation run if reusing the same run_id.
        """
        self._state = {
            "run_id": self.run_id,
            "last_retrain_day": {ds: _NEVER_RETRAINED for ds in _KNOWN_DATASETS},
        }
        self._write(self._state)
        logger.info("Reset cooldown state for run_id=%s", self.run_id)

    def snapshot(self) -> dict:
        """Returns a copy of current state. Used for MLflow logging."""
        return {
            "run_id": self._state["run_id"],
            "last_retrain_day": dict(self._state["last_retrain_day"]),
        }