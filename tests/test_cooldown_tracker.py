"""
test_cooldown_tracker.py
"""

import json
import os
import pytest
from cooldown_tracker import CooldownTracker, _NEVER_RETRAINED


@pytest.fixture
def tracker(tmp_path):
    return CooldownTracker(state_dir=str(tmp_path), run_id="test_run")


@pytest.fixture
def state_dir(tmp_path):
    return str(tmp_path)


# --- initialization ---

def test_state_file_created_on_init(tmp_path):
    CooldownTracker(state_dir=str(tmp_path), run_id="seed_0")
    assert os.path.exists(os.path.join(str(tmp_path), "cooldown_seed_0.json"))


def test_initial_state_all_never_retrained(tmp_path):
    tracker = CooldownTracker(state_dir=str(tmp_path), run_id="seed_0")
    snap = tracker.snapshot()
    for ds in ["fraud", "cifar100", "ag_news", "ett"]:
        assert snap["last_retrain_day"][ds] == _NEVER_RETRAINED


def test_run_id_stored_in_state(tmp_path):
    tracker = CooldownTracker(state_dir=str(tmp_path), run_id="seed_2")
    assert tracker.snapshot()["run_id"] == "seed_2"


def test_separate_run_ids_get_separate_files(tmp_path):
    CooldownTracker(state_dir=str(tmp_path), run_id="seed_0")
    CooldownTracker(state_dir=str(tmp_path), run_id="seed_1")
    assert os.path.exists(os.path.join(str(tmp_path), "cooldown_seed_0.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "cooldown_seed_1.json"))


# --- is_eligible() ---

def test_never_retrained_is_always_eligible(tracker):
    assert tracker.is_eligible("cifar100", current_day=0) is True
    assert tracker.is_eligible("ag_news", current_day=0) is True
    assert tracker.is_eligible("ett", current_day=0) is True


def test_fraud_always_eligible_regardless_of_history(tracker):
    tracker.record_retrain("fraud", current_day=5)
    assert tracker.is_eligible("fraud", current_day=6) is True
    assert tracker.is_eligible("fraud", current_day=7) is True


def test_medium_eligible_after_cooldown(tracker):
    tracker.record_retrain("cifar100", current_day=10)
    assert tracker.is_eligible("cifar100", current_day=12) is False
    assert tracker.is_eligible("cifar100", current_day=13) is True


def test_low_eligible_after_cooldown(tracker):
    tracker.record_retrain("ett", current_day=10)
    assert tracker.is_eligible("ett", current_day=12) is False
    assert tracker.is_eligible("ett", current_day=13) is True


def test_medium_not_eligible_within_cooldown(tracker):
    tracker.record_retrain("ag_news", current_day=20)
    assert tracker.is_eligible("ag_news", current_day=21) is False
    assert tracker.is_eligible("ag_news", current_day=22) is False


def test_eligible_exactly_at_cooldown_boundary(tracker):
    tracker.record_retrain("cifar100", current_day=15)
    assert tracker.is_eligible("cifar100", current_day=18) is True


def test_eligible_well_past_cooldown(tracker):
    tracker.record_retrain("ett", current_day=5)
    assert tracker.is_eligible("ett", current_day=30) is True


# --- record_retrain() ---

def test_record_retrain_updates_state(tracker):
    tracker.record_retrain("cifar100", current_day=7)
    assert tracker.snapshot()["last_retrain_day"]["cifar100"] == 7


def test_record_retrain_persists_to_file(tmp_path):
    tracker = CooldownTracker(state_dir=str(tmp_path), run_id="seed_0")
    tracker.record_retrain("ett", current_day=12)

    with open(os.path.join(str(tmp_path), "cooldown_seed_0.json"), "r") as f:
        state = json.load(f)
    assert state["last_retrain_day"]["ett"] == 12


def test_record_retrain_does_not_affect_other_datasets(tracker):
    tracker.record_retrain("cifar100", current_day=10)
    snap = tracker.snapshot()
    assert snap["last_retrain_day"]["ag_news"] == _NEVER_RETRAINED
    assert snap["last_retrain_day"]["fraud"] == _NEVER_RETRAINED
    assert snap["last_retrain_day"]["ett"] == _NEVER_RETRAINED


def test_record_retrain_unknown_dataset_raises(tracker):
    with pytest.raises(ValueError, match="Unknown dataset"):
        tracker.record_retrain("telco", current_day=5)


def test_record_retrain_overwrites_previous(tracker):
    tracker.record_retrain("cifar100", current_day=5)
    tracker.record_retrain("cifar100", current_day=10)
    assert tracker.snapshot()["last_retrain_day"]["cifar100"] == 10


# --- days_since_retrain() ---

def test_days_since_retrain_never_retrained_returns_none(tracker):
    assert tracker.days_since_retrain("cifar100", current_day=10) is None


def test_days_since_retrain_correct_value(tracker):
    tracker.record_retrain("ett", current_day=5)
    assert tracker.days_since_retrain("ett", current_day=10) == 5


def test_days_since_retrain_same_day(tracker):
    tracker.record_retrain("ag_news", current_day=15)
    assert tracker.days_since_retrain("ag_news", current_day=15) == 0


# --- state persistence across instances ---

def test_state_loads_from_existing_file(tmp_path):
    t1 = CooldownTracker(state_dir=str(tmp_path), run_id="seed_0")
    t1.record_retrain("cifar100", current_day=20)

    t2 = CooldownTracker(state_dir=str(tmp_path), run_id="seed_0")
    assert t2.snapshot()["last_retrain_day"]["cifar100"] == 20


def test_different_run_ids_have_independent_state(tmp_path):
    t1 = CooldownTracker(state_dir=str(tmp_path), run_id="seed_0")
    t2 = CooldownTracker(state_dir=str(tmp_path), run_id="seed_1")

    t1.record_retrain("cifar100", current_day=10)
    assert t2.snapshot()["last_retrain_day"]["cifar100"] == _NEVER_RETRAINED


# --- reset() ---

def test_reset_clears_all_state(tracker):
    tracker.record_retrain("cifar100", current_day=10)
    tracker.record_retrain("ett", current_day=15)
    tracker.reset()
    snap = tracker.snapshot()
    for ds in ["fraud", "cifar100", "ag_news", "ett"]:
        assert snap["last_retrain_day"][ds] == _NEVER_RETRAINED


def test_reset_persists_to_file(tmp_path):
    tracker = CooldownTracker(state_dir=str(tmp_path), run_id="seed_0")
    tracker.record_retrain("ett", current_day=5)
    tracker.reset()

    with open(os.path.join(str(tmp_path), "cooldown_seed_0.json"), "r") as f:
        state = json.load(f)
    assert state["last_retrain_day"]["ett"] == _NEVER_RETRAINED


def test_eligible_after_reset(tracker):
    tracker.record_retrain("cifar100", current_day=10)
    assert tracker.is_eligible("cifar100", current_day=11) is False
    tracker.reset()
    assert tracker.is_eligible("cifar100", current_day=11) is True