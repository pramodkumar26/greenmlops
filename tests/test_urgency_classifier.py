"""
test_urgency_classifier.py
"""

import pytest
from urgency_classifier import UrgencyClassifier, UrgencyConfig


@pytest.fixture
def clf():
    return UrgencyClassifier()


# --- classify() by dataset name ---

def test_fraud_is_critical(clf):
    cfg = clf.classify("fraud")
    assert cfg.urgency_class == "CRITICAL"
    assert cfg.d_max_hours == 0
    assert cfg.delta_max_pct == 0.0
    assert cfg.clean_threshold_gco2 is None
    assert cfg.cooldown_days == 0


def test_cifar100_is_medium(clf):
    cfg = clf.classify("cifar100")
    assert cfg.urgency_class == "MEDIUM"
    assert cfg.d_max_hours == 12
    assert cfg.delta_max_pct == 2.0
    assert cfg.clean_threshold_gco2 == 180.0
    assert cfg.cooldown_days == 3


def test_ag_news_is_medium(clf):
    cfg = clf.classify("ag_news")
    assert cfg.urgency_class == "MEDIUM"
    assert cfg.d_max_hours == 12


def test_ett_is_low(clf):
    cfg = clf.classify("ett")
    assert cfg.urgency_class == "LOW"
    assert cfg.d_max_hours == 24
    assert cfg.delta_max_pct == 3.0
    assert cfg.cooldown_days == 3


def test_classify_case_insensitive(clf):
    assert clf.classify("FRAUD").urgency_class == "CRITICAL"
    assert clf.classify("CIFAR100").urgency_class == "MEDIUM"
    assert clf.classify("AG_NEWS").urgency_class == "MEDIUM"
    assert clf.classify("ETT").urgency_class == "LOW"


def test_classify_unknown_dataset_raises(clf):
    with pytest.raises(ValueError, match="Unknown dataset"):
        clf.classify("telco")


# --- get_config() by urgency class ---

def test_get_config_critical(clf):
    cfg = clf.get_config("CRITICAL")
    assert cfg.d_max_hours == 0
    assert cfg.cooldown_days == 0


def test_get_config_medium(clf):
    cfg = clf.get_config("MEDIUM")
    assert cfg.d_max_hours == 12
    assert cfg.cooldown_days == 3


def test_get_config_low(clf):
    cfg = clf.get_config("LOW")
    assert cfg.d_max_hours == 24
    assert cfg.cooldown_days == 3


def test_get_config_case_insensitive(clf):
    assert clf.get_config("critical").urgency_class == "CRITICAL"
    assert clf.get_config("medium").urgency_class == "MEDIUM"
    assert clf.get_config("low").urgency_class == "LOW"


def test_get_config_unknown_raises(clf):
    with pytest.raises(ValueError, match="Unknown urgency class"):
        clf.get_config("HIGH")


# --- urgency_class_for() convenience method ---

def test_urgency_class_for_fraud(clf):
    assert clf.urgency_class_for("fraud") == "CRITICAL"


def test_urgency_class_for_cifar100(clf):
    assert clf.urgency_class_for("cifar100") == "MEDIUM"


def test_urgency_class_for_ett(clf):
    assert clf.urgency_class_for("ett") == "LOW"


# --- UrgencyConfig is frozen (immutable) ---

def test_config_is_frozen(clf):
    cfg = clf.classify("ett")
    with pytest.raises((AttributeError, TypeError)):
        cfg.d_max_hours = 99


# --- all_datasets() ---

def test_all_datasets_contains_all_four(clf):
    datasets = clf.all_datasets()
    assert set(datasets.keys()) == {"fraud", "cifar100", "ag_news", "ett"}
    assert datasets["fraud"] == "CRITICAL"
    assert datasets["cifar100"] == "MEDIUM"
    assert datasets["ag_news"] == "MEDIUM"
    assert datasets["ett"] == "LOW"


# --- MEDIUM datasets share identical config ---

def test_medium_datasets_share_config(clf):
    cifar_cfg = clf.classify("cifar100")
    ag_cfg = clf.classify("ag_news")
    assert cifar_cfg == ag_cfg