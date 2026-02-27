import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
import sys
import os

from carbon.carbon_scheduler import (
    CarbonScheduler, load_caiso_data, URGENCY_CONFIG, POLICIES, _floor_to_hour
)

def make_scheduler_with_mock(rows: list) -> CarbonScheduler:
    scheduler = CarbonScheduler.__new__(CarbonScheduler)
    scheduler.caiso = pd.DataFrame(rows, columns=["timestamp", "carbon_intensity"])
    return scheduler


def ts(hour: int, day: int = 15, minute: int = 0) -> datetime:
    return datetime(2024, 6, day, hour, minute, 0, tzinfo=timezone.utc)


def build_two_day_caiso() -> list:
    rows = []
    for day in [15, 16]:
        for h in range(24):
            ci = 120.0 if 17 <= h <= 21 else 220.0
            rows.append({"timestamp": ts(h, day), "carbon_intensity": ci})
    return rows


# ---------------------------------------------------------------------------
# Fix 1: MaxDelay fallback
# ---------------------------------------------------------------------------

class TestMaxDelayFallback:
    def _dirty_dip_rows(self):
        # t0=20:00 CI=410, dips to 300 at 22:00 (above 250 threshold), rises again.
        # t0 is not the argmin, so immediate_optimal does not fire.
        # Nothing clears clean_threshold, so maxdelay_fallback should fire.
        return [
            {"timestamp": ts(20), "carbon_intensity": 410.0},
            {"timestamp": ts(21), "carbon_intensity": 350.0},
            {"timestamp": ts(22), "carbon_intensity": 300.0},
            {"timestamp": ts(23), "carbon_intensity": 320.0},
            {"timestamp": ts(0, day=16), "carbon_intensity": 340.0},
            {"timestamp": ts(1, day=16), "carbon_intensity": 360.0},
            {"timestamp": ts(2, day=16), "carbon_intensity": 370.0},
            {"timestamp": ts(3, day=16), "carbon_intensity": 380.0},
            {"timestamp": ts(4, day=16), "carbon_intensity": 390.0},
            {"timestamp": ts(5, day=16), "carbon_intensity": 395.0},
            {"timestamp": ts(6, day=16), "carbon_intensity": 400.0},
            {"timestamp": ts(7, day=16), "carbon_intensity": 405.0},
            {"timestamp": ts(8, day=16), "carbon_intensity": 410.0},
        ]

    def test_emits_maxdelay_when_no_clean_window_in_budget(self):
        result = make_scheduler_with_mock(self._dirty_dip_rows()).schedule(
            t0=ts(20), urgency_class="MEDIUM", d_max_hours=12
        )
        assert result["policy"] == "maxdelay_fallback"

    def test_maxdelay_t_star_is_t0_plus_d_max(self):
        t_trigger = ts(20)
        result = make_scheduler_with_mock(self._dirty_dip_rows()).schedule(
            t0=t_trigger, urgency_class="MEDIUM", d_max_hours=12
        )
        assert result["t_star"] == t_trigger + timedelta(hours=12)

    def test_maxdelay_wait_hours_equals_d_max(self):
        result = make_scheduler_with_mock(self._dirty_dip_rows()).schedule(
            t0=ts(20), urgency_class="MEDIUM", d_max_hours=12
        )
        assert result["wait_hours"] == 12.0

    def test_clean_window_found_does_not_use_maxdelay(self):
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(5), urgency_class="LOW", d_max_hours=24
        )
        assert result["policy"] == "scheduled_clean_window"

    def test_maxdelay_policy_is_documented(self):
        assert "maxdelay_fallback" in POLICIES


class TestImmediateOptimal:
    def test_reachable_when_t0_is_lowest_in_dirty_window(self):
        # Carbon rises through the evening - t0=20:00 is the minimum but never clean.
        # immediate_optimal should fire, not maxdelay_fallback.
        rows = [
            {"timestamp": ts(20), "carbon_intensity": 380.0},
            {"timestamp": ts(21), "carbon_intensity": 400.0},
            {"timestamp": ts(22), "carbon_intensity": 420.0},
            {"timestamp": ts(23), "carbon_intensity": 440.0},
        ]
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20), urgency_class="MEDIUM", d_max_hours=3
        )
        assert result["policy"] == "immediate_optimal"
        assert result["t_star"] == ts(20)
        assert result["wait_hours"] == 0.0

    def test_immediate_optimal_not_confused_with_immediate_already_clean(self):
        # immediate_already_clean fires when C(t0) <= threshold.
        # immediate_optimal fires when t0 is argmin but still above threshold.
        rows = [
            {"timestamp": ts(20), "carbon_intensity": 380.0},  # above 250 threshold
            {"timestamp": ts(21), "carbon_intensity": 400.0},
        ]
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20), urgency_class="MEDIUM", d_max_hours=1
        )
        assert result["policy"] == "immediate_optimal"
        assert result["policy"] != "immediate_already_clean"


# ---------------------------------------------------------------------------
# Fix 2: Column name tolerance
# ---------------------------------------------------------------------------

class TestColumnNameTolerance:
    def _make_csv(self, tmp_path, ts_col: str, ci_col: str):
        import csv
        path = tmp_path / "caiso.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ts_col, ci_col, "extra_col"])
            writer.writerow(["2024-01-01 00:00:00", "322.46", "x"])
            writer.writerow(["2024-01-01 01:00:00", "310.00", "x"])
        return str(path)

    def test_loads_unicode_subscript_column(self, tmp_path):
        path = self._make_csv(
            tmp_path,
            "Datetime (UTC)",
            "Carbon intensity gCO\u2082eq/kWh (direct)",
        )
        df = load_caiso_data(path)
        assert len(df) == 2
        assert "carbon_intensity" in df.columns

    def test_loads_ascii_fallback_column(self, tmp_path):
        path = self._make_csv(
            tmp_path,
            "Datetime (UTC)",
            "Carbon intensity gCO2eq/kWh (direct)",
        )
        df = load_caiso_data(path)
        assert len(df) == 2

    def test_raises_on_completely_unknown_columns(self, tmp_path):
        import csv
        path = tmp_path / "bad.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["some_date", "some_value"])
            writer.writerow(["2024-01-01", "100"])
        with pytest.raises(ValueError, match="timestamp"):
            load_caiso_data(str(path))


# ---------------------------------------------------------------------------
# Fix 3: Timestamp alignment (floor to hour)
# ---------------------------------------------------------------------------

class TestTimestampAlignment:
    def test_t0_with_minutes_floors_to_hour(self):
        t_with_minutes = ts(20, minute=37)
        floored = _floor_to_hour(t_with_minutes)
        assert floored == ts(20, minute=0)

    def test_window_includes_t0_hour_when_t0_has_minutes(self):
        # t0=20:37 should still include the 20:00 CAISO row in its window
        rows = build_two_day_caiso()
        scheduler = make_scheduler_with_mock(rows)
        window = scheduler._get_window(ts(20, minute=37), d_max_hours=4)
        window_timestamps = list(window["timestamp"])
        assert pd.Timestamp("2024-06-15 20:00:00", tz="UTC") in window_timestamps

    def test_schedule_with_non_zero_minutes_does_not_crash(self):
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20, minute=15), urgency_class="LOW", d_max_hours=24
        )
        assert "t_star" in result

    def test_t0_floored_in_result(self):
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20, minute=45), urgency_class="CRITICAL"
        )
        assert result["t0"].minute == 0


# ---------------------------------------------------------------------------
# Core scheduling logic (regression)
# ---------------------------------------------------------------------------

class TestCriticalUrgency:
    def test_immediate_regardless_of_carbon(self):
        rows = [{"timestamp": ts(20), "carbon_intensity": 480.0}]
        result = make_scheduler_with_mock(rows).schedule(t0=ts(20), urgency_class="CRITICAL")
        assert result["policy"] == "immediate_critical"
        assert result["wait_hours"] == 0.0

    def test_accuracy_drop_ignored(self):
        rows = [{"timestamp": ts(20), "carbon_intensity": 480.0}]
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20), urgency_class="CRITICAL", current_accuracy_drop_pct=5.0
        )
        assert result["policy"] == "immediate_critical"


class TestAccuracyGuard:
    def test_forces_immediate_at_exact_threshold(self):
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20), urgency_class="MEDIUM", current_accuracy_drop_pct=2.0
        )
        assert result["policy"] == "immediate_accuracy_exceeded"

    def test_allows_delay_below_threshold(self):
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(5), urgency_class="LOW", d_max_hours=24, current_accuracy_drop_pct=1.9
        )
        assert result["policy"] == "scheduled_clean_window"


class TestCleanWindowScheduling:
    def test_delays_to_solar_window(self):
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(5), urgency_class="LOW", d_max_hours=24
        )
        assert result["policy"] == "scheduled_clean_window"
        assert result["carbon_intensity_at_t_star"] < result["carbon_intensity_at_t0"]

    def test_immediate_when_already_clean(self):
        rows = [{"timestamp": ts(10), "carbon_intensity": 150.0}]
        result = make_scheduler_with_mock(rows).schedule(t0=ts(10), urgency_class="MEDIUM")
        assert result["policy"] == "immediate_already_clean"

    def test_fallback_no_data(self):
        # fallback_no_data fires only when the CAISO window is completely empty.
        # A single row outside the window simulates missing data for that period.
        rows = [{"timestamp": ts(1), "carbon_intensity": 400.0}]  # far from t0=20
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20), urgency_class="MEDIUM", d_max_hours=12
        )
        assert result["policy"] == "fallback_no_data"


class TestResultSchema:
    def test_all_keys_present(self):
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(t0=ts(20), urgency_class="MEDIUM")
        for key in [
            "t0", "t_star", "urgency_class", "d_max_hours", "delta_max_pct",
            "dataset_name", "carbon_intensity_at_t0", "carbon_intensity_at_t_star",
            "carbon_saved_pct", "wait_hours", "policy",
        ]:
            assert key in result

    def test_all_policies_documented(self):
        rows = build_two_day_caiso()
        scheduler = make_scheduler_with_mock(rows)
        scenarios = [
            dict(t0=ts(20), urgency_class="CRITICAL"),
            dict(t0=ts(20), urgency_class="MEDIUM", current_accuracy_drop_pct=2.5),
            dict(t0=ts(10), urgency_class="MEDIUM"),
            dict(t0=ts(20), urgency_class="LOW", d_max_hours=24),
            dict(t0=ts(20), urgency_class="MEDIUM", d_max_hours=12),
        ]
        for kwargs in scenarios:
            result = scheduler.schedule(**kwargs)
            assert result["policy"] in POLICIES, f"Undocumented policy for {kwargs}"


class TestDatasetRouting:
    def setup_method(self):
        self.scheduler = make_scheduler_with_mock(build_two_day_caiso())

    def test_fraud_critical(self):
        assert self.scheduler.get_urgency_for_dataset("fraud") == "CRITICAL"

    def test_cifar100_medium(self):
        assert self.scheduler.get_urgency_for_dataset("cifar100") == "MEDIUM"

    def test_ag_news_medium(self):
        assert self.scheduler.get_urgency_for_dataset("ag_news") == "MEDIUM"

    def test_ett_low(self):
        assert self.scheduler.get_urgency_for_dataset("ett") == "LOW"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            self.scheduler.get_urgency_for_dataset("nonexistent")


# ---------------------------------------------------------------------------
# Fix: timestamp types, numeric coercion, out-of-range guard
# ---------------------------------------------------------------------------

class TestTimestampTypes:
    def test_t0_is_python_datetime(self):
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20), urgency_class="CRITICAL"
        )
        assert isinstance(result["t0"], datetime)
        assert not type(result["t0"]).__name__ == "Timestamp"

    def test_t_star_is_python_datetime_when_scheduled(self):
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20), urgency_class="LOW", d_max_hours=24
        )
        assert isinstance(result["t_star"], datetime)
        assert not type(result["t_star"]).__name__ == "Timestamp"

    def test_t_star_is_python_datetime_for_immediate(self):
        rows = [{"timestamp": ts(10), "carbon_intensity": 150.0}]
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(10), urgency_class="MEDIUM"
        )
        assert isinstance(result["t_star"], datetime)

    def test_result_is_json_serializable(self):
        import json
        rows = build_two_day_caiso()
        result = make_scheduler_with_mock(rows).schedule(
            t0=ts(20), urgency_class="LOW", d_max_hours=24
        )
        serializable = {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in result.items()
        }
        json.dumps(serializable)  # should not raise


class TestNumericCoercion:
    def test_string_carbon_values_are_coerced(self, tmp_path):
        import csv
        path = tmp_path / "caiso_str.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Datetime (UTC)", "Carbon intensity gCO2eq/kWh (direct)"])
            writer.writerow(["2024-01-01 00:00:00", "322.46"])
            writer.writerow(["2024-01-01 01:00:00", "310.00"])
        df = load_caiso_data(str(path))
        assert df["carbon_intensity"].dtype in [float, "float64"]
        assert len(df) == 2

    def test_non_numeric_rows_are_dropped(self, tmp_path):
        import csv
        path = tmp_path / "caiso_bad.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Datetime (UTC)", "Carbon intensity gCO2eq/kWh (direct)"])
            writer.writerow(["2024-01-01 00:00:00", "322.46"])
            writer.writerow(["2024-01-01 01:00:00", "N/A"])
            writer.writerow(["2024-01-01 02:00:00", "310.00"])
        df = load_caiso_data(str(path))
        assert len(df) == 2


class TestOutOfRangeGuard:
    def test_warns_when_t0_far_from_nearest_row(self, caplog):
        import logging
        rows = [{"timestamp": ts(10), "carbon_intensity": 300.0}]
        scheduler = make_scheduler_with_mock(rows)
        # Request carbon at ts(23) - 13 hours away from the only row at ts(10)
        with caplog.at_level(logging.WARNING, logger="carbon_scheduler"):
            scheduler._get_carbon_at(ts(23))
        assert any("outside the CSV date range" in r.message for r in caplog.records)

    def test_no_warning_when_t0_within_one_hour(self, caplog):
        import logging
        rows = [{"timestamp": ts(10), "carbon_intensity": 300.0}]
        scheduler = make_scheduler_with_mock(rows)
        with caplog.at_level(logging.WARNING, logger="carbon_scheduler"):
            scheduler._get_carbon_at(ts(10))
        assert not any("outside the CSV date range" in r.message for r in caplog.records)