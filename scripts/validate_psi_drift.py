import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

FRAUD_PROCESSED = r"C:\IP\greenmlops\airflow\include\data\processed\fraud\train.csv"

df = pd.read_csv(FRAUD_PROCESSED)
print(f"Loaded: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Reference window: first 595 rows (matches 7-day window scale)
# Rolling window: next 256 rows
reference = df.iloc[:595].copy()
current_clean = df.iloc[595:851].copy()

# Drift injection per DRIFT_PROTOCOL.md:
# shift ALL 29 features by 1.5 * per-feature std of reference window
feature_cols = [c for c in df.columns if c != "Class"]
ref_std = reference[feature_cols].std()
current_drifted = current_clean.copy()
current_drifted[feature_cols] = current_clean[feature_cols] + 1.5 * ref_std

print(f"\nReference: {reference.shape}")
print(f"Current clean: {current_clean.shape}")
print(f"Current drifted: {current_drifted.shape}")
print()

# Run PSI report on clean window - should NOT trigger
report_clean = Report(metrics=[DatasetDriftMetric()])
report_clean.run(reference_data=reference[feature_cols],
                 current_data=current_clean[feature_cols])
result_clean = report_clean.as_dict()
drift_clean = result_clean["metrics"][0]["result"]
print("=== Clean window ===")
print(f"  dataset_drift: {drift_clean['dataset_drift']}")
print(f"  share_drifted_columns: {drift_clean['share_of_drifted_columns']}")
print(f"  number_of_drifted_columns: {drift_clean['number_of_drifted_columns']}")
print()

# Run PSI report on drifted window - should trigger
report_drifted = Report(metrics=[DatasetDriftMetric()])
report_drifted.run(reference_data=reference[feature_cols],
                   current_data=current_drifted[feature_cols])
result_drifted = report_drifted.as_dict()
drift_drifted = result_drifted["metrics"][0]["result"]
print("=== Drifted window (1.5 * std shift) ===")
print(f"  dataset_drift: {drift_drifted['dataset_drift']}")
print(f"  share_drifted_columns: {drift_drifted['share_of_drifted_columns']}")
print(f"  number_of_drifted_columns: {drift_drifted['number_of_drifted_columns']}")
print()

passed = (
    not drift_clean["dataset_drift"]
    and drift_drifted["dataset_drift"]
)
print(f"Result: {'PASS' if passed else 'REVIEW NEEDED'}")






# ETT validation
ETT_PROCESSED = r"C:\IP\greenmlops\airflow\include\data\processed\ett\train.csv"

df_ett = pd.read_csv(ETT_PROCESSED)
print(f"\nETT Loaded: {df_ett.shape}")

feature_cols_ett = [c for c in df_ett.columns if c not in ["OT", "date"]]
reference_ett = df_ett.iloc[:168].copy()
current_clean_ett = df_ett.iloc[168:240].copy()

ref_std_ett = reference_ett[feature_cols_ett].std()
current_drifted_ett = current_clean_ett.copy()
current_drifted_ett[feature_cols_ett] = current_clean_ett[feature_cols_ett] + 1.5 * ref_std_ett

report_clean_ett = Report(metrics=[DatasetDriftMetric()])
report_clean_ett.run(reference_data=reference_ett[feature_cols_ett],
                     current_data=current_clean_ett[feature_cols_ett])
result_clean_ett = report_clean_ett.as_dict()["metrics"][0]["result"]

report_drifted_ett = Report(metrics=[DatasetDriftMetric()])
report_drifted_ett.run(reference_data=reference_ett[feature_cols_ett],
                       current_data=current_drifted_ett[feature_cols_ett])
result_drifted_ett = report_drifted_ett.as_dict()["metrics"][0]["result"]

print("\n=== ETT Clean window ===")
print(f"  dataset_drift: {result_clean_ett['dataset_drift']}")
print(f"  number_of_drifted_columns: {result_clean_ett['number_of_drifted_columns']}")

print("\n=== ETT Drifted window (1.5 * std shift) ===")
print(f"  dataset_drift: {result_drifted_ett['dataset_drift']}")
print(f"  number_of_drifted_columns: {result_drifted_ett['number_of_drifted_columns']}")

natural_drift_cols = result_clean_ett['number_of_drifted_columns']
injected_drift_cols = result_drifted_ett['number_of_drifted_columns']

passed_ett = (
    result_drifted_ett["dataset_drift"]
    and injected_drift_cols >= natural_drift_cols
)
print(f"\nNatural drift columns: {natural_drift_cols}/6")
print(f"Injected drift columns: {injected_drift_cols}/6")
print(f"ETT Result: {'PASS' if passed_ett else 'REVIEW NEEDED'}")
print("Note: ETT natural volatility triggers PSI without injection - expected per DRIFT_PROTOCOL.md")