import time
import pandas as pd
import mlflow
import dagshub
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report
)
from xgboost import XGBClassifier
from codecarbon import EmissionsTracker

DATA_PATH     = r"C:\IP\greenmlops\data\processed\fraud\creditcard_clean.csv"
DAGSHUB_USER  = "pramodkumar26"
DAGSHUB_REPO  = "greenmlops"
RANDOM_STATE  = 42
EMISSIONS_DIR = r"C:\IP\greenmlops\emissions"

dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
mlflow.set_experiment("baseline_training")

df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["Class"])
y  = df["Class"]

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Train fraud rate: {y_train.mean():.4f} | Test fraud rate: {y_test.mean():.4f}")

with mlflow.start_run(run_name="xgboost_fraud_baseline"):

    mlflow.log_params({
        "model":         "XGBoost",
        "dataset":       "credit_card_fraud",
        "urgency_class": "CRITICAL",
        "compute_type":  "CPU",
        "train_size":    len(X_train),
        "test_size":     len(X_test),
        "random_state":  RANDOM_STATE,
        "n_estimators":  1000,
        "max_depth":     8,
        "learning_rate": 0.05,
    })

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=y_train.value_counts()[0] / y_train.value_counts()[1],
        random_state=RANDOM_STATE,
        eval_metric="aucpr",
        verbosity=0,
        device="cpu"
    )

    tracker = EmissionsTracker(
        project_name="greenmlops_fraud_baseline",
        output_dir=EMISSIONS_DIR,
        output_file="emissions_fraud.csv",
        log_level="error"
    )

    tracker.start()
    t_start = time.time()
    model.fit(X_train, y_train)
    t_end = time.time()
    emissions_kg = tracker.stop()

    training_time = t_end - t_start
    energy_kwh    = tracker._total_energy.kWh if emissions_kg else None

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # primary metrics
    primary = {
        "energy_kWh":            energy_kwh,
        "training_time_seconds": training_time,
        "f1_score":              f1_score(y_test, y_pred),
        "precision":             precision_score(y_test, y_pred),
        "recall":                recall_score(y_test, y_pred),
        "roc_auc":               roc_auc_score(y_test, y_proba),
        "avg_precision_score":   average_precision_score(y_test, y_proba),
    }

    # diagnostic only - not used in paper tables
    diagnostic = {
        "diag_codecarbon_gCO2": emissions_kg * 1000 if emissions_kg else 0,
    }

    mlflow.log_metrics({**primary, **diagnostic})
    mlflow.xgboost.log_model(model, name="model")

    print("\n--- Fraud XGBoost Baseline ---")
    print("primary:")
    for k, v in primary.items():
        print(f"  {k:<30} {v:.6f}")
    print("diagnostic:")
    for k, v in diagnostic.items():
        print(f"  {k:<30} {v:.6f}")
    print("\n", classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"\nhttps://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")