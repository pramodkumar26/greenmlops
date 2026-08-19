import os
import mlflow
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

TRACKING_URI = "https://dagshub.com/pramodkumar26/greenmlops.mlflow"
mlflow.set_tracking_uri(TRACKING_URI)

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
MAIN_EXPERIMENTS = [
    "ett_simulation",
    "cifar100_simulation",
    "ag_news_simulation",
    "fraud_simulation",
]

PARETO_EXPERIMENTS = [
    "ett_pareto",
    "cifar100_pareto",
    "ag_news_pareto",
]
OUTPUT_DIR  = Path(__file__).parent.parent / "experiments" / "results"
OUTPUT_FILE = OUTPUT_DIR / "simulation_results.csv"

METRICS = [
    "aggregate_carbon_saved_pct",
    "retrain_count",
    "total_carbon_immediate_gco2",
    "total_carbon_scheduled_gco2",
    "energy_kwh",
    "wait_hours",
]
PARAMS = [
    "approach",
    "dataset",
    "seed",
    "window",
    "anchor_date",
    "sim_days",
    "d_max_hours",
]


def collect_experiment(client, experiment_name):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"experiment not found: {experiment_name}")
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        max_results=5000,
        order_by=["attributes.start_time ASC"],
    )

    records = []
    for run in runs:
        parent_id = run.data.tags.get("mlflow.parentRunId", "")
        if parent_id != "":
            continue

        record = {"run_id": run.info.run_id, "run_name": run.info.run_name}

        for param in PARAMS:
            record[param] = run.data.params.get(param, None)

        for metric in METRICS:
            record[metric] = run.data.metrics.get(metric, None)
        if record.get("window") is None:
            continue
        records.append(record)

    return records


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    all_records = []
    for experiment_name in MAIN_EXPERIMENTS + PARETO_EXPERIMENTS:
        print(f"collecting {experiment_name}...")
        records = collect_experiment(client, experiment_name)
        for r in records:
            r["experiment"] = experiment_name
            r["is_pareto"]  = experiment_name in PARETO_EXPERIMENTS
        all_records.extend(records)
        print(f"  {len(records)} parent runs found")

    if not all_records:
        print("no runs collected, check experiment names and credentials")
        return

    df = pd.DataFrame(all_records)

    col_order = [
        "experiment", "is_pareto", "run_name", "run_id",
        "approach", "dataset", "window", "seed", "anchor_date", "sim_days",
        "d_max_hours",
        "aggregate_carbon_saved_pct", "retrain_count",
        "total_carbon_immediate_gco2", "total_carbon_scheduled_gco2",
        "energy_kwh", "wait_hours",
    ]
    existing_cols = [c for c in col_order if c in df.columns]
    df = df[existing_cols]

    df["window"]                      = pd.to_numeric(df["window"],                      errors="coerce")
    df["seed"]                        = pd.to_numeric(df["seed"],                        errors="coerce")
    df["d_max_hours"]                 = pd.to_numeric(df["d_max_hours"],                 errors="coerce")
    df["aggregate_carbon_saved_pct"]  = pd.to_numeric(df["aggregate_carbon_saved_pct"],  errors="coerce")
    df["retrain_count"]               = pd.to_numeric(df["retrain_count"],               errors="coerce")
    df["total_carbon_immediate_gco2"] = pd.to_numeric(df["total_carbon_immediate_gco2"], errors="coerce")
    df["total_carbon_scheduled_gco2"] = pd.to_numeric(df["total_carbon_scheduled_gco2"], errors="coerce")

    df = df.sort_values(["experiment", "approach", "window", "seed"]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nsaved {len(df)} rows to {OUTPUT_FILE}")

    main_df   = df[~df["is_pareto"] & (df["approach"] == "carbon_aware")]
    pareto_df = df[df["is_pareto"]]

    print("\nmain simulation summary (carbon_aware):")
    summary = (
        main_df.groupby("experiment")["aggregate_carbon_saved_pct"]
        .agg(["mean", "min", "max"])
        .round(2)
    )
    print(summary)

    if not pareto_df.empty:
        print("\npareto summary (mean savings by dataset and d_max_hours):")
        pareto_summary = (
            pareto_df.groupby(["experiment", "d_max_hours"])["aggregate_carbon_saved_pct"]
            .mean()
            .round(2)
            .reset_index()
        )
        print(pareto_summary.to_string(index=False))


if __name__ == "__main__":
    main()
