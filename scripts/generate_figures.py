import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

RESULTS_CSV  = "experiments/results/simulation_results.csv"
FIGURES_DIR  = "experiments/figures"
CAISO_CSV    = "airflow/include/data/raw/carbon/caiso_2024_hourly.csv"

ENERGY_MAP = {
    "ett":      0.022,
    "cifar100": 0.007853,
    "ag_news":  0.187614,
    "fraud":    0.001007,
}

COMPUTE_TYPE = {
    "ett":      "CPU",
    "cifar100": "GPU",
    "ag_news":  "GPU",
    "fraud":    "CPU",
}

URGENCY = {
    "ett":      "LOW",
    "cifar100": "MEDIUM",
    "ag_news":  "MEDIUM",
    "fraud":    "CRITICAL",
}

DMAX_MAIN = {
    "ett":      24.0,
    "cifar100": 12.0,
    "ag_news":  12.0,
    "fraud":     0.0,
}

DATASET_LABELS = {
    "ett":      "ETT/LSTM",
    "cifar100": "CIFAR-100/ResNet-18",
    "ag_news":  "AG News/DistilBERT",
    "fraud":    "Fraud/XGBoost",
}

APPROACH_LABELS = {
    "periodic":        "Periodic",
    "drift_immediate": "Drift-Immediate",
    "carbon_aware":    "Carbon-Aware (Ours)",
}

WINDOW_LABELS = {
    0: "Jan 2024",
    1: "May 2024",
    2: "Oct 2024",
}

COLORS = {
    "periodic":        "#a8c5da",
    "drift_immediate": "#f4a261",
    "carbon_aware":    "#2a9d8f",
    "GPU":             "#e76f51",
    "CPU":             "#457b9d",
}

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_results(csv_path):
    df = pd.read_csv(csv_path)

    def infer_dataset(row):
        if pd.notna(row["dataset"]):
            return row["dataset"]
        for name in ["ag_news", "cifar100", "fraud", "ett"]:
            if name in row["experiment"]:
                return name
        return None

    df["dataset"]      = df.apply(infer_dataset, axis=1)
    df["energy_kwh"]   = df.apply(
        lambda r: ENERGY_MAP[r["dataset"]] if pd.isna(r["energy_kwh"]) else r["energy_kwh"],
        axis=1,
    )
    df["compute_type"] = df["dataset"].map(COMPUTE_TYPE)
    df["urgency"]      = df["dataset"].map(URGENCY)
    return df


def figure1_carbon_by_approach(df):
    """Bar chart: mean total carbon per approach per dataset (main simulation)."""
    main = df[df["is_pareto"] == False].copy()

    datasets  = ["ett", "cifar100", "ag_news", "fraud"]
    approaches = ["periodic", "drift_immediate", "carbon_aware"]

    grouped = (
        main.groupby(["dataset", "approach"])["total_carbon_immediate_gco2"]
        .mean()
        .reset_index()
    )
    # for carbon_aware use scheduled carbon, others use immediate
    ca_rows = main[main["approach"] == "carbon_aware"].groupby("dataset")["total_carbon_scheduled_gco2"].mean()
    for idx, row in grouped.iterrows():
        if row["approach"] == "carbon_aware":
            grouped.at[idx, "total_carbon_immediate_gco2"] = ca_rows[row["dataset"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    n_datasets  = len(datasets)
    n_approaches = len(approaches)
    bar_width   = 0.25
    x           = np.arange(n_datasets)

    for i, approach in enumerate(approaches):
        vals = []
        for ds in datasets:
            subset = grouped[(grouped["dataset"] == ds) & (grouped["approach"] == approach)]
            vals.append(float(subset["total_carbon_immediate_gco2"].values[0]) if len(subset) > 0 else 0.0)
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=APPROACH_LABELS[approach],
                      color=COLORS[approach], edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Dataset / Model", fontsize=12)
    ax.set_ylabel("Mean Total Carbon Emissions (gCO\u2082)", fontsize=12)
    ax.set_title("Carbon Emissions per Approach across Datasets\n(averaged over 3 seasonal windows \u00d7 3 seeds)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in datasets], fontsize=10)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure1_carbon_by_approach.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def figure2_tradeoff_scatter(df):
    """Scatter: carbon reduction vs compute type (GPU vs CPU), main carbon_aware runs."""
    main_ca = df[(df["is_pareto"] == False) & (df["approach"] == "carbon_aware")].copy()

    fig, ax = plt.subplots(figsize=(8, 6))

    for ds in main_ca["dataset"].unique():
        subset = main_ca[main_ca["dataset"] == ds]
        ctype  = COMPUTE_TYPE[ds]
        color  = COLORS[ctype]
        ax.scatter(
            subset["aggregate_carbon_saved_pct"],
            [DATASET_LABELS[ds]] * len(subset),
            color=color,
            s=80,
            alpha=0.85,
            zorder=3,
        )
        mean_val = subset["aggregate_carbon_saved_pct"].mean()
        ax.axvline(mean_val, color=color, linestyle="--", alpha=0.4, linewidth=1)

    gpu_patch = mpatches.Patch(color=COLORS["GPU"], label="GPU workload")
    cpu_patch = mpatches.Patch(color=COLORS["CPU"], label="CPU workload")
    ax.legend(handles=[gpu_patch, cpu_patch], fontsize=10)

    ax.set_xlabel("Carbon Savings (%)", fontsize=12)
    ax.set_title("Carbon Savings Distribution by Dataset and Compute Type\n(carbon_aware approach, all windows and seeds)", fontsize=12)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.axvline(30, color="gray", linestyle=":", linewidth=1, label="30% publishable floor")

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure2_tradeoff_scatter.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def figure3_carbon_intensity_timeseries():
    """Carbon intensity time-series for Jan window with scheduled training markers."""
    if not os.path.exists(CAISO_CSV):
        print("CAISO CSV not found, skipping figure3")
        return

    import pandas as pd
    df = pd.read_csv(CAISO_CSV)

    ts_col = None
    ci_col = None
    for c in df.columns:
        if "datetime" in c.lower() or "date" in c.lower() or "timestamp" in c.lower():
            ts_col = c
        if "carbon" in c.lower() and "direct" in c.lower():
            ci_col = c

    if ts_col is None or ci_col is None:
        print(f"could not find required columns in CAISO CSV, skipping figure3")
        return

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df[ci_col] = pd.to_numeric(df[ci_col], errors="coerce")
    df = df.dropna(subset=[ci_col]).sort_values(ts_col)

    jan = df[(df[ts_col] >= "2024-01-01") & (df[ts_col] < "2024-03-02")]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(jan[ts_col], jan[ci_col], color="#457b9d", linewidth=0.8, alpha=0.9, label="Grid carbon intensity")
    ax.axhline(180, color="#2a9d8f", linestyle="--", linewidth=1.2, label="Clean window threshold (180 gCO\u2082/kWh)")

    ax.fill_between(jan[ts_col], jan[ci_col], 180,
                    where=(jan[ci_col] <= 180),
                    alpha=0.25, color="#2a9d8f", label="Clean energy windows")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Carbon Intensity (gCO\u2082/kWh)", fontsize=12)
    ax.set_title("CAISO Grid Carbon Intensity \u2014 Jan\u2013Feb 2024\nClean energy windows available for scheduled retraining", fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure3_carbon_intensity_timeseries.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def figure4_savings_by_window(df):
    """Carbon savings across 3 seasonal windows, carbon_aware, non-fraud datasets."""
    main_ca = df[
        (df["is_pareto"] == False) &
        (df["approach"] == "carbon_aware") &
        (df["dataset"] != "fraud")
    ].copy()

    datasets = ["ett", "cifar100", "ag_news"]
    windows  = [0, 1, 2]

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width   = 0.25
    x           = np.arange(len(windows))

    for i, ds in enumerate(datasets):
        vals = []
        errs = []
        for w in windows:
            subset = main_ca[(main_ca["dataset"] == ds) & (main_ca["window"] == w)]
            vals.append(subset["aggregate_carbon_saved_pct"].mean())
            errs.append(subset["aggregate_carbon_saved_pct"].std())
        offset = (i - 1) * bar_width
        ax.bar(x + offset, vals, bar_width, yerr=errs, capsize=3,
               label=DATASET_LABELS[ds], edgecolor="white", linewidth=0.5, alpha=0.9)

    ax.set_xlabel("Seasonal Window", fontsize=12)
    ax.set_ylabel("Carbon Savings (%)", fontsize=12)
    ax.set_title("Carbon Savings by Seasonal Window\n(carbon_aware approach, error bars = std across 3 seeds)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([WINDOW_LABELS[w] for w in windows], fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure4_savings_by_window.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def figure5_pareto_frontier(df):
    """Pareto frontier: D_max vs carbon savings per dataset."""
    pareto = df[df["is_pareto"] == True].copy()

    # pull main simulation D_max values to complete the curves
    main_ca = df[
        (df["is_pareto"] == False) &
        (df["approach"] == "carbon_aware") &
        (df["dataset"] != "fraud")
    ].copy()

    for ds, dmax in DMAX_MAIN.items():
        if ds == "fraud":
            continue
        subset = main_ca[main_ca["dataset"] == ds]
        if len(subset) == 0:
            continue
        mean_savings = subset["aggregate_carbon_saved_pct"].mean()
        exp_name     = f"{ds}_pareto" if ds != "ag_news" else "ag_news_pareto"
        new_rows = pd.DataFrame([{
            "experiment":                  exp_name,
            "is_pareto":                   True,
            "dataset":                     ds,
            "d_max_hours":                 dmax,
            "aggregate_carbon_saved_pct":  mean_savings,
        }])
        pareto = pd.concat([pareto, new_rows], ignore_index=True)

    pareto["dataset"] = pareto.apply(
        lambda r: infer_dataset_from_experiment(r) if pd.isna(r["dataset"]) else r["dataset"],
        axis=1,
    )

    datasets = ["ett", "cifar100", "ag_news"]
    markers  = {"ett": "o", "cifar100": "s", "ag_news": "^"}
    ds_colors = {"ett": "#457b9d", "cifar100": "#e76f51", "ag_news": "#2a9d8f"}

    fig, ax = plt.subplots(figsize=(9, 6))

    for ds in datasets:
        subset = (
            pareto[pareto["dataset"] == ds]
            .groupby("d_max_hours")["aggregate_carbon_saved_pct"]
            .mean()
            .reset_index()
            .sort_values("d_max_hours")
        )
        ax.plot(
            subset["d_max_hours"],
            subset["aggregate_carbon_saved_pct"],
            marker=markers[ds],
            color=ds_colors[ds],
            linewidth=2,
            markersize=7,
            label=DATASET_LABELS[ds],
        )

    ax.axhline(30, color="gray", linestyle=":", linewidth=1)
    ax.text(1, 31, "30% publishable floor", fontsize=9, color="gray")

    ax.set_xlabel("Maximum Delay Budget D_max (hours)", fontsize=12)
    ax.set_ylabel("Mean Carbon Savings (%)", fontsize=12)
    ax.set_title("Pareto Frontier: Carbon Savings vs Scheduling Flexibility\n(averaged over 3 seasonal windows \u00d7 3 seeds)", fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks([3, 6, 12, 24])

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "figure5_pareto_frontier.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def infer_dataset_from_experiment(row):
    exp = row["experiment"]
    for name in ["ag_news", "cifar100", "fraud", "ett"]:
        if name in exp:
            return name
    return None


def table1_main_results(df):
    """Table 1: mean carbon savings per dataset per approach with std."""
    main = df[df["is_pareto"] == False].copy()

    rows = []
    for ds in ["ett", "cifar100", "ag_news", "fraud"]:
        for approach in ["periodic", "drift_immediate", "carbon_aware"]:
            subset = main[(main["dataset"] == ds) & (main["approach"] == approach)]
            if len(subset) == 0:
                continue
            mean_s = subset["aggregate_carbon_saved_pct"].mean()
            std_s  = subset["aggregate_carbon_saved_pct"].std()
            rows.append({
                "Dataset":         DATASET_LABELS[ds],
                "Approach":        APPROACH_LABELS[approach],
                "Compute":         COMPUTE_TYPE[ds],
                "Urgency":         URGENCY[ds],
                "Mean Savings (%)": round(mean_s, 2),
                "Std (%)":         round(std_s, 2),
                "N runs":          len(subset),
            })

    t1 = pd.DataFrame(rows)
    path = os.path.join(FIGURES_DIR, "table1_main_results.csv")
    t1.to_csv(path, index=False)
    print(f"saved {path}")
    print(t1.to_string(index=False))


def table2_gpu_vs_cpu(df):
    """Table 2: GPU vs CPU comparison for carbon_aware approach."""
    main_ca = df[
        (df["is_pareto"] == False) &
        (df["approach"] == "carbon_aware")
    ].copy()

    rows = []
    for ds in ["ett", "cifar100", "ag_news", "fraud"]:
        subset = main_ca[main_ca["dataset"] == ds]
        if len(subset) == 0:
            continue
        rows.append({
            "Dataset":                   DATASET_LABELS[ds],
            "Compute":                   COMPUTE_TYPE[ds],
            "Energy (kWh)":              ENERGY_MAP[ds],
            "Mean Carbon Immediate (gCO2)": round(subset["total_carbon_immediate_gco2"].mean(), 2),
            "Mean Carbon Scheduled (gCO2)": round(subset["total_carbon_scheduled_gco2"].mean(), 2),
            "Mean Savings (%)":          round(subset["aggregate_carbon_saved_pct"].mean(), 2),
        })

    t2 = pd.DataFrame(rows)
    path = os.path.join(FIGURES_DIR, "table2_gpu_vs_cpu.csv")
    t2.to_csv(path, index=False)
    print(f"saved {path}")
    print(t2.to_string(index=False))


def table3_pareto_sensitivity(df):
    """Table 3: Pareto sensitivity — mean savings at each D_max per dataset."""
    pareto = df[df["is_pareto"] == True].copy()
    pareto["dataset"] = pareto.apply(
        lambda r: infer_dataset_from_experiment(r) if pd.isna(r["dataset"]) else r["dataset"],
        axis=1,
    )

    main_ca = df[
        (df["is_pareto"] == False) &
        (df["approach"] == "carbon_aware") &
        (df["dataset"] != "fraud")
    ].copy()

    rows = []
    for ds in ["ett", "cifar100", "ag_news"]:
        dmax_vals = sorted(
            list(pareto[pareto["dataset"] == ds]["d_max_hours"].unique()) +
            [DMAX_MAIN[ds]]
        )
        for dmax in dmax_vals:
            if dmax == DMAX_MAIN[ds]:
                subset = main_ca[main_ca["dataset"] == ds]
            else:
                subset = pareto[(pareto["dataset"] == ds) & (pareto["d_max_hours"] == dmax)]
            if len(subset) == 0:
                continue
            rows.append({
                "Dataset":          DATASET_LABELS[ds],
                "D_max (hours)":    int(dmax),
                "Mean Savings (%)": round(subset["aggregate_carbon_saved_pct"].mean(), 2),
                "Std (%)":          round(subset["aggregate_carbon_saved_pct"].std(), 2),
            })

    t3 = pd.DataFrame(rows)
    path = os.path.join(FIGURES_DIR, "table3_pareto_sensitivity.csv")
    t3.to_csv(path, index=False)
    print(f"saved {path}")
    print(t3.to_string(index=False))


if __name__ == "__main__":
    df = load_results(RESULTS_CSV)
    print(f"loaded {len(df)} rows\n")

    print("--- figure 1 ---")
    figure1_carbon_by_approach(df)

    print("--- figure 2 ---")
    figure2_tradeoff_scatter(df)

    print("--- figure 3 ---")
    figure3_carbon_intensity_timeseries()

    print("--- figure 4 ---")
    figure4_savings_by_window(df)

    print("--- figure 5 ---")
    figure5_pareto_frontier(df)

    print("--- table 1 ---")
    table1_main_results(df)

    print("--- table 2 ---")
    table2_gpu_vs_cpu(df)

    print("--- table 3 ---")
    table3_pareto_sensitivity(df)

    print("\ndone. all outputs in", FIGURES_DIR)