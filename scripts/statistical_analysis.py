import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RESULTS_CSV = Path(__file__).parent.parent / "experiments" / "results" / "simulation_results.csv"
OUTPUT_DIR  = Path(__file__).parent.parent / "experiments" / "results"

DATASETS_FOR_TEST = ["ett_simulation", "cifar100_simulation", "ag_news_simulation"]


def load_results():
    df = pd.read_csv(RESULTS_CSV)
    df["aggregate_carbon_saved_pct"]  = pd.to_numeric(df["aggregate_carbon_saved_pct"],  errors="coerce")
    df["total_carbon_immediate_gco2"] = pd.to_numeric(df["total_carbon_immediate_gco2"], errors="coerce")
    df["total_carbon_scheduled_gco2"] = pd.to_numeric(df["total_carbon_scheduled_gco2"], errors="coerce")
    df["retrain_count"]               = pd.to_numeric(df["retrain_count"],               errors="coerce")
    df["window"]                      = pd.to_numeric(df["window"],                      errors="coerce")
    df["seed"]                        = pd.to_numeric(df["seed"],                        errors="coerce")
    return df


def bootstrap_ci(values, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return float(np.mean(values)), float(lo), float(hi)


def wilcoxon_test(carbon_aware_vals, drift_immediate_vals):
    differences = np.array(carbon_aware_vals) - np.array(drift_immediate_vals)
    if np.all(differences == 0):
        return None, None
    stat, p = stats.wilcoxon(differences, alternative="greater")
    return float(stat), float(p)


def main():
    df = load_results()

    carbon_aware     = df[df["approach"] == "carbon_aware"].copy()
    drift_immediate  = df[df["approach"] == "drift_immediate"].copy()
    periodic         = df[df["approach"] == "periodic"].copy()

    summary_rows = []
    wilcoxon_rows = []

    for experiment in df["experiment"].unique():
        ca  = carbon_aware[carbon_aware["experiment"] == experiment].sort_values(["window", "seed"])
        di  = drift_immediate[drift_immediate["experiment"] == experiment].sort_values(["window", "seed"])
        per = periodic[periodic["experiment"] == experiment].sort_values(["window", "seed"])

        ca_savings = ca["aggregate_carbon_saved_pct"].dropna().values
        di_savings = di["aggregate_carbon_saved_pct"].dropna().values

        mean, lo, hi = bootstrap_ci(ca_savings)

        summary_rows.append({
            "experiment":           experiment,
            "n_runs":               len(ca_savings),
            "mean_carbon_saved_pct": round(mean, 2),
            "ci_95_lo":             round(lo, 2),
            "ci_95_hi":             round(hi, 2),
            "min_carbon_saved_pct": round(ca_savings.min(), 2),
            "max_carbon_saved_pct": round(ca_savings.max(), 2),
            "mean_retrain_count":   round(ca["retrain_count"].mean(), 1),
            "mean_carbon_immediate": round(ca["total_carbon_immediate_gco2"].mean(), 2),
            "mean_carbon_scheduled": round(ca["total_carbon_scheduled_gco2"].mean(), 2),
        })

        if experiment not in DATASETS_FOR_TEST:
            continue

        if len(ca_savings) != len(di_savings):
            print(f"warning: {experiment} carbon_aware ({len(ca_savings)}) and drift_immediate ({len(di_savings)}) run counts differ - skipping wilcoxon")
            continue

        stat, p = wilcoxon_test(ca_savings, di_savings)
        if stat is None:
            print(f"warning: {experiment} all differences are zero - skipping wilcoxon")
            continue

        wilcoxon_rows.append({
            "experiment":    experiment,
            "wilcoxon_stat": round(stat, 4),
            "p_value":       round(p, 6),
            "significant":   p < 0.05,
            "n_pairs":       len(ca_savings),
        })

    summary_df  = pd.DataFrame(summary_rows)
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)

    print("\n--- carbon savings summary (carbon_aware) ---")
    print(summary_df.to_string(index=False))

    print("\n--- wilcoxon signed-rank test: carbon_aware vs drift_immediate ---")
    print(wilcoxon_df.to_string(index=False))

    summary_df.to_csv(OUTPUT_DIR  / "summary_stats.csv",   index=False)
    wilcoxon_df.to_csv(OUTPUT_DIR / "wilcoxon_results.csv", index=False)
    print(f"\nsaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
