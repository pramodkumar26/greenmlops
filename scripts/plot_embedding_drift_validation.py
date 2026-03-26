"""
plot_embedding_drift_validation.py

Generates the embedding drift validation figure for Checkpoint 2.
Shows clean vs drifted MMD scores against the detection threshold
for CIFAR-100 (ResNet-18) and AG News (DistilBERT).

Numbers are from Week 5 Colab validation runs, logged in MLflow.
Run locally in the mlops conda environment -- no GPU needed.

Output: figures/embedding_drift_validation.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_DIR = "figures"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embedding_drift_validation.png")

# Week 5 validation results
# Source: Colab validation runs, real model checkpoints
RESULTS = {
    "CIFAR-100\n(ResNet-18)": {
        "clean_mmd":   0.054,
        "drifted_mmd": 0.430,
        "threshold":   0.076,
        "pca_var":     0.84,
    },
    "AG News\n(DistilBERT)": {
        "clean_mmd":   0.029,
        "drifted_mmd": 0.500,
        "threshold":   0.104,
        "pca_var":     0.93,
    },
}

COLOR_CLEAN   = "#2196F3"
COLOR_DRIFTED = "#F44336"
COLOR_THRESH  = "#333333"


def build_figure(results):
    datasets = list(results.keys())
    n        = len(datasets)

    fig, axes = plt.subplots(1, n, figsize=(9, 4.5), sharey=False)
    fig.suptitle(
        "Embedding Drift Validation — MMD Score: Clean vs Drifted",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for ax, dataset in zip(axes, datasets):
        vals      = results[dataset]
        clean     = vals["clean_mmd"]
        drifted   = vals["drifted_mmd"]
        threshold = vals["threshold"]
        pca_var   = vals["pca_var"]

        x      = [0.3, 0.7]
        scores = [clean, drifted]
        colors = [COLOR_CLEAN, COLOR_DRIFTED]

        bars = ax.bar(x, scores, width=0.25, color=colors, zorder=3)

        ax.axhline(
            threshold,
            color=COLOR_THRESH,
            linewidth=1.4,
            linestyle="--",
            zorder=4,
            label=f"Threshold = {threshold:.3f}",
        )

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                score + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.text(
            0.98, threshold + 0.012,
            f"threshold = {threshold:.3f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color=COLOR_THRESH,
        )

        ax.set_title(
            f"{dataset}\nPCA explained variance: {pca_var:.0%}",
            fontsize=11,
            pad=10,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Clean\nwindow", "Drifted\nwindow"], fontsize=10)
        ax.set_ylabel("MMD Score", fontsize=10)
        ax.set_ylim(0, max(drifted * 1.25, threshold * 1.5))
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        no_trigger = mpatches.Patch(color=COLOR_CLEAN,   label="Clean — no trigger")
        triggered  = mpatches.Patch(color=COLOR_DRIFTED, label="Drifted — trigger")
        ax.legend(handles=[no_trigger, triggered], fontsize=8, loc="upper left")

    fig.tight_layout()
    return fig


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig = build_figure(RESULTS)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()