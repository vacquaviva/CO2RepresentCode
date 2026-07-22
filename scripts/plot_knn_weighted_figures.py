#!/usr/bin/env python3
"""Plot reconstructed kNN weighted-feature figures from saved score CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FEATURE_COUNT_FULL = 13
GOBM_MODELS = ["ETHZ", "FESOM", "NorESM", "MRI", "IPSL"]
GOBM_COLORS = {
    "SOCAT": "#35cbe8",
    "ETHZ": "blue",
    "FESOM": "red",
    "NorESM": "green",
    "MRI": "black",
    "IPSL": "magenta",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def plot_weighted_scores(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, (ax_mae, ax_r2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        "Model skill, kNN with weighted features; SOCAT data",
        fontsize=18,
        y=0.96,
    )

    x = metrics["n_features"].to_numpy()
    mae_mean = metrics["mae_weighted_mean"].to_numpy()
    mae_std = metrics["mae_weighted_std"].to_numpy()
    r2_mean = metrics["r2_weighted_mean"].to_numpy()
    r2_std = metrics["r2_weighted_std"].to_numpy()

    original_mae = metrics["mae_original_13d_mean"].iloc[0]
    original_r2 = metrics["r2_original_13d_mean"].iloc[0]

    ax_mae.fill_between(
        x,
        mae_mean - mae_std,
        mae_mean + mae_std,
        color="black",
        alpha=0.20,
    )
    ax_mae.plot(x, mae_mean, color="black", lw=1.8, label="MAE, optimal metric")
    ax_mae.scatter(
        [FEATURE_COUNT_FULL],
        [original_mae],
        color="black",
        s=80,
        label="MAE, orig space",
        zorder=3,
    )
    ax_mae.set_ylabel("Median Test Absolute Error", fontsize=20)
    ax_mae.set_ylim(0.0, 0.5)
    ax_mae.legend(frameon=True, loc="upper right", fontsize=18)

    ax_r2.fill_between(
        x,
        r2_mean - r2_std,
        r2_mean + r2_std,
        color="blue",
        alpha=0.20,
    )
    ax_r2.plot(x, r2_mean, color="blue", lw=1.8, label=r"R$^2$ score, optimal metric")
    ax_r2.scatter(
        [FEATURE_COUNT_FULL],
        [original_r2],
        color="blue",
        s=80,
        label=r"R$^2$ score, orig space",
        zorder=3,
    )
    ax_r2.set_ylabel(r"Test R$^2$ score", fontsize=20)
    ax_r2.set_xlabel("Number of features", fontsize=20)
    ax_r2.set_ylim(0.4, 1.0)
    ax_r2.set_xticks(range(4, 14))
    ax_r2.legend(frameon=True, loc="lower left", fontsize=18)

    for ax in (ax_mae, ax_r2):
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(axis="both", labelsize=18)

    fig.tight_layout()
    fig.savefig(
        output_dir / "knn_weighted_scores_seed13.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)


def plot_model_weights(metrics: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(
        "Model skill, kNN with weighted features derived from GOBMs",
        fontsize=20,
    )

    for weight_source in ["SOCAT", *GOBM_MODELS]:
        subset = metrics[metrics["weight_source"] == weight_source]
        if subset.empty:
            raise ValueError(f"Missing rows for weight source: {weight_source}")

        label = "weights from SOCAT" if weight_source == "SOCAT" else f"weights from {weight_source}"
        ax.plot(
            subset["n_features"],
            subset["mae_mean"],
            color=GOBM_COLORS[weight_source],
            lw=2.0,
            label=label,
        )

    ax.set_xlabel("Number of features", fontsize=22)
    ax.set_ylabel("Median Absolute Error", fontsize=22)
    ax.set_xlim(3, 14)
    ax.set_ylim(0.20, 0.40)
    ax.set_yticks([0.20, 0.25, 0.30, 0.35, 0.40])
    ax.set_xticks([4, 6, 8, 10, 12, 14])
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(frameon=True, loc="upper right", fontsize=20)

    fig.tight_layout()
    fig.savefig(
        output_dir / "kNN_weights_models_seed13.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reconstructed kNN figures.")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediates"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_metrics = pd.read_csv(
        require(args.intermediate_dir / "knn_weighted_scores_seed13_metrics.csv")
    )
    model_weight_metrics = pd.read_csv(
        require(args.intermediate_dir / "kNN_weights_models_seed13_metrics.csv")
    )

    plot_weighted_scores(score_metrics, args.output_dir)
    plot_model_weights(model_weight_metrics, args.output_dir)


if __name__ == "__main__":
    main()
