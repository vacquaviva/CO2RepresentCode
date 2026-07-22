#!/usr/bin/env python3
"""Reproduce combined_ID_weights_DII.png from saved intermediates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ["ETHZ", "FESOM", "NorESM", "MRI", "IPSL"]
COLORS = ["blue", "red", "green", "black", "magenta"]
LABELS = [
    r"$\bf{SST}$",
    r"$\bf{SST\_an}$",
    r"$\bf{SSS}$",
    r"$\bf{SSS\_an}$",
    r"$\bf{Chl}$",
    r"$\bf{Chl\_an}$",
    r"$\bf{MLD}$",
    r"$\bf xCO_2$",
    r"$\bf{A}$",
    r"$\bf{B}$",
    r"$\bf{C}$",
    r"$\bf{T_0}$",
    r"$\bf{T_1}$",
]


def read_csv(base_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(base_dir / filename)


def loadtxt(base_dir: Path, filename: str) -> np.ndarray:
    return np.loadtxt(base_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the GOBM convergence summary figure."
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("intermediates"),
        help="Directory containing saved convergence CSV files.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("intermediates"),
        help="Directory containing saved weights and DII text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the figure will be written.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 9))

    gs = fig.add_gridspec(
        2, 5,
        height_ratios=[1, 1.45],
        hspace=0.32,
        wspace=0.15,
    )
    top_axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    bottom_gs = gs[1, :].subgridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)
    ax_weights = fig.add_subplot(bottom_gs[0, :2])
    ax_dii = fig.add_subplot(bottom_gs[0, 2])

    fig.subplots_adjust(top=0.92)
    fig.text(0.50, 0.96, "      Intrinsic dimension", ha="center", fontsize=16)

    df_01_dim_all = read_csv(args.intermediate_dir, "df_01_dim_all_seed10.csv")
    df_01_scales_all = read_csv(args.intermediate_dir, "df_01_scales_all_seed10.csv")
    df_02_dim_all = read_csv(args.intermediate_dir, "df_02_dim_all_seed10.csv")
    df_02_scales_all = read_csv(args.intermediate_dir, "df_02_scales_all_seed10.csv")

    for i, model in enumerate(MODELS):
        ax = top_axes[i]

        ax.plot(
            df_01_scales_all[model],
            df_01_dim_all[model],
            marker="o",
            markeredgecolor="k",
            alpha=0.85,
            color=COLORS[i],
            markersize=5,
            label="n = 16,000",
        )
        ax.plot(
            df_02_scales_all[model],
            df_02_dim_all[model],
            linestyle="-.",
            marker="d",
            markeredgecolor="k",
            alpha=0.85,
            color=COLORS[i],
            markersize=5,
            label="n = 32,000",
        )
        ax.set_xlabel(r"Scale", size=13)
        if i == 0:
            ax.set_ylabel("Estimated ID", size=13)
        ax.annotate(
            model,
            xy=(0.07, 0.9),
            xycoords="axes fraction",
            color=COLORS[i],
            fontsize=13,
        )
        ax.set_ylim(1.5, 6)
        ax.set_xlim(0.1, 3.3)
        ax.tick_params(labelsize=11)
        ax.legend(fontsize=12, frameon=False, loc="lower right")

    feature_row = 1
    weights_16k = pd.DataFrame()
    weights_20k = pd.DataFrame()

    for j, model in enumerate(MODELS):
        weights = loadtxt(args.weights_dir, "Finalweights_sampled16k_" + model + ".txt")
        weights[feature_row - 1] = weights[feature_row - 1] / np.linalg.norm(
            weights[feature_row - 1]
        )
        weights_16k[model] = weights[feature_row - 1]

        weights20 = loadtxt(args.weights_dir, "Finalweights_sampled20k_" + model + ".txt")
        weights20[feature_row - 1] = weights20[feature_row - 1] / np.linalg.norm(
            weights20[feature_row - 1]
        )
        weights_20k[model] = weights20[feature_row - 1]

        rho = np.corrcoef(weights_16k[model], weights_20k[model])[0, 1]
        ax_weights.annotate(
            rf"{model}, $\rho={rho:.2f}$",
            xy=(0.2, 0.9 - j * 0.08),
            xycoords="axes fraction",
            fontsize=14,
            color=COLORS[j],
        )

        x = np.arange(13) - 0.3 + j * 0.15
        ax_weights.bar(
            x,
            weights[feature_row - 1],
            width=0.1,
            alpha=0.5,
            color=COLORS[j],
            label=model + ", n = 16,000" if j == 0 else None,
        )
        ax_weights.bar(
            x,
            weights20[feature_row - 1],
            width=0.1,
            fill=False,
            edgecolor=COLORS[j],
            label=model + ", n = 20,000" if j == 0 else None,
        )

    ax_weights.set_title("Feature weights", fontsize=16)
    ax_weights.set_xticks(np.arange(13))
    ax_weights.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=13)
    ax_weights.tick_params(axis="y", labelsize=12)
    ax_weights.legend(fontsize=14, loc="upper right", frameon=False, ncol=2)

    for i, model in enumerate(MODELS):
        if i == 1 or i == 2:
            imbs = loadtxt(args.weights_dir, "Finalimbs_sampled16k_" + model + ".txt")
            ax_dii.plot(
                np.arange(len(imbs), 0, -1),
                imbs,
                "o-",
                markersize=4,
                lw=1,
                label=model + ", n = 16,000",
                color=COLORS[i],
            )

            imbs = loadtxt(args.weights_dir, "Finalimbs_sampled20k_" + model + ".txt")
            ax_dii.plot(
                np.arange(len(imbs), 0, -1),
                imbs,
                "d-.",
                markersize=4,
                lw=1,
                label=model + ", n = 20,000",
                color=COLORS[i],
            )

    ax_dii.set_xlabel("# of features", fontsize=14)
    ax_dii.set_ylabel("DII", fontsize=14)
    ax_dii.tick_params(labelsize=12)
    ax_dii.legend(fontsize=14, frameon=False)
    ax_dii.set_title("DII", fontsize=16)

    all_axes = top_axes + [ax_weights, ax_dii]
    panel_labels = ["a", "b", "c", "d", "e", "f", "g"]
    for ax, panel_label in zip(all_axes, panel_labels):
        ax.text(
            0.02,
            1.05,
            panel_label,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="bottom",
        )

    plt.savefig(
        args.output_dir / "combined_ID_weights_DII.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )


if __name__ == "__main__":
    main()
