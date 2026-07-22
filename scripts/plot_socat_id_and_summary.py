#!/usr/bin/env python3
"""Reproduce SOCAT intrinsic-dimension figures from saved intermediates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def normalize_weight_rows(weights: np.ndarray) -> np.ndarray:
    weights = weights.copy()
    for i in range(1, 12):
        weights[i - 1] = weights[i - 1] / np.linalg.norm(weights[i - 1])
    return weights


def plot_id_socat_square(intermediate_dir: Path, output_dir: Path) -> None:
    socat_dim_all = read_csv(intermediate_dir, "SOCATobs_dim_all.csv")
    socat_dim_input = read_csv(intermediate_dir, "SOCATobs_dim_input.csv")
    socat_scales_all = read_csv(intermediate_dir, "SOCATobs_scales_all.csv")
    socat_scales_input = read_csv(intermediate_dir, "SOCATobs_scales_input.csv")

    ids_gride_data = socat_dim_all["SOCAT"]
    ids_gride_in_data = socat_dim_input["SOCAT"]
    scales_gride_data = socat_scales_all["SOCAT"]
    scales_gride_in_data = socat_scales_input["SOCAT"]

    plt.figure(figsize=(6, 6))
    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))

    plt.plot(scales_gride_data, ids_gride_data, alpha=0.85, color=oceancolors[2])
    plt.scatter(
        scales_gride_data,
        ids_gride_data,
        edgecolors="k",
        color=oceancolors[2],
        s=50,
    )

    plt.plot(
        scales_gride_in_data,
        ids_gride_in_data,
        "--",
        alpha=0.85,
        lw=2,
        color=oceancolors[2],
        label="SOCAT obs (features only)",
    )
    plt.scatter(
        scales_gride_in_data,
        ids_gride_in_data,
        edgecolors="k",
        color=oceancolors[2],
        s=50,
        label=r"SOCAT obs (features+$\delta fCO_2$)",
    )

    plt.xlabel(r"Scale", size=14)
    plt.ylabel("Estimated ID", size=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.legend(frameon=False, fontsize=13)
    plt.savefig(
        output_dir / "ID_SOCAT_square.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )


def plot_socat_three_panel_summary(
    intermediate_dir: Path, weights_dir: Path, output_dir: Path
) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(21, 5), gridspec_kw={"width_ratios": [1, 1.6, 1]}
    )

    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))

    socat_dim_all = read_csv(intermediate_dir, "SOCATobs_dim_all.csv")
    socat_scales_all = read_csv(intermediate_dir, "SOCATobs_scales_all.csv")
    socat_full_dim_all = read_csv(intermediate_dir, "SOCATobs_full_dim_all.csv")
    socat_full_scales_all = read_csv(intermediate_dir, "SOCATobs_full_scales_all.csv")

    ax1.plot(
        socat_scales_all["SOCAT"],
        socat_dim_all["SOCAT"],
        marker="o",
        markeredgecolor="k",
        alpha=0.85,
        color=oceancolors[2],
        markersize=5,
        lw=1.5,
        label="n = 16,000",
    )

    ax1.plot(
        socat_full_scales_all["SOCAT"],
        socat_full_dim_all["SOCAT"],
        linestyle="-.",
        marker="d",
        markeredgecolor="k",
        color="purple",
        markersize=5,
        lw=1.5,
        label="n = 28,621 (all)",
    )

    ax1.legend(fontsize=14, frameon=False, loc="upper left")
    ax1.set_title("Intrinsic dimension", fontsize=14)
    ax1.set_xlabel(r"Scale", size=14)
    ax1.set_ylabel("Estimated ID", size=14)

    weightsfiles16 = [
        "Finalweights_SOCAT16k_v2.txt",
        "Finalweights_SOCAT16k_v2_seed11.txt",
        "Finalweights_SOCAT16k_v2_seed12.txt",
        "Finalweights_SOCAT16k_v2_seed13.txt",
        "Finalweights_SOCAT16k_v2_seed14.txt",
    ]

    normweights16 = [
        normalize_weight_rows(loadtxt(weights_dir, filename))
        for filename in weightsfiles16
    ]
    normweights20 = [
        normalize_weight_rows(loadtxt(weights_dir, "Finalweights_SOCAT20k_v2_seed13.txt"))
    ]

    i = 1
    ax2.bar(
        np.arange(13) - 0.15,
        np.array(normweights16).mean(axis=0)[i - 1],
        width=0.2,
        alpha=0.7,
        color=oceancolors[2],
        label="n = 16,000",
    )
    ax2.errorbar(
        np.arange(13) - 0.15,
        np.array(normweights16).mean(axis=0)[i - 1],
        yerr=np.array(normweights16).std(axis=0)[i - 1],
        fmt="none",
        lw=2,
        color="gray",
        capsize=3,
    )
    ax2.bar(
        np.arange(13) + 0.15,
        np.array(normweights20).mean(axis=0)[i - 1],
        width=0.2,
        alpha=0.7,
        color="purple",
        label="n = 20,000",
    )
    ax2.set_xticks(np.arange(13))
    ax2.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=12)
    ax2.annotate(r"$\rho$ = 0.992", xy=(0.04, 0.9), xycoords="axes fraction", fontsize=14)
    ax2.legend(fontsize=14, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax2.set_title("Feature weights", fontsize=14)

    imbs_list = [
        loadtxt(weights_dir, "Finalimbs_SOCAT16k_v2.txt"),
        loadtxt(weights_dir, "Finalimbs_SOCAT16k_v2_seed11.txt"),
        loadtxt(weights_dir, "Finalimbs_SOCAT16k_v2_seed12.txt"),
        loadtxt(weights_dir, "Finalimbs_SOCAT16k_v2_seed13.txt"),
        loadtxt(weights_dir, "Finalimbs_SOCAT16k_v2_seed14.txt"),
    ]

    mean = np.array([np.nan_to_num(imbs, nan=1.0) for imbs in imbs_list]).mean(axis=0)
    stdev = np.array([np.nan_to_num(imbs, nan=1.0) for imbs in imbs_list]).std(axis=0)

    imbs_socat_13_20k = loadtxt(weights_dir, "Finalimbs_SOCAT20k_v2_seed13.txt")
    mean20 = np.array([np.nan_to_num(imbs_socat_13_20k, nan=1.0)]).mean(axis=0)
    stdev20 = np.array([np.nan_to_num(imbs_socat_13_20k, nan=1.0)]).std(axis=0)

    ax3.plot(
        np.arange(len(mean), 0, -1),
        mean,
        "o-",
        markersize=5,
        lw=1.5,
        label="n = 16,000",
        color=oceancolors[2],
    )
    ax3.errorbar(
        np.arange(len(mean), 0, -1),
        mean,
        yerr=stdev,
        fmt="none",
        color="gray",
        capsize=3,
    )
    ax3.plot(
        np.arange(len(mean20), 0, -1),
        mean20,
        "d-.",
        markersize=5,
        lw=1.5,
        label="n = 20,000",
        color="purple",
    )
    ax3.errorbar(
        np.arange(len(mean20), 0, -1),
        mean20,
        yerr=stdev20,
        fmt="none",
        color="gray",
        capsize=3,
    )
    ax3.set_xlim(2, 13)
    ax3.set_xlabel("# of features", fontsize=14)
    ax3.set_ylabel("DII", fontsize=14)
    ax3.legend(fontsize=14, frameon=False)
    ax3.set_title("DII", fontsize=14)

    for ax, label in zip([ax1, ax2, ax3], ["a", "b", "c"]):
        ax.tick_params(labelsize=12)
        ax.text(
            0.02,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="bottom",
        )

    plt.subplots_adjust(wspace=0.15)
    plt.savefig(
        output_dir / "SOCAT_three_panel_summary.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SOCAT ID square and three-panel summary figures."
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("intermediates"),
        help="Directory containing saved SOCAT ID CSV files.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("intermediates"),
        help="Directory containing saved SOCAT weights and DII text files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where figures will be written.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_id_socat_square(args.intermediate_dir, args.output_dir)
    plot_socat_three_panel_summary(args.intermediate_dir, args.weights_dir, args.output_dir)


if __name__ == "__main__":
    main()
