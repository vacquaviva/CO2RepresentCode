#!/usr/bin/env python3
"""Reproduce SamplingDomains.png from saved stratified intermediates."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ["ETHZ", "FESOM", "NorESM", "MRI", "IPSL"]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def read_csv(base_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(require(base_dir / filename))


def loadtxt_any(intermediate_dir: Path, romy_dir: Path, filename: str) -> np.ndarray:
    candidates = [
        intermediate_dir / filename,
        romy_dir / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return np.loadtxt(candidate)
    raise FileNotFoundError(
        "Required input not found; checked: " + ", ".join(str(path) for path in candidates)
    )


def value_at_scale(scales: pd.DataFrame, dims: pd.DataFrame, model: str, scale: float = 2.5) -> float:
    idx = (scales[model] - scale).abs().idxmin()
    return float(dims[model][idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SamplingDomains.png.")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediates"))
    parser.add_argument("--romy-dir", type=Path, default=Path("intermediates/FromRomy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=MODELS,
        help=(
            "Model names to include. Defaults to all models; use this to make a "
            "temporary diagnostic figure when one model's intermediates are missing."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = args.models

    df_01_dim_all = read_csv(args.intermediate_dir, "df_01_dim_all_seed10.csv")
    df_01_scales_all = read_csv(args.intermediate_dir, "df_01_scales_all_seed10.csv")
    dsocat_dim_all = read_csv(args.intermediate_dir, "dSOCAT_dim_all.csv")
    dsocat_scales_all = read_csv(args.intermediate_dir, "dSOCAT_scales_all.csv")
    south_dim_all = read_csv(args.intermediate_dir, "south_dim_all.csv")
    south_scales_all = read_csv(args.intermediate_dir, "south_scales_all.csv")
    north_dim_all = read_csv(args.intermediate_dir, "north_dim_all.csv")
    north_scales_all = read_csv(args.intermediate_dir, "north_scales_all.csv")

    results = {model: value_at_scale(df_01_scales_all, df_01_dim_all, model) for model in models}
    results_socat = {
        model: value_at_scale(dsocat_scales_all, dsocat_dim_all, model) for model in models
    }
    results_south = {
        model: value_at_scale(south_scales_all, south_dim_all, model) for model in models
    }
    results_north = {
        model: value_at_scale(north_scales_all, north_dim_all, model) for model in models
    }

    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.suptitle("Comparison of GOBMs across sampling domains", fontsize=15, y=0.96)

    for j, model in enumerate(models):
        ax_left.barh(j + 0.30, results[model], height=0.1, alpha=0.6, color=colors[0], label="Uniform" if j == 0 else None)
        ax_left.barh(j + 0.15, results_socat[model], height=0.1, alpha=0.6, color=colors[1], label="SOCAT" if j == 0 else None)
        ax_left.barh(j, results_north[model], height=0.1, alpha=0.6, color=colors[3], label="North" if j == 0 else None)
        ax_left.barh(j - 0.15, results_south[model], height=0.1, alpha=0.6, color=colors[2], label="South" if j == 0 else None)

    ax_left.set_yticks(range(len(models)))
    ax_left.set_yticklabels(models, fontsize=14)
    ax_left.set_xlabel("Intrinsic Dimension at scale 2.5", fontsize=14)
    ax_left.set_xlim(3.5, 5.5)
    ax_left.set_xticks([3.5, 4.0, 4.5, 5.0, 5.5])

    ax2 = ax_right.twinx()
    ax2.yaxis.set_ticks_position("right")
    ax2.yaxis.set_label_position("right")
    ax_right.yaxis.set_ticks_position("none")

    for j, model in enumerate(models):
        imbs_uniform = loadtxt_any(args.intermediate_dir, args.romy_dir, f"Finalimbs_sampled16k_{model}.txt")
        imbs_socat = loadtxt_any(args.intermediate_dir, args.romy_dir, f"Finalimbs_sampled1_seed10_{model}.txt")
        ax2.barh(j + 0.30, imbs_uniform[0], height=0.1, alpha=0.6, color=colors[0], label="Uniform" if j == 0 else None)
        ax2.barh(j + 0.15, imbs_socat[0], height=0.1, alpha=0.6, color=colors[1], label="SOCAT" if j == 0 else None)

    for j, model in enumerate(models):
        imbs_south = loadtxt_any(args.intermediate_dir, args.romy_dir, f"Finalimbs_sampled16k_South_{model}.txt")
        imbs_north = loadtxt_any(args.intermediate_dir, args.romy_dir, f"Finalimbs_sampled16k_North_{model}.txt")
        ax2.barh(j, imbs_north[0], height=0.1, alpha=0.6, color=colors[3], label="North" if j == 0 else None)
        ax2.barh(j - 0.15, imbs_south[0], height=0.1, alpha=0.6, color=colors[2], label="South" if j == 0 else None)

    ax2.set_yticks(range(len(models)))
    ax2.set_yticklabels(models, fontsize=14)
    ax2.set_xlim(0, 0.35)
    ax2.invert_xaxis()
    ax2.legend(loc="lower left", frameon=False, fontsize=13)
    ax_right.set_xlabel(r"Residual DII (error) for $n=13$", fontsize=14)
    ax_right.tick_params(axis="y", left=False, labelleft=False)

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(args.output_dir / "SamplingDomains.png", dpi=300, bbox_inches="tight", pad_inches=0.2)


if __name__ == "__main__":
    main()
