#!/usr/bin/env python3
"""Reproduce the three-panel SOCAT subset map figures.

This script uses cached point-location CSVs so users do not have to rebuild
large xarray data frames from GCS just to draw the maps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cf
except ImportError as exc:
    raise SystemExit("cartopy is required for the map panels") from exc

try:
    import cmocean.cm as cm
except ImportError:
    cm = plt


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def read_points(base_dir: Path, filename: str) -> pd.DataFrame:
    df = pd.read_csv(require(base_dir / filename))
    missing = {"lat_deg", "lon_deg"} - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {sorted(missing)}")
    return df


def loadtxt(base_dir: Path, filename: str) -> np.ndarray:
    return np.loadtxt(require(base_dir / filename))


def importance_bootstrap(values: np.ndarray, weights: np.ndarray, n: int | None = None, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = weights / weights.sum()
    idx = rng.choice(len(values), size=(len(values) if n is None else n), replace=True, p=probabilities)
    return values[idx]


def balance_cmap():
    return getattr(cm, "balance", "viridis")


def plot_uniform_marginals(intermediate_dir: Path, output_dir: Path) -> None:
    source = pd.read_csv(require(intermediate_dir / "SOCAT2_spatial_with_OTweights.csv"))
    target = pd.read_csv(require(intermediate_dir / "Uniform_split_spatial.csv"))
    cols = ["A", "B", "C"]
    for col in cols + ["uot_weight", "final_weight"]:
        if col not in source:
            raise ValueError(f"SOCAT2_spatial_with_OTweights.csv is missing column: {col}")
    for col in cols:
        if col not in target:
            raise ValueError(f"Uniform_split_spatial.csv is missing column: {col}")

    x_source = source[cols].to_numpy()
    x_target = target[cols].to_numpy()
    w_uot = source["uot_weight"].to_numpy()
    w_final = source["final_weight"].to_numpy()

    def hist_panel(axes, x_src, x_tgt, w_raw, w_ipf, name):
        bins = 50
        axes.hist(x_src, bins=bins, density=True, histtype="step", label="source")
        axes.hist(
            importance_bootstrap(x_src.reshape(-1, 1), w_raw, seed=10)[:, 0],
            bins=bins,
            density=True,
            histtype="step",
            label="source UOT",
        )
        axes.hist(
            importance_bootstrap(x_src.reshape(-1, 1), w_ipf, seed=11)[:, 0],
            bins=bins,
            density=True,
            histtype="step",
            label="source UOT+IPF",
        )
        axes.hist(x_tgt, bins=bins, density=True, histtype="step", label="target")
        axes.set_title(f"1D marginal: {name}")

    fig, axs = plt.subplots(1, len(cols), figsize=(10, 3))
    for j, col in enumerate(cols):
        hist_panel(axs[j], x_source[:, j], x_target[:, j], w_uot, w_final, col)
    axs[0].legend()
    plt.tight_layout()
    plt.savefig(output_dir / "SOCAT_Uniform_Marginals.png", bbox_inches="tight", pad_inches=0.2)


def draw_three_panel(
    left_points: pd.DataFrame,
    right_points: pd.DataFrame,
    left_title: str,
    right_title: str,
    left_imbs: np.ndarray,
    right_imbs: np.ndarray,
    left_label: str,
    right_label: str,
    right_color: str,
    output_path: Path,
    suptitle: str | None = None,
    pad_inches: float = 0.1,
) -> None:
    import matplotlib.gridspec as gridspec

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(24, 8), constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=24, y=0.87)

    gs = gridspec.GridSpec(1, 3, width_ratios=[2, 2, 1.3], figure=fig)
    ax0 = fig.add_subplot(gs[0], projection=proj)
    ax1 = fig.add_subplot(gs[1], projection=proj)
    ax2 = fig.add_subplot(gs[2])

    for ax in [ax0, ax1]:
        ax.coastlines(resolution="50m", lw=0.2)
        ax.add_feature(cf.LAND, facecolor="lightgrey", alpha=0.3)

    hb = ax0.hexbin(
        left_points["lon_deg"],
        left_points["lat_deg"],
        gridsize=500,
        cmap=balance_cmap(),
        bins="log",
        vmin=0.1,
        vmax=100,
        transform=ccrs.PlateCarree(),
    )
    cb = plt.colorbar(hb, ax=ax0, shrink=0.5, label="log(point density)")
    cb.set_label("log(point density)", fontsize=14)
    cb.ax.tick_params(labelsize=14)
    ax0.set_title(left_title, fontsize=18)

    ax1.hexbin(
        right_points["lon_deg"],
        right_points["lat_deg"],
        gridsize=500,
        cmap=balance_cmap(),
        bins="log",
        vmin=0.1,
        vmax=100,
        transform=ccrs.PlateCarree(),
    )
    ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax1.set_title(right_title, fontsize=18)

    ax2.set_aspect(6.5)
    ax2.plot(np.arange(len(left_imbs), 0, -1), np.nan_to_num(left_imbs, nan=1.0), "d-", c="k", lw=1.2, label=left_label)
    ax2.plot(np.arange(len(right_imbs), 0, -1), np.nan_to_num(right_imbs, nan=1.0), "x--", c=right_color, lw=1.2, label=right_label)
    ax2.legend(fontsize=16)
    ax2.set_xlim(2, 13)
    ax2.set_ylim(0.18, 1.42)
    ax2.tick_params(labelsize=12)
    ax2.set_xlabel("# of features", fontsize=16)
    ax2.set_ylabel("DII", fontsize=16)

    plt.savefig(output_path, bbox_inches="tight", pad_inches=pad_inches)


def plot_uniform(intermediate_dir: Path, output_dir: Path) -> None:
    draw_three_panel(
        read_points(intermediate_dir, "SOCAT1_points.csv"),
        read_points(intermediate_dir, "SOCAT2_resampled_points.csv"),
        "Density of original SOCAT data points",
        r"Density of resampled $\rightarrow$ uniform SOCAT data points",
        loadtxt(intermediate_dir, "standardimbs_SOCAT1_my_pipeline.txt"),
        loadtxt(intermediate_dir, "imbs_weightedSOCAT2_fromSOCAT1.txt"),
        "SOCAT 1",
        r"SOCAT 2, resampled $\rightarrow$ uniform",
        "b",
        output_dir / "3panels_SOCAT_orig_unif_v2.png",
        suptitle="Transferability of feature weights across subsets of SOCAT observations",
        pad_inches=0.2,
    )


def plot_na(intermediate_dir: Path, output_dir: Path) -> None:
    draw_three_panel(
        read_points(intermediate_dir, "SM_minus_NA_sample_points.csv"),
        read_points(intermediate_dir, "NA_STPS_points.csv"),
        "Density of SOCAT data points, excluding NA STPS",
        "Density of NA STPS data points",
        loadtxt(intermediate_dir, "StandardII_imbs_SOCAT_minus_NA.txt"),
        loadtxt(intermediate_dir, "StandardII_imbs_NA_STPS.txt"),
        "SOCAT excl. NA STPS",
        "NA STPS",
        "g",
        output_dir / "3panels_SOCAT_vs_NA.png",
    )


def plot_so(intermediate_dir: Path, output_dir: Path) -> None:
    draw_three_panel(
        read_points(intermediate_dir, "SM_minus_SO_sample_points.csv"),
        read_points(intermediate_dir, "SO_SPSS_points.csv"),
        "Density of SOCAT data points, excluding SO SPSS",
        "Density of SO SPSS data points",
        loadtxt(intermediate_dir, "StandardII_imbs_SOCAT_minus_SO.txt"),
        loadtxt(intermediate_dir, "StandardII_imbs_SO_SPSS.txt"),
        "SOCAT excl. SO SPSS",
        "SO SPSS",
        "m",
        output_dir / "3panels_SOCAT_vs_SO.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot three-panel SOCAT subset map figures.")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediates"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--only", choices=["all", "marginals", "uniform", "na", "so"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "marginals"):
        plot_uniform_marginals(args.intermediate_dir, args.output_dir)
    if args.only in ("all", "uniform"):
        plot_uniform(args.intermediate_dir, args.output_dir)
    if args.only in ("all", "na"):
        plot_na(args.intermediate_dir, args.output_dir)
    if args.only in ("all", "so"):
        plot_so(args.intermediate_dir, args.output_dir)


if __name__ == "__main__":
    main()
