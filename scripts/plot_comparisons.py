#!/usr/bin/env python3
"""Reproduce listed figures from Comparisons.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


MODELS = ["ETHZ", "FESOM", "NorESM", "MRI", "IPSL"]
COLORS = ["blue", "red", "green", "black", "magenta"]
FEATURES = [
    "sst",
    "sst_anom",
    "sss",
    "sss_anom",
    "chl_log",
    "chl_log_anom",
    "mld_log",
    "xco2",
    "A",
    "B",
    "C",
    "T0",
    "T1",
]
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
SOCAT_WEIGHT_FILES = [
    "Finalweights_SOCAT16k_v2.txt",
    "Finalweights_SOCAT16k_v2_seed11.txt",
    "Finalweights_SOCAT16k_v2_seed12.txt",
    "Finalweights_SOCAT16k_v2_seed13.txt",
    "Finalweights_SOCAT16k_v2_seed14.txt",
]
SOCAT_IMB_FILES = [
    "Finalimbs_SOCAT16k_v2.txt",
    "Finalimbs_SOCAT16k_v2_seed11.txt",
    "Finalimbs_SOCAT16k_v2_seed12.txt",
    "Finalimbs_SOCAT16k_v2_seed13.txt",
    "Finalimbs_SOCAT16k_v2_seed14.txt",
]


def loadtxt(base_dir: Path, filename: str) -> np.ndarray:
    return np.loadtxt(base_dir / filename)


def normalized_weights(base_dir: Path, filenames: list[str]) -> list[np.ndarray]:
    out = []
    for filename in filenames:
        weights = loadtxt(base_dir, filename).copy()
        for i in range(1, 12):
            norm = np.linalg.norm(weights[i - 1])
            if norm != 0:
                weights[i - 1] = weights[i - 1] / norm
        out.append(weights)
    return out


def plot_gobm_weights(intermediate_dir: Path, output_dir: Path) -> None:
    plt.figure(figsize=(12, 24))
    plt.subplots_adjust(bottom=0.5, hspace=0.8)
    avgcorr = np.zeros([len(MODELS), len(MODELS)])
    k = 1

    for i in range(1, 9, 2):
        df = pd.DataFrame()
        plt.subplot(4, 2, k)
        for j, model in enumerate(MODELS):
            weights = loadtxt(intermediate_dir, f"Finalweights_sampled16k_{model}.txt").copy()
            norm = np.linalg.norm(weights[i - 1])
            if norm != 0:
                weights[i - 1] = weights[i - 1] / norm
            df[model] = weights[i - 1]
            plt.annotate("n = " + str(len(FEATURES) - i + 1), xy=(10.7, 0.6), fontsize=12)
            plt.bar(
                np.arange(13) - 0.3 + j * 0.15,
                weights[i - 1],
                width=0.1,
                alpha=0.4,
                label=model,
                color=COLORS[j],
            )
            plt.xticks(np.arange(13), LABELS, rotation=45, ha="right", fontsize=11)
        if i == 1:
            plt.legend(bbox_to_anchor=(0.15, 0.55), ncol=2, fontsize="medium", frameon=False)

        plt.subplot(4, 2, k + 1)
        corr_np = df.to_numpy()
        correlations = np.ones([corr_np.shape[-1], corr_np.shape[-1]])
        for col_1 in range(corr_np.shape[-1]):
            for col_2 in range(corr_np.shape[-1]):
                keep = ~((corr_np[:, col_1] == 0.0) & (corr_np[:, col_2] == 0.0))
                correlations[col_1, col_2] = np.corrcoef(
                    corr_np[keep, col_1], corr_np[keep, col_2]
                )[-1, 0]
        avgcorr += correlations
        sns.heatmap(
            pd.DataFrame(correlations, columns=MODELS, index=MODELS),
            vmin=0.3,
            vmax=1,
            annot=True,
            cmap="GnBu",
        )
        k += 2

    plt.subplots_adjust(hspace=0.42)
    plt.savefig(output_dir / "GOBMs_delta_1Dxco2.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def socat_imbs(intermediate_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    imbs = [loadtxt(intermediate_dir, filename) for filename in SOCAT_IMB_FILES]
    arr = np.array([np.nan_to_num(imb, nan=1.0) for imb in imbs])
    return arr.mean(axis=0), arr.std(axis=0)


def plot_mean_dii_socat(intermediate_dir: Path, output_dir: Path) -> None:
    mean, stdev = socat_imbs(intermediate_dir)
    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))

    plt.figure(figsize=(6, 6))
    plt.plot(
        np.arange(len(mean), 0, -1),
        mean,
        "-",
        lw=1.5,
        label="mean of 5 random seeds",
        color=oceancolors[2],
    )
    plt.errorbar(
        np.arange(len(mean), 0, -1),
        mean,
        yerr=stdev,
        fmt="none",
        color="gray",
        capsize=3,
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=13)
    plt.xlim(2, 13)
    plt.xlabel("# of features", fontsize=14)
    plt.ylabel("DII", fontsize=14)
    plt.savefig(output_dir / "Mean_DII_SOCAT_square.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def plot_mean_feature_weights_socat(intermediate_dir: Path, output_dir: Path) -> None:
    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))
    normweights = normalized_weights(intermediate_dir, SOCAT_WEIGHT_FILES)
    weights_arr = np.array(normweights)

    plt.figure(figsize=(10.5, 10.5))
    plt.subplot(211)
    i = 1
    plt.bar(
        np.arange(13),
        weights_arr.mean(axis=0)[i - 1],
        width=0.3,
        alpha=0.7,
        color=oceancolors[2],
        label="Mean feature weights, 5 random seeds, n = 13",
    )
    plt.errorbar(
        np.arange(13),
        weights_arr.mean(axis=0)[i - 1],
        yerr=weights_arr.std(axis=0)[i - 1],
        fmt="none",
        lw=2,
        color="gray",
        capsize=3,
    )
    plt.ylim(0, 0.7)
    plt.text(1, 0.64, "Mean feature weights, 5 random seeds, n = 13", fontsize=16)
    plt.tick_params(labelbottom=False)

    plt.subplot(212)
    plt.subplots_adjust(hspace=0.05)
    i = 8
    plt.bar(
        np.arange(13),
        weights_arr.mean(axis=0)[i - 1],
        width=0.3,
        alpha=0.7,
        color=oceancolors[2],
        label="Mean feature weights, 5 random seeds, n = 6",
    )
    plt.xticks(np.arange(13), LABELS, rotation=45, ha="center", fontsize=18)
    plt.errorbar(
        np.arange(13),
        weights_arr.mean(axis=0)[i - 1],
        yerr=weights_arr.std(axis=0)[i - 1],
        fmt="none",
        lw=2,
        color="gray",
        capsize=3,
    )
    plt.ylim(0, 0.7)
    plt.text(1, 0.64, "Mean feature weights, 5 random seeds, n = 6", fontsize=16)
    plt.savefig(
        output_dir / "Mean_feature_weights_SOCAT_noxlabels.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()


def plot_dii_across_time(intermediate_dir: Path, output_dir: Path) -> None:
    mean, stdev = socat_imbs(intermediate_dir)
    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))
    means = {
        "2018-2019": ("k", loadtxt(intermediate_dir, "Finalimbs_SOCAT_2018_2019_16k_seed13.txt")),
        "2011-2012": ("g", loadtxt(intermediate_dir, "Finalimbs_SOCAT2010s_16k_seed13.txt")),
        "2001-2004": ("orange", loadtxt(intermediate_dir, "Finalimbs_SOCAT2000s_16k_seed13.txt")),
        "1990-1996": ("magenta", loadtxt(intermediate_dir, "Finalimbs_SOCAT90s_16k_seed13.txt")),
    }

    plt.figure(figsize=(6, 6))
    x = np.arange(len(mean), 0, -1)
    plt.plot(x, mean, "-", markersize=3, lw=1.5, label="2020-2022", color=oceancolors[2])
    plt.scatter(x, mean, edgecolors="k", color=oceancolors[2], s=50)
    plt.errorbar(x, mean, yerr=stdev, fmt="none", color="gray", capsize=3)

    for label, (color, imbs) in means.items():
        y = np.array([np.nan_to_num(imbs, nan=1.0)]).mean(axis=0)
        x = np.arange(len(y), 0, -1)
        plt.plot(x, y, "-", markersize=4, lw=1.5, label=label, color=color)
        plt.scatter(x, y, edgecolors="k", color=color, s=50)

    plt.legend(fontsize=15, frameon=False)
    plt.xlim(2, 13)
    plt.xlabel("# of features", fontsize=16)
    plt.ylabel("DII", fontsize=16)
    plt.savefig(output_dir / "DII_across_time.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def plot_feature_weights_across_time(intermediate_dir: Path, output_dir: Path) -> None:
    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))
    normweights16 = np.array(normalized_weights(intermediate_dir, SOCAT_WEIGHT_FILES))
    weights_by_period = {
        "2018-2019": ("k", normalized_weights(intermediate_dir, ["Finalweights_SOCAT_2018_2019_16k_seed13.txt"])),
        "2011-2012": ("green", normalized_weights(intermediate_dir, ["Finalweights_SOCAT2010s_16k_seed13.txt"])),
        "2001-2004": ("orange", normalized_weights(intermediate_dir, ["Finalweights_SOCAT2000s_16k_seed13.txt"])),
        "1990-1996": ("magenta", normalized_weights(intermediate_dir, ["Finalweights_SOCAT90s_16k_seed13.txt"])),
    }

    plt.figure(figsize=(10.5, 10.5))
    plt.subplot(211)
    plt.annotate("Feature weights, n = 13", xy=(0, 0.62), fontsize=18)
    i = 1
    plt.bar(np.arange(13) + 0.3, normweights16.mean(axis=0)[i - 1], width=0.1, alpha=0.7, color=oceancolors[2], label="2020-2022")
    plt.errorbar(np.arange(13) + 0.3, normweights16.mean(axis=0)[i - 1], yerr=normweights16.std(axis=0)[i - 1], fmt="none", lw=2, color="gray", capsize=3)
    for offset, (label, (color, weights)) in zip([0.15, 0, -0.15, -0.3], weights_by_period.items()):
        plt.bar(np.arange(13) + offset, np.array(weights).mean(axis=0)[i - 1], width=0.1, alpha=0.7, color=color, label=label)
    plt.tick_params(labelbottom=False)
    plt.legend(fontsize=14, frameon=False, bbox_to_anchor=(0.77, 0.57))
    plt.ylim(0, 0.7)

    plt.subplot(212)
    plt.annotate("Feature weights, n = 6", xy=(0, 0.62), fontsize=18)
    i = 8
    plt.bar(np.arange(13) + 0.3, normweights16.mean(axis=0)[i - 1], width=0.1, alpha=0.7, color=oceancolors[2])
    plt.xticks(np.arange(13), LABELS, rotation=45, ha="right", fontsize=18)
    plt.errorbar(np.arange(13) + 0.3, normweights16.mean(axis=0)[i - 1], yerr=normweights16.std(axis=0)[i - 1], fmt="none", lw=2, color="gray", capsize=3)
    for offset, (_, (color, weights)) in zip([0.15, 0, -0.15, -0.3], weights_by_period.items()):
        plt.bar(np.arange(13) + offset, np.array(weights).mean(axis=0)[i - 1], width=0.1, alpha=0.7, color=color)
    plt.ylim(0, 0.7)
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(output_dir / "Mean_featureweights_acrosstime_noxlabels.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot listed figures from Comparisons.ipynb.")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediates"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_gobm_weights(args.intermediate_dir, args.output_dir)
    plot_mean_dii_socat(args.intermediate_dir, args.output_dir)
    plot_mean_feature_weights_socat(args.intermediate_dir, args.output_dir)
    plot_dii_across_time(args.intermediate_dir, args.output_dir)
    plot_feature_weights_across_time(args.intermediate_dir, args.output_dir)


if __name__ == "__main__":
    main()
