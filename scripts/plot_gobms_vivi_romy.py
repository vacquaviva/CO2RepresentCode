#!/usr/bin/env python3
"""Reproduce listed figures from ComparisonsGOBMsViviRomy.ipynb."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ["ETHZ", "FESOM", "NorESM", "MRI", "IPSL"]
COLORS = ["blue", "red", "green", "black", "magenta"]
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
LABELS_SHORT = [
    r"$\bf{SST}$",
    r"$\bf{SST\_a}$",
    r"$\bf{SSS}$",
    r"$\bf{SSS\_a}$",
    r"$\bf{Chl}$",
    r"$\bf{Chl\_a}$",
    r"$\bf{MLD}$",
    r"$\bf xCO_2$",
    r"$\bf{A}$",
    r"$\bf{B}$",
    r"$\bf{C}$",
    r"$\bf{T_0}$",
    r"$\bf{T_1}$",
]
LABELS_LONG = [
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


def loadtxt(base_dir: Path, filename: str) -> np.ndarray:
    return np.loadtxt(base_dir / filename)


def load_romy(romy_dir: Path, filename: str) -> np.ndarray:
    return np.loadtxt(romy_dir / filename)


def normalize_rows(weights: np.ndarray) -> np.ndarray:
    weights = weights.copy()
    for i in range(1, 12):
        norm = np.linalg.norm(weights[i - 1])
        if norm != 0:
            weights[i - 1] = weights[i - 1] / norm
    return weights


def socat_weight_stats(intermediate_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    weights = [normalize_rows(loadtxt(intermediate_dir, filename)) for filename in SOCAT_WEIGHT_FILES]
    arr = np.array(weights)
    return arr.mean(axis=0), arr.std(axis=0)


def socat_imb_stats(intermediate_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    imbs = [loadtxt(intermediate_dir, filename) for filename in SOCAT_IMB_FILES]
    arr = np.array([np.nan_to_num(imb, nan=1.0) for imb in imbs])
    return arr.mean(axis=0), arr.std(axis=0)


def romy_weights_for_model(romy_dir: Path, model: str) -> list[np.ndarray]:
    return [
        normalize_rows(load_romy(romy_dir, f"Finalweights_sampled1_seed{seed}_{model}.txt"))
        for seed in range(10, 15)
    ]


def plot_weights_full_vs_socat(intermediate_dir: Path, romy_dir: Path, output_dir: Path) -> None:
    plt.figure(figsize=(20, 12))
    for subplot, feature_row, title in [
        (211, 1, "Feature weights, n = 13"),
        (212, 7, "Feature weights, n = 7"),
    ]:
        df = pd.DataFrame()
        df_sm = pd.DataFrame()
        plt.subplot(subplot)
        plt.annotate(title, xy=(6, 0.63), fontsize=18)
        for j, model in enumerate(MODELS):
            weights = normalize_rows(loadtxt(intermediate_dir, f"Finalweights_sampled16k_{model}.txt"))
            df[model] = weights[feature_row - 1]
            weights_sm = normalize_rows(load_romy(romy_dir, f"Finalweights_sampled1_seed10_{model}.txt"))
            df_sm[model] = weights_sm[feature_row - 1]
            label = model + ", full domain" if j == 0 else model
            plt.bar(np.arange(13) - 0.3 + j * 0.15, weights[feature_row - 1], width=0.1, alpha=0.4, color=COLORS[j], label=label)
            plt.bar(np.arange(13) - 0.3 + j * 0.15, weights_sm[feature_row - 1], width=0.1, fill=False, edgecolor=COLORS[j], label=(model + ", SOCAT domain") if j == 0 else None)
            if subplot == 211:
                plt.tick_params(labelbottom=False)
            else:
                plt.xticks(np.arange(13), LABELS_LONG, rotation=45, ha="center", fontsize=18)
            plt.text(
                1.5,
                0.63 - 0.06 * j,
                r"$\rho_{\mathrm{corr}}$ for "
                + model
                + ": "
                + str(np.round(np.corrcoef(df[model], df_sm[model])[0, 1], 2)),
                fontsize=15,
                ha="left",
                va="bottom",
            )
        if subplot == 211:
            plt.legend(fontsize=15, frameon=False)

    plt.subplots_adjust(hspace=0.05)
    plt.savefig(output_dir / "Weights_full_vs_SOCAT_n7_n13.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def plot_dii_with_weights_full_vs_socat(intermediate_dir: Path, output_dir: Path) -> None:
    plt.figure(figsize=(14, 2))
    for i, model in enumerate(MODELS):
        plt.subplot(1, 5, i + 1)
        imbs = loadtxt(intermediate_dir, f"Finalimbs_sampled16k_{model}.txt")
        imbs_sm = loadtxt(intermediate_dir, f"Method2GeneralizedImbs_{model}.txt")
        plt.plot(np.arange(len(imbs), 0, -1), np.nan_to_num(imbs, nan=1.0), "o-", markersize=3, lw=0.8, label="From full space", color=COLORS[i])
        plt.plot(np.arange(len(imbs_sm), 0, -1), np.nan_to_num(imbs_sm, nan=1.0), "x--", markersize=4, lw=0.8, label="From SOCAT domain", color=COLORS[i])
        plt.legend(frameon=False, fontsize=9.5)
        plt.text(12, 0.4, model, fontsize=12, ha="right", va="bottom")
        plt.xlim(2, 13)
        plt.xlabel("# of features")
        plt.ylabel("DII" if i == 0 else None, fontsize=12)
        if i > 0:
            plt.yticks([])
    plt.suptitle("Transferability of feature weights derived from SOCAT domain to the full space in GOBMs", x=0.5, ha="center", y=1.05, fontsize=12)
    plt.savefig(output_dir / "DII_with weights_full_vs_SOCAT_inline.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def plot_dii_models_vs_socat(intermediate_dir: Path, romy_dir: Path, output_dir: Path) -> None:
    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))
    mean_socat, stdev_socat = socat_imb_stats(intermediate_dir)

    plt.figure(figsize=(14, 2))
    for i, model in enumerate(MODELS):
        plt.subplot(1, 5, i + 1)
        plt.plot(np.arange(len(mean_socat), 0, -1), mean_socat, "o-", markersize=3, lw=1.5, label="SOCAT" if i == 0 else None, color=oceancolors[2])
        plt.scatter(6, mean_socat[13 - 6], s=36, color=oceancolors[2])
        plt.errorbar(np.arange(len(mean_socat), 0, -1), mean_socat, yerr=stdev_socat, fmt="none", color=oceancolors[2], capsize=3)

        model_imbs = [
            load_romy(romy_dir, f"Finalimbs_sampled1_seed{seed}_{model}.txt")
            for seed in range(10, 15)
        ]
        model_arr = np.array([np.nan_to_num(imbs, nan=1.0) for imbs in model_imbs])
        mean_model = model_arr.mean(axis=0)
        stdev_model = model_arr.std(axis=0)
        plt.plot(np.arange(len(mean_model), 0, -1), mean_model, "o-", markersize=3, lw=1.0, label=model, color=COLORS[i])
        plt.errorbar(np.arange(len(mean_model), 0, -1), mean_model, yerr=stdev_model, fmt="none", color=COLORS[i], capsize=3)
        plt.scatter(7, mean_model[13 - 7], s=36, color=COLORS[i])
        plt.xlim(2, 13)
        plt.xlabel("# of features")
        plt.ylabel("DII" if i == 0 else None, fontsize=12)
        if i > 0:
            plt.yticks([])
    plt.suptitle("Differentiable information imbalance for SOCAT observations and GOBMs in SOCAT domain", x=0.5, ha="center", y=1.05, fontsize=12)
    plt.savefig(output_dir / "DII_models_vs_SOCAT_inline.png", dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def model_socat_correlations(intermediate_dir: Path, romy_dir: Path, feature_row: int) -> np.ndarray:
    df = pd.DataFrame()
    for seed, filename in zip(range(10, 15), SOCAT_WEIGHT_FILES):
        weights = normalize_rows(loadtxt(intermediate_dir, filename))
        df[str(seed)] = weights[feature_row - 1]

    corr = np.zeros((5, 5))
    for seed in range(10, 15):
        for j, model in enumerate(MODELS):
            weights = normalize_rows(load_romy(romy_dir, f"Finalweights_sampled1_seed{seed}_{model}.txt"))
            corr[seed - 10, j] = np.round(np.corrcoef(df[str(seed)], weights[feature_row - 1])[0, 1], 2)
    return corr


def plot_weights_models_vs_socat(intermediate_dir: Path, romy_dir: Path, output_dir: Path, feature_row: int, output_name: str, title_d: int) -> None:
    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))
    socat_mean, _ = socat_weight_stats(intermediate_dir)
    corr = model_socat_correlations(intermediate_dir, romy_dir, feature_row)

    plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    plt.bar(np.arange(13), socat_mean[feature_row - 1], width=0.85, alpha=0.6, color=oceancolors[1], label="SOCAT observations")
    for j, model in enumerate(MODELS):
        weights = np.array(romy_weights_for_model(romy_dir, model))
        model_mean = weights.mean(axis=0)
        model_std = weights.std(axis=0)
        plt.title(f"Feature weights in GOBMs vs SOCAT observations, d = {title_d}, n = 16,000", fontsize=16)
        plt.bar(np.arange(13) - 0.3 + j * 0.15, model_mean[feature_row - 1], width=0.1, alpha=0.4, color=COLORS[j], label=model + ", SOCAT domain")
        plt.errorbar(np.arange(13) - 0.3 + j * 0.15, model_mean[feature_row - 1], yerr=model_std[feature_row - 1], fmt="none", lw=2, color="black", capsize=3)
        plt.xticks(np.arange(13), LABELS_SHORT, rotation=45, ha="right", fontsize=16)
        x_text = 1.5 if feature_row == 1 else 4
        plt.text(
            x_text,
            0.63 - 0.06 * j,
            r"$\rho_{\mathrm{corr}}$ for "
            + model
            + ": "
            + str(np.round(corr.mean(axis=0)[j], 2))
            + r" $\pm$ "
            + str(np.round(corr.std(axis=0)[j], 2)),
            fontsize=15,
            ha="left",
            va="bottom",
        )
    plt.ylim(0, 0.7)
    if feature_row == 7:
        plt.xlim(-1, 13.5)
    plt.legend(fontsize=14, loc="upper right" if feature_row == 7 else "best")
    plt.savefig(output_dir / output_name, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()


def plot_dii_from_models_in_socat(intermediate_dir: Path, output_dir: Path, seed: int) -> Path:
    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))
    plt.figure(figsize=(14, 2))
    socat_filename = (
        "Finalimbs_SOCAT16k_v2.txt"
        if seed == 10
        else f"Finalimbs_SOCAT16k_v2_seed{seed}.txt"
    )
    for i, model in enumerate(MODELS):
        plt.subplot(1, 5, i + 1)
        imbs_socat = loadtxt(intermediate_dir, socat_filename)
        imbs_from_m = loadtxt(intermediate_dir, f"Method2GeneralizedImbs_{model}_toSOCAT_seed{seed}.txt")
        plt.plot(np.arange(len(imbs_socat), 0, -1), np.nan_to_num(imbs_socat, nan=1.0), "o-", markersize=3, lw=1, label="From SOCAT" if i == 0 else None, color=oceancolors[2])
        plt.plot(np.arange(len(imbs_from_m), 0, -1), np.nan_to_num(imbs_from_m, nan=1.0), "o--", markersize=3, lw=1, label="From " + model, color=COLORS[i])
        plt.legend(frameon=False, fontsize=9)
        plt.xlim(2, 13)
        plt.xlabel("# of features")
        plt.ylabel("DII" if i == 0 else None)
        plt.text(5.5, 0.75, r"$\Delta$DII (n = 13): " + str(np.round(imbs_from_m[0] - imbs_socat[0], 2)), fontsize=9, va="top")
        plt.text(5.5, 0.65, r"$\Delta$DII (n = 7): " + str(np.round(imbs_from_m[6] - imbs_socat[6], 2)), fontsize=9, va="top")
    plt.suptitle("Transferability of feature weights derived from GOBMs onto SOCAT observations", x=0.5, ha="center", y=1.05, fontsize=12)
    output_path = output_dir / f"DII_from_models_in_SOCAT_seed{seed}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot listed figures from ComparisonsGOBMsViviRomy.ipynb.")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediates"))
    parser.add_argument("--romy-dir", type=Path, default=Path("intermediates/FromRomy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_weights_full_vs_socat(args.intermediate_dir, args.romy_dir, args.output_dir)
    plot_dii_with_weights_full_vs_socat(args.intermediate_dir, args.output_dir)
    plot_dii_models_vs_socat(args.intermediate_dir, args.romy_dir, args.output_dir)
    plot_weights_models_vs_socat(args.intermediate_dir, args.romy_dir, args.output_dir, 1, "Weights_n13_models_vs_SOCAT.png", 13)
    plot_weights_models_vs_socat(args.intermediate_dir, args.romy_dir, args.output_dir, 7, "Weights_n7_models_vs_SOCAT.png", 7)
    plot_dii_from_models_in_socat(args.intermediate_dir, args.output_dir, seed=10)
    plot_dii_from_models_in_socat(args.intermediate_dir, args.output_dir, seed=13)
    seed13_path = args.output_dir / "DII_from_models_in_SOCAT_seed13.png"
    shutil.copyfile(seed13_path, args.output_dir / "DII_from_models_in_SOCAT.png")


if __name__ == "__main__":
    main()
