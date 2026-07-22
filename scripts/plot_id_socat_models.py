#!/usr/bin/env python3
"""Reproduce ID_SOCAT_models_v2.png from saved intermediate CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = ["ETHZ", "FESOM", "NorESM", "MRI", "IPSL"]
COLORS = ["blue", "red", "green", "black", "magenta"]


def read_csv(base_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(base_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot intrinsic dimensions for SOCAT observations and GOBMs in the SOCAT domain."
    )
    parser.add_argument(
        "--intermediate-dir",
        type=Path,
        default=Path("intermediates"),
        help="Directory containing the saved intrinsic-dimension CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory where the figure will be written.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dsocat_dim_all = read_csv(args.intermediate_dir, "dSOCAT_dim_all.csv")
    dsocat_dim_input = read_csv(args.intermediate_dir, "dSOCAT_dim_input.csv")
    dsocat_scales_all = read_csv(args.intermediate_dir, "dSOCAT_scales_all.csv")
    dsocat_scales_input = read_csv(args.intermediate_dir, "dSOCAT_scales_input.csv")

    socat_dim_all = read_csv(args.intermediate_dir, "SOCATobs_dim_all.csv")
    socat_dim_input = read_csv(args.intermediate_dir, "SOCATobs_dim_input.csv")
    socat_scales_all = read_csv(args.intermediate_dir, "SOCATobs_scales_all.csv")
    socat_scales_input = read_csv(args.intermediate_dir, "SOCATobs_scales_input.csv")

    ids_gride_data = socat_dim_all["SOCAT"]
    ids_gride_in_data = socat_dim_input["SOCAT"]
    scales_gride_data = socat_scales_all["SOCAT"]
    scales_gride_in_data = socat_scales_input["SOCAT"]

    plt.figure(figsize=[14, 3])
    oceancolors = plt.cm.cool(np.linspace(0, 0.3, 5))

    for i, model in enumerate(MODELS):
        ids_gride_s = dsocat_dim_all[model]
        ids_gride_in_s = dsocat_dim_input[model]
        scales_gride_s = dsocat_scales_all[model]
        scales_gride_in_s = dsocat_scales_input[model]

        plt.subplot(1, 5, i + 1)

        plt.plot(
            scales_gride_s,
            ids_gride_s,
            "-",
            alpha=0.85,
            color=COLORS[i],
            label=r"S mask (features+$\delta fCO_2$)",
        )
        plt.scatter(scales_gride_s, ids_gride_s, edgecolors="k", color=COLORS[i], s=50)

        plt.plot(
            scales_gride_in_s,
            ids_gride_in_s,
            "--",
            alpha=0.85,
            color=COLORS[i],
            label="S mask (features)",
        )
        plt.scatter(
            scales_gride_in_s,
            ids_gride_in_s,
            edgecolors="k",
            color=COLORS[i],
            s=50,
        )

        plt.plot(
            scales_gride_data,
            ids_gride_data,
            alpha=0.85,
            color=oceancolors[2],
            label=r"S data (features+$\delta fCO_2$)",
        )
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
            color=oceancolors[3],
        )
        plt.scatter(
            scales_gride_in_data,
            ids_gride_in_data,
            edgecolors="k",
            color=oceancolors[2],
            s=50,
        )

        plt.xlabel(r"Scale", size=14)
        if i == 0:
            plt.ylabel("Estimated ID", size=14)
            plt.annotate("SOCAT", (2, 2.3), color=oceancolors[2], fontsize=14)
        plt.annotate(model, (0.5, 5), color=COLORS[i], fontsize=14)

        plt.ylim(1.5, 6)
        plt.xlim(0.1, 3.3)
        plt.tight_layout()

    plt.suptitle(
        "Intrinsic dimension for SOCAT observations and GOBMs in SOCAT domain",
        x=0.5,
        ha="center",
        y=1.05,
        fontsize=14,
    )
    plt.savefig(
        args.output_dir / "ID_SOCAT_models_v2.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.2,
    )


if __name__ == "__main__":
    main()
