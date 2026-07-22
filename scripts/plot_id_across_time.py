#!/usr/bin/env python3
"""Reproduce ID_across_time.png from saved across-time ID CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TAGS = ["SOCAT90s", "SOCAT2000s", "SOCAT2010s", "SOCAT201819", "SOCAT202022"]
LABELS = ["1990-1996", "2001-2004", "2011-2012", "2018-2019", "2020-2022"]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ID_across_time.png.")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediates"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    time_dim_all = pd.read_csv(require(args.intermediate_dir / "time_dim_all.csv"))
    time_scales_all = pd.read_csv(require(args.intermediate_dir / "time_scales_all.csv"))

    missing = [tag for tag in TAGS if tag not in time_dim_all or tag not in time_scales_all]
    if missing:
        raise ValueError(
            "time_dim_all.csv and time_scales_all.csv must contain these columns: "
            + ", ".join(missing)
        )

    plt.figure(figsize=(6, 6))
    colors = ["m", "orange", "g", "k", plt.cm.cool(np.linspace(0, 0.3, 5))[2]]
    for i, tag in reversed(list(enumerate(TAGS))):
        plt.plot(time_scales_all[tag], time_dim_all[tag], "-", alpha=0.85, color=colors[i], label=LABELS[i], lw=1.5)
        plt.scatter(time_scales_all[tag], time_dim_all[tag], edgecolors="k", color=colors[i], s=50)
    plt.legend(frameon=False, fontsize=15)
    plt.xlabel(r"Scale", size=16)
    plt.ylabel("Estimated ID", size=16)
    plt.savefig(args.output_dir / "ID_across_time.png", dpi=300, bbox_inches="tight", pad_inches=0.2)


if __name__ == "__main__":
    main()
