#!/usr/bin/env python3
"""Generate SOCAT DII/weight text intermediates from raw SOCAT data.

This optional slow script generalizes the code in SOCATdata.ipynb and
SOCATdata20k.ipynb. It can recreate the SOCAT files used by
plot_socat_id_and_summary.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dadapy.feature_weighting import FeatureWeighting
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "sst",
    "sst_anomaly",
    "sss",
    "sss_anomaly",
    "chl_log",
    "chl_log_anomaly",
    "mld_log",
    "xco2_trend",
    "A",
    "B",
    "C",
    "T0",
    "T1",
]
TARGET = ["delta_fco2_1D"]


def output_stem(n_sample: int, seed: int) -> str:
    if n_sample == 16000 and seed == 10:
        return "SOCAT16k_v2"
    if n_sample == 16000:
        return f"SOCAT16k_v2_seed{seed}"
    return f"SOCAT{n_sample // 1000}k_v2_seed{seed}"


def load_socat_dataframe() -> pd.DataFrame:
    socat_mask = xr.open_dataset(
        "gs://leap-persistent/abbysh/zarr_files_/socat_mask_feb1982-dec2022.zarr",
        engine="zarr",
    )
    ds = xr.open_dataset(
        "gs://leap-persistent/fayamanda/reconstructions/"
        "pCO2_LEAP_fco2-residual-full-dataset-preML_198201-202412.zarr",
        engine="zarr",
    )
    ds = ds.sel(time=slice("1982-02-01", "2022-12-31"))
    aligned_mask = socat_mask.reindex(time=ds.time, method="nearest")
    ds = ds.where(aligned_mask.socat_mask == 1)
    ds["delta_fco2_1D"] = ds["fco2"] - ds["xco2_trend"]
    ds = ds[FEATURES + TARGET]
    ds = ds.sel(time=slice("2020-01-01", "2022-12-31"))
    return ds.to_dataframe().dropna()


def generate_one(socat: pd.DataFrame, output_dir: Path, n_sample: int, seed: int) -> None:
    print(f"Generating SOCAT weights/DII for n={n_sample}, seed={seed}")
    sample = socat.sample(n=n_sample, random_state=seed)
    scaled = StandardScaler().fit_transform(sample.loc[:, FEATURES + TARGET])
    scaled = pd.DataFrame(scaled, columns=FEATURES + TARGET)

    target = FeatureWeighting(scaled[TARGET].to_numpy(), verbose=True)
    inputs = FeatureWeighting(scaled[FEATURES].to_numpy(), verbose=True)
    final_imbs, final_weights = inputs.return_backward_greedy_dii_elimination(
        target_data=target,
        initial_weights=None,
        n_epochs=80,
        learning_rate=None,
        decaying_lr="cos",
    )

    stem = output_stem(n_sample, seed)
    np.savetxt(output_dir / f"Finalweights_{stem}.txt", final_weights)
    np.savetxt(output_dir / f"Finalimbs_{stem}.txt", final_imbs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SOCAT weight/DII text files.")
    parser.add_argument("--output-dir", type=Path, default=Path("intermediates"))
    parser.add_argument(
        "--jobs",
        default="16000:10,16000:11,16000:12,16000:13,16000:14,20000:13",
        help="Comma-separated n:seed jobs.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    socat = load_socat_dataframe()
    for job in args.jobs.split(","):
        n_sample, seed = [int(part) for part in job.split(":")]
        generate_one(socat, args.output_dir, n_sample=n_sample, seed=seed)


if __name__ == "__main__":
    main()
