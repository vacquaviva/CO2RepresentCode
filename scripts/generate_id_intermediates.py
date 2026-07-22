#!/usr/bin/env python3
"""Generate intrinsic-dimension CSV intermediates used by the figure scripts.

This is the slow/raw-data path extracted from the notebooks. The plotting scripts
do not need to run this if the files in intermediates/ are already present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import xarray as xr


MODELS = ["ETHZ", "FESOM", "NorESM", "MRI", "IPSL"]
GOBM_FILES = [
    "CESM_ETHZ/MLinput_CESM_ETHZ_mon_1x1_197001_202212.pkl",
    "FESOM2_REcoM/MLinput_FESOM2_REcoM_mon_1x1_197001_202212.pkl",
    "NorESM/MLinput_NorESM_mon_1x1_197001_202212.pkl",
    "MRI_ESM2_2/MLinput_MRI_ESM2_2_mon_1x1_197001_202212.pkl",
    "IPSL/MLinput_IPSL_mon_1x1_197001_202212.pkl",
]
GOBM_FEATURES = [
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
SOCAT_FEATURES = [
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
TARGET = "delta_fco2_1D"
TIME_WINDOWS = {
    "SOCAT90s": ("1990-01-01", "1996-12-31"),
    "SOCAT2000s": ("2001-01-01", "2004-12-31"),
    "SOCAT2010s": ("2011-01-01", "2012-12-31"),
    "SOCAT201819": ("2018-01-01", "2019-12-31"),
    "SOCAT202022": ("2020-01-01", "2022-12-31"),
}


def compute_id_tables(sample: pd.DataFrame, include_input_only: bool) -> tuple:
    from dadapy import Data

    all_data = Data(sample.to_numpy(), verbose=False)
    all_data.compute_distances(maxk=min(sample.shape[0] - 1, 10000))
    ids_all, _, scales_all = all_data.return_id_scaling_gride(range_max=1024)

    if not include_input_only:
        return ids_all, None, scales_all, None

    input_sample = sample.drop([TARGET], axis=1)
    input_data = Data(input_sample.to_numpy(), verbose=False)
    input_data.compute_distances(maxk=min(input_sample.shape[0] - 1, 10000))
    ids_input, _, scales_input = input_data.return_id_scaling_gride(range_max=1024)
    return ids_all, ids_input, scales_all, scales_input


def load_gobm_sample(
    gcs_root: str,
    fileloc: str,
    n_sample: int,
    random_state: int,
    socat_mask_only: bool,
) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler

    df = pd.read_pickle(gcs_root.rstrip("/") + "/" + fileloc)
    df[TARGET] = df["sfco2"] - df["xco2"]
    df = df[GOBM_FEATURES + ["socat_mask", TARGET]]
    df = df[df.index.get_level_values("time") >= "2020-01-01"]

    scaled = StandardScaler().fit_transform(df.loc[:, GOBM_FEATURES + [TARGET]])
    scaled = pd.DataFrame(scaled, columns=GOBM_FEATURES + [TARGET])
    scaled["socat_mask"] = df["socat_mask"].values

    if socat_mask_only:
        scaled = scaled[scaled["socat_mask"] == 1]

    scaled = scaled.drop_duplicates(subset=[TARGET]).dropna()
    return scaled.sample(n=n_sample, random_state=random_state)


def generate_gobm_id_csvs(
    output_dir: Path,
    gcs_root: str,
    n_sample: int,
    random_state: int,
    prefix: str,
    socat_mask_only: bool,
) -> None:
    dim_all = pd.DataFrame()
    dim_input = pd.DataFrame()
    scales_all = pd.DataFrame()
    scales_input = pd.DataFrame()

    for model, fileloc in zip(MODELS, GOBM_FILES):
        print(f"Generating {prefix} for {model}")
        sample = load_gobm_sample(gcs_root, fileloc, n_sample, random_state, socat_mask_only)
        sample = sample.drop(["socat_mask"], axis=1)
        ids_all, ids_input, scale_all, scale_input = compute_id_tables(
            sample, include_input_only=True
        )
        dim_all[model] = ids_all
        dim_input[model] = ids_input
        scales_all[model] = scale_all
        scales_input[model] = scale_input

    dim_all.to_csv(output_dir / f"{prefix}_dim_all.csv")
    dim_input.to_csv(output_dir / f"{prefix}_dim_input.csv")
    scales_all.to_csv(output_dir / f"{prefix}_scales_all.csv")
    scales_input.to_csv(output_dir / f"{prefix}_scales_input.csv")


def load_socat_dataframe(reconstruction_zarr: str, socat_mask_zarr: str) -> pd.DataFrame:
    socat_mask = xr.open_dataset(
        socat_mask_zarr,
        engine="zarr",
    )
    ds = xr.open_dataset(
        reconstruction_zarr,
        engine="zarr",
    )
    ds = ds.sel(time=slice("1982-02-01", "2022-12-31"))
    aligned_mask = socat_mask.reindex(time=ds.time, method="nearest")
    ds = ds.where(aligned_mask.socat_mask == 1)
    ds[TARGET] = ds["fco2"] - ds["xco2_trend"]
    ds = ds[SOCAT_FEATURES + [TARGET]]
    ds = ds.sel(time=slice("2020-01-01", "2022-12-31"))
    return ds.to_dataframe().dropna()


def load_socat_dataset(reconstruction_zarr: str, socat_mask_zarr: str) -> xr.Dataset:
    socat_mask = xr.open_dataset(
        socat_mask_zarr,
        engine="zarr",
    )
    ds = xr.open_dataset(
        reconstruction_zarr,
        engine="zarr",
    )
    ds = ds.sel(time=slice("1982-02-01", "2022-12-31"))
    aligned_mask = socat_mask.reindex(time=ds.time, method="nearest")
    ds = ds.where(aligned_mask.socat_mask == 1)
    ds[TARGET] = ds["fco2"] - ds["xco2_trend"]
    return ds[SOCAT_FEATURES + [TARGET]]


def scale_socat(socat: pd.DataFrame, n_sample: int | None, random_state: int) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler

    if n_sample is not None:
        socat = socat.sample(n=n_sample, random_state=random_state)
    scaled = StandardScaler().fit_transform(socat.loc[:, SOCAT_FEATURES + [TARGET]])
    return pd.DataFrame(scaled, columns=SOCAT_FEATURES + [TARGET])


def generate_socat_id_csvs(
    output_dir: Path,
    random_state: int,
    reconstruction_zarr: str,
    socat_mask_zarr: str,
) -> None:
    socat = load_socat_dataframe(reconstruction_zarr, socat_mask_zarr)

    sample_16k = scale_socat(socat, n_sample=16000, random_state=random_state)
    ids_all, ids_input, scales_all, scales_input = compute_id_tables(
        sample_16k, include_input_only=True
    )
    pd.DataFrame({"SOCAT": ids_all}).to_csv(output_dir / "SOCATobs_dim_all.csv")
    pd.DataFrame({"SOCAT": ids_input}).to_csv(output_dir / "SOCATobs_dim_input.csv")
    pd.DataFrame({"SOCAT": scales_all}).to_csv(output_dir / "SOCATobs_scales_all.csv")
    pd.DataFrame({"SOCAT": scales_input}).to_csv(output_dir / "SOCATobs_scales_input.csv")

    full_sample = scale_socat(socat, n_sample=None, random_state=random_state)
    ids_full, _, scales_full, _ = compute_id_tables(full_sample, include_input_only=False)
    pd.DataFrame({"SOCAT": ids_full}).to_csv(output_dir / "SOCATobs_full_dim_all.csv")
    pd.DataFrame({"SOCAT": scales_full}).to_csv(output_dir / "SOCATobs_full_scales_all.csv")


def generate_socat_time_id_csvs(
    output_dir: Path,
    random_state: int,
    reconstruction_zarr: str,
    socat_mask_zarr: str,
) -> None:
    ds = load_socat_dataset(reconstruction_zarr, socat_mask_zarr)
    time_dim_all = pd.DataFrame()
    time_scales_all = pd.DataFrame()

    for tag, (start, stop) in TIME_WINDOWS.items():
        from sklearn.preprocessing import StandardScaler

        print(f"Generating across-time ID for {tag}: {start} to {stop}")
        frame = ds.sel(time=slice(start, stop)).to_dataframe().dropna()
        sample = frame.sample(n=16000, random_state=random_state)
        scaled = StandardScaler().fit_transform(sample.loc[:, SOCAT_FEATURES + [TARGET]])
        scaled = pd.DataFrame(scaled, columns=SOCAT_FEATURES + [TARGET])
        ids_all, _, scales_all, _ = compute_id_tables(scaled, include_input_only=False)
        time_dim_all[tag] = ids_all
        time_scales_all[tag] = scales_all

    time_dim_all.to_csv(output_dir / "time_dim_all.csv", index=False)
    time_scales_all.to_csv(output_dir / "time_scales_all.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate slow ID CSV intermediates.")
    parser.add_argument("--output-dir", type=Path, default=Path("intermediates"))
    parser.add_argument(
        "--gcs-root",
        default="gs://leap-persistent/vacquaviva/GOBMs",
        help="Root path containing the GOBM pickle inputs.",
    )
    parser.add_argument(
        "--reconstruction-zarr",
        default=(
            "gs://leap-persistent/fayamanda/reconstructions/"
            "pCO2_LEAP_fco2-residual-full-dataset-preML_198201-202412.zarr"
        ),
        help="Path or gs:// URL for the pCO2 LEAP reconstruction zarr store.",
    )
    parser.add_argument(
        "--socat-mask-zarr",
        default="gs://leap-persistent/abbysh/zarr_files_/socat_mask_feb1982-dec2022.zarr",
        help="Path or gs:// URL for the SOCAT mask zarr store.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "gobm-16k", "gobm-32k", "gobm-socat-mask", "socat", "socat-time"],
        default="all",
        help="Subset of intermediates to generate.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "gobm-16k"):
        generate_gobm_id_csvs(
            args.output_dir,
            args.gcs_root,
            n_sample=16000,
            random_state=10,
            prefix="df_01",
            socat_mask_only=False,
        )
        for suffix in ["dim_all", "dim_input", "scales_all", "scales_input"]:
            (args.output_dir / f"df_01_{suffix}.csv").rename(
                args.output_dir / f"df_01_{suffix}_seed10.csv"
            )

    if args.only in ("all", "gobm-32k"):
        generate_gobm_id_csvs(
            args.output_dir,
            args.gcs_root,
            n_sample=32000,
            random_state=10,
            prefix="df_02",
            socat_mask_only=False,
        )
        for suffix in ["dim_all", "dim_input", "scales_all", "scales_input"]:
            (args.output_dir / f"df_02_{suffix}.csv").rename(
                args.output_dir / f"df_02_{suffix}_seed10.csv"
            )

    if args.only in ("all", "gobm-socat-mask"):
        generate_gobm_id_csvs(
            args.output_dir,
            args.gcs_root,
            n_sample=16000,
            random_state=10,
            prefix="dSOCAT",
            socat_mask_only=True,
        )

    if args.only in ("all", "socat"):
        generate_socat_id_csvs(
            args.output_dir,
            random_state=10,
            reconstruction_zarr=args.reconstruction_zarr,
            socat_mask_zarr=args.socat_mask_zarr,
        )

    if args.only in ("all", "socat-time"):
        generate_socat_time_id_csvs(
            args.output_dir,
            random_state=13,
            reconstruction_zarr=args.reconstruction_zarr,
            socat_mask_zarr=args.socat_mask_zarr,
        )


if __name__ == "__main__":
    main()
