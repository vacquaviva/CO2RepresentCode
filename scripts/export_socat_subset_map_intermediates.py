#!/usr/bin/env python3
"""Export cached map inputs used by the SOCAT subset plotting script.

This reproduces the non-DII data-prep parts of:
- Comparison_SOCAT_uniform.ipynb
- Comparisons_SOCAT_vs_NorthAtlantic.ipynb
- Comparisons_SOCAT_vs_SouthernOcean.ipynb

It writes small CSVs to intermediates/ so plotting does not need to reopen the
large zarr datasets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


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
SPATIAL = ["A", "B", "C"]
TARGET = "delta_fco2_1D"


def add_lat_lon(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    x = out["A"].to_numpy()
    y = out["B"].to_numpy()
    z = out["C"].to_numpy()
    radius = np.sqrt(x * x + y * y + z * z)
    x, y, z = x / radius, y / radius, z / radius
    out["lat_deg"] = np.degrees(np.arcsin(x))
    out["lon_deg"] = (np.degrees(-np.arctan2(y, z))) % 360 - 180
    return out


def pairwise_sqeuclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x**2).sum(axis=1)[:, None] + (y**2).sum(axis=1)[None, :] - 2 * x @ y.T


def unbalanced_sinkhorn(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    epsilon: float = 0.15,
    tau_a: float = 0.5,
    tau_b: float = 0.5,
    max_iter: int = 250,
) -> np.ndarray:
    kernel = np.exp(-cost / epsilon)
    kernel = np.maximum(kernel, 1e-300)
    u = np.ones_like(a)
    v = np.ones_like(b)
    alpha = tau_a / (tau_a + epsilon)
    beta = tau_b / (tau_b + epsilon)
    for _ in range(max_iter):
        u = (a / (kernel @ v + 1e-300)) ** alpha
        v = (b / (kernel.T @ u + 1e-300)) ** beta
    return (u[:, None] * kernel) * v[None, :]


def add_uniform_weights(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    from sklearn.neighbors import KNeighborsRegressor

    x_source = source[SPATIAL].to_numpy()
    x_target = target[SPATIAL].to_numpy()
    rng = np.random.default_rng(0)
    ns = min(len(x_source), 2000)
    nt = min(len(x_target), 2000)
    idx_source = rng.choice(len(x_source), size=ns, replace=False)
    idx_target = rng.choice(len(x_target), size=nt, replace=False)

    cost = pairwise_sqeuclidean(x_source[idx_source], x_target[idx_target])
    a = np.ones(ns) / ns
    b = np.ones(nt) / nt
    transport = unbalanced_sinkhorn(a, b, cost)
    weights_sub = transport.sum(axis=1) / a

    knn = KNeighborsRegressor(n_neighbors=15, weights="distance")
    knn.fit(x_source[idx_source], weights_sub)
    weights_uot = knn.predict(x_source)
    weights_uot = np.clip(weights_uot, 0, np.quantile(weights_uot, 0.99))
    weights_uot = weights_uot / weights_uot.mean()

    q = np.linspace(0, 1, 17)
    edges = [np.quantile(x_target[:, j], q) for j in range(x_target.shape[1])]

    def digitize_cols(values: np.ndarray) -> list[np.ndarray]:
        indexes = []
        for j, edge in enumerate(edges):
            bins = np.digitize(values[:, j], edge[1:-1], right=False)
            indexes.append(np.clip(bins, 0, len(edge) - 2))
        return indexes

    source_bins = digitize_cols(x_source)
    target_bins = digitize_cols(x_target)

    def hist1d(indexes: np.ndarray) -> np.ndarray:
        hist = np.bincount(indexes, minlength=16).astype(float)
        return hist / hist.sum()

    target_hists = [hist1d(indexes) for indexes in target_bins]
    weights_ipf = weights_uot.copy()
    for _ in range(8):
        for source_index, target_hist in zip(source_bins, target_hists):
            source_hist = np.bincount(source_index, weights=weights_ipf, minlength=16)
            source_hist = source_hist / source_hist.sum()
            scale = (target_hist + 1e-6) / (source_hist + 1e-6)
            weights_ipf *= np.clip(scale, 0.25, 4.0)[source_index]
        weights_ipf = np.clip(weights_ipf, 0, np.quantile(weights_ipf, 0.995))
        weights_ipf = weights_ipf / weights_ipf.mean()

    out = source.copy()
    out["uot_weight"] = weights_uot
    out["final_weight"] = weights_ipf
    return out


def load_reconstruction(path: str) -> xr.Dataset:
    ds = xr.open_dataset(
        path,
        engine="zarr",
    )
    ds = ds.sel(time=slice("1982-02-01", "2022-12-31"))
    ds[TARGET] = ds["fco2"] - ds["xco2_trend"]
    return ds[FEATURES + [TARGET]]


def load_socat_mask(path: str, reference_time: xr.DataArray) -> xr.Dataset:
    mask = xr.open_dataset(
        path,
        engine="zarr",
    )
    return mask.reindex(time=reference_time, method="nearest")


def shifted_region_mask(mask_file: Path, variable: str, region_value: int) -> xr.DataArray:
    mask = xr.open_dataset(mask_file)[variable]
    mask = mask.roll(lon=180, roll_coords="lon")
    mask["lon"] = np.arange(-179.5, 180.5, 1)
    mask = mask.rename({"lon": "xlon", "lat": "ylat"})
    return mask == region_value


def export_uniform_inputs(ds: xr.Dataset, socat_ds: xr.Dataset, mask_file: Path, output_dir: Path) -> None:
    from sklearn.preprocessing import StandardScaler

    socat = socat_ds.sel(time=slice("2020-01-01", "2022-12-31")).to_dataframe().dropna()
    scaler = StandardScaler()
    socat_scaled = pd.DataFrame(
        scaler.fit_transform(socat.loc[:, FEATURES + [TARGET]]),
        columns=FEATURES + [TARGET],
    )
    socat_1 = socat_scaled.sample(frac=0.5, random_state=67)
    socat_2 = socat_scaled.drop(socat_1.index)

    oceanmask = xr.open_dataset(mask_file).open_ocean
    oceanmask = oceanmask.roll(lon=180, roll_coords="lon")
    oceanmask["lon"] = np.arange(-179.5, 180.5, 1)
    oceanmask = oceanmask.rename({"lon": "xlon", "lat": "ylat"})
    dfocean = ds.where(oceanmask != 0).sel(time=slice("2020-01-01", "2022-12-31"))
    uniform = dfocean.to_dataframe().sample(n=22000, random_state=10)
    uniform_scaled = pd.DataFrame(
        scaler.transform(uniform.loc[:, FEATURES + [TARGET]]),
        columns=FEATURES + [TARGET],
    )[SPATIAL].dropna()

    socat_2_spatial = socat_2[SPATIAL]
    weighted = add_uniform_weights(socat_2_spatial, uniform_scaled)
    weighted.to_csv(output_dir / "SOCAT2_spatial_with_OTweights.csv", index=False)
    uniform_scaled.to_csv(output_dir / "Uniform_split_spatial.csv", index=False)

    weights = weighted["final_weight"].to_numpy()
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(weighted), size=len(weighted), replace=True, p=probabilities)

    idx_subset = [FEATURES.index(col) for col in SPATIAL]
    socat_2_inv = scaler.inverse_transform(socat_2.iloc[idx, :])
    socat_2_points = pd.DataFrame(socat_2_inv[:, idx_subset], columns=SPATIAL)
    add_lat_lon(socat_2_points).to_csv(output_dir / "SOCAT2_resampled_points.csv", index=False)

    socat_1_inv = scaler.inverse_transform(socat_scaled.iloc[socat_1.index, :])
    socat_1_points = pd.DataFrame(socat_1_inv[:, idx_subset], columns=SPATIAL)
    add_lat_lon(socat_1_points).to_csv(output_dir / "SOCAT1_points.csv", index=False)


def export_region_inputs(
    ds: xr.Dataset,
    socat_ds: xr.Dataset,
    mask_file: Path,
    output_dir: Path,
    variable: str,
    region_value: int,
    left_filename: str,
    right_filename: str,
) -> None:
    region_mask = shifted_region_mask(mask_file, variable, region_value)
    region_points = ds.where(region_mask).sel(time=slice("2020-01-01", "2022-12-31")).to_dataframe().dropna()
    socat_minus_region = socat_ds.sel(time=slice("2020-01-01", "2022-12-31")).where(~region_mask)
    socat_minus_region = socat_minus_region.to_dataframe().dropna().sample(n=16000, random_state=10)
    add_lat_lon(socat_minus_region[SPATIAL]).to_csv(output_dir / left_filename, index=False)
    add_lat_lon(region_points[SPATIAL]).to_csv(output_dir / right_filename, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cached map point intermediates.")
    parser.add_argument("--output-dir", type=Path, default=Path("intermediates"))
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
        "--region-mask",
        type=Path,
        default=Path("../2023vs2024/RECCAP2_region_masks_all_v20221025.nc"),
        help="Path to RECCAP2_region_masks_all_v20221025.nc.",
    )
    parser.add_argument("--only", choices=["all", "uniform", "na", "so"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_reconstruction(args.reconstruction_zarr)
    socat_mask = load_socat_mask(args.socat_mask_zarr, ds.time)
    socat_ds = ds.where(socat_mask.socat_mask == 1)

    if args.only in ("all", "uniform"):
        export_uniform_inputs(ds, socat_ds, args.region_mask, args.output_dir)
    if args.only in ("all", "na"):
        export_region_inputs(
            ds,
            socat_ds,
            args.region_mask,
            args.output_dir,
            variable="atlantic",
            region_value=3,
            left_filename="SM_minus_NA_sample_points.csv",
            right_filename="NA_STPS_points.csv",
        )
    if args.only in ("all", "so"):
        export_region_inputs(
            ds,
            socat_ds,
            args.region_mask,
            args.output_dir,
            variable="southern",
            region_value=2,
            left_filename="SM_minus_SO_sample_points.csv",
            right_filename="SO_SPSS_points.csv",
        )


if __name__ == "__main__":
    main()
