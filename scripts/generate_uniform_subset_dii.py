#!/usr/bin/env python3
"""Generate DII intermediates for the SOCAT uniform-subset figure.

This extracts the slow DADAPy portion of Comparison_SOCAT_uniform.ipynb. It
creates:

- Finalweights_SOCAT1.txt
- Finalimbs_SOCAT1.txt
- standardimbs_SOCAT1_my_pipeline.txt
- imbs_weightedSOCAT2_fromSOCAT1.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
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


def ranks_from_distances(distances: np.ndarray) -> np.ndarray:
    """Return rank matrix with -1 on the diagonal and 1 for nearest neighbor."""
    distances = np.asarray(distances)
    n_points = distances.shape[0]
    ranks = np.full((n_points, n_points), -1, dtype=np.int32)

    for i in range(n_points):
        mask = np.ones(n_points, dtype=bool)
        mask[i] = False
        neighbors = np.arange(n_points)[mask]
        order = neighbors[np.argsort(distances[i, mask])]
        ranks[i, order] = np.arange(1, n_points, dtype=np.int32)

    return ranks


def weighted_info_imbalance_first_nn(
    ranks_a: np.ndarray,
    ranks_b: np.ndarray,
    weights: np.ndarray,
    first_rank: int = 1,
) -> float:
    """Compute weighted information imbalance using first neighbors in A."""
    ranks_a = np.asarray(ranks_a)
    ranks_b = np.asarray(ranks_b)
    weights = np.asarray(weights, dtype=float)

    n_points = ranks_a.shape[0]
    if ranks_a.shape != (n_points, n_points) or ranks_b.shape != (n_points, n_points):
        raise ValueError("Rank matrices must both be square with the same shape.")
    if weights.shape[0] != n_points:
        raise ValueError("Weights length must match rank matrix size.")

    cumulative_b = np.zeros((n_points, n_points), dtype=float)
    for i in range(n_points):
        neighbor_index = np.where(ranks_b[i] > 0)[0]
        ordered = neighbor_index[np.argsort(ranks_b[i, neighbor_index])]
        cumulative_b[i, ordered] = np.cumsum(weights[ordered])

    first_a = ranks_a == first_rank
    numerator = np.sum((weights[:, None] * cumulative_b)[first_a])
    return float(2.0 * numerator / np.sum(weights))


def load_socat_scaled(reconstruction_zarr: str, socat_mask_zarr: str) -> pd.DataFrame:
    socat_mask = xr.open_dataset(socat_mask_zarr, engine="zarr")
    ds = xr.open_dataset(reconstruction_zarr, engine="zarr")
    ds = ds.sel(time=slice("1982-02-01", "2022-12-31"))
    aligned_mask = socat_mask.reindex(time=ds.time, method="nearest")
    ds = ds.where(aligned_mask.socat_mask == 1)
    ds["delta_fco2_1D"] = ds["fco2"] - ds["xco2_trend"]
    ds = ds[FEATURES + TARGET]
    ds = ds.sel(time=slice("2020-01-01", "2022-12-31"))
    socat = ds.to_dataframe().dropna()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(socat.loc[:, FEATURES + TARGET])
    return pd.DataFrame(scaled, columns=FEATURES + TARGET)


def compute_rank_matrix(values: np.ndarray, verbose: bool) -> np.ndarray:
    from dadapy.feature_weighting import FeatureWeighting

    feature_weighting = FeatureWeighting(values, verbose=verbose)
    return ranks_from_distances(feature_weighting.full_distance_matrix)


def load_or_generate_socat1_weights(
    socat1_scaled: pd.DataFrame,
    output_dir: Path,
    n_epochs: int,
    verbose: bool,
    force: bool,
) -> np.ndarray:
    weights_path = output_dir / "Finalweights_SOCAT1.txt"
    imbs_path = output_dir / "Finalimbs_SOCAT1.txt"
    if weights_path.exists() and imbs_path.exists() and not force:
        print(f"Using existing {weights_path}")
        return np.loadtxt(weights_path)

    from dadapy.feature_weighting import FeatureWeighting

    print("Learning SOCAT1 feature weights with DADAPy.")
    target = FeatureWeighting(socat1_scaled[TARGET].to_numpy(), verbose=verbose)
    inputs = FeatureWeighting(socat1_scaled[FEATURES].to_numpy(), verbose=verbose)
    final_imbs, final_weights = inputs.return_backward_greedy_dii_elimination(
        target_data=target,
        initial_weights=None,
        n_epochs=n_epochs,
        learning_rate=None,
        decaying_lr="cos",
    )
    np.savetxt(weights_path, final_weights)
    np.savetxt(imbs_path, final_imbs)
    return final_weights


def compute_curve(
    scaled: pd.DataFrame,
    final_weights: np.ndarray,
    sample_weights: np.ndarray,
    label: str,
    verbose: bool,
) -> list[float]:
    print(f"Computing target ranks for {label}.")
    target_ranks = compute_rank_matrix(scaled[TARGET].to_numpy(), verbose=verbose)

    imbalances = []
    normalized_weights = sample_weights / sample_weights.sum()
    for k, weights in enumerate(final_weights):
        if np.any(np.isnan(weights)):
            print(f"{label}: NaNs encountered for index {k}")
            imbalances.append(np.nan)
            continue

        print(f"{label}: computing DII for {len(final_weights) - k} features")
        source_ranks = compute_rank_matrix(scaled[FEATURES].to_numpy() * weights, verbose=verbose)
        imbalance = weighted_info_imbalance_first_nn(
            source_ranks,
            target_ranks,
            normalized_weights,
            first_rank=1,
        )
        imbalances.append(imbalance)
        print(f"{label}: {imbalance}")

    return imbalances


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate uniform SOCAT subset DII intermediates.")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediates"))
    parser.add_argument(
        "--reconstruction-zarr",
        default=(
            "raw/pCO2_LEAP_fco2-residual-full-dataset-preML_198201-202412.zarr"
        ),
    )
    parser.add_argument(
        "--socat-mask-zarr",
        default="raw/socat_mask_feb1982-dec2022.zarr",
    )
    parser.add_argument("--n-epochs", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)

    weighted_spatial = pd.read_csv(args.intermediate_dir / "SOCAT2_spatial_with_OTweights.csv")
    sampling_weights = weighted_spatial["final_weight"].to_numpy()

    socat_scaled = load_socat_scaled(args.reconstruction_zarr, args.socat_mask_zarr)
    socat1_scaled = socat_scaled.sample(frac=0.5, random_state=67)
    socat2_scaled = socat_scaled.drop(socat1_scaled.index)

    if len(socat2_scaled) != len(sampling_weights):
        raise ValueError(
            "SOCAT2 rows do not match SOCAT2_spatial_with_OTweights.csv: "
            f"{len(socat2_scaled)} vs {len(sampling_weights)}"
        )

    final_weights = load_or_generate_socat1_weights(
        socat1_scaled,
        args.intermediate_dir,
        n_epochs=args.n_epochs,
        verbose=args.verbose,
        force=args.force,
    )

    socat2_curve_path = args.intermediate_dir / "imbs_weightedSOCAT2_fromSOCAT1.txt"
    if args.force or not socat2_curve_path.exists():
        socat2_curve = compute_curve(
            socat2_scaled,
            final_weights,
            sampling_weights,
            label="SOCAT2 weighted",
            verbose=args.verbose,
        )
        np.savetxt(socat2_curve_path, socat2_curve)
    else:
        print(f"Using existing {socat2_curve_path}")

    socat1_curve_path = args.intermediate_dir / "standardimbs_SOCAT1_my_pipeline.txt"
    if args.force or not socat1_curve_path.exists():
        socat1_curve = compute_curve(
            socat1_scaled,
            final_weights,
            np.ones(len(socat1_scaled)),
            label="SOCAT1 uniform",
            verbose=args.verbose,
        )
        np.savetxt(socat1_curve_path, socat1_curve)
    else:
        print(f"Using existing {socat1_curve_path}")


if __name__ == "__main__":
    main()
