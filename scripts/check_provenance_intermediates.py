#!/usr/bin/env python3
"""Check presence of provenance intermediates grouped by generation route."""

from __future__ import annotations

import argparse
from pathlib import Path


MODELS = ["ETHZ", "FESOM", "IPSL", "MRI", "NorESM"]
SEEDS = [10, 11, 12, 13, 14]


REQUIRED_GROUPS = {
    "id_csvs": [
        "df_01_dim_all_seed10.csv",
        "df_01_scales_all_seed10.csv",
        "df_02_dim_all_seed10.csv",
        "df_02_scales_all_seed10.csv",
        "dSOCAT_dim_all.csv",
        "dSOCAT_dim_input.csv",
        "dSOCAT_scales_all.csv",
        "dSOCAT_scales_input.csv",
        "SOCATobs_dim_all.csv",
        "SOCATobs_dim_input.csv",
        "SOCATobs_scales_all.csv",
        "SOCATobs_scales_input.csv",
        "SOCATobs_full_dim_all.csv",
        "SOCATobs_full_scales_all.csv",
    ],
    "socat_weights_dii": [
        "Finalweights_SOCAT16k_v2.txt",
        "Finalweights_SOCAT16k_v2_seed11.txt",
        "Finalweights_SOCAT16k_v2_seed12.txt",
        "Finalweights_SOCAT16k_v2_seed13.txt",
        "Finalweights_SOCAT16k_v2_seed14.txt",
        "Finalweights_SOCAT20k_v2_seed13.txt",
        "Finalimbs_SOCAT16k_v2.txt",
        "Finalimbs_SOCAT16k_v2_seed11.txt",
        "Finalimbs_SOCAT16k_v2_seed12.txt",
        "Finalimbs_SOCAT16k_v2_seed13.txt",
        "Finalimbs_SOCAT16k_v2_seed14.txt",
        "Finalimbs_SOCAT20k_v2_seed13.txt",
    ],
    "across_time": [
        "Finalweights_SOCAT90s_16k_seed13.txt",
        "Finalweights_SOCAT2000s_16k_seed13.txt",
        "Finalweights_SOCAT2010s_16k_seed13.txt",
        "Finalweights_SOCAT_2018_2019_16k_seed13.txt",
        "Finalimbs_SOCAT90s_16k_seed13.txt",
        "Finalimbs_SOCAT2000s_16k_seed13.txt",
        "Finalimbs_SOCAT2010s_16k_seed13.txt",
        "Finalimbs_SOCAT_2018_2019_16k_seed13.txt",
        "time_dim_all.csv",
        "time_scales_all.csv",
    ],
    "subset_points": [
        "SOCAT2_spatial_with_OTweights.csv",
        "Uniform_split_spatial.csv",
        "SOCAT1_points.csv",
        "SOCAT2_resampled_points.csv",
        "SM_minus_NA_sample_points.csv",
        "NA_STPS_points.csv",
        "SM_minus_SO_sample_points.csv",
        "SO_SPSS_points.csv",
    ],
    "subset_dii": [
        "Finalweights_SOCAT1.txt",
        "Finalimbs_SOCAT1.txt",
        "standardimbs_SOCAT1_my_pipeline.txt",
        "imbs_weightedSOCAT2_fromSOCAT1.txt",
        "Finalweights_SOCAT_minus_NA.txt",
        "Finalimbs_SOCAT_minus_NA.txt",
        "StandardII_imbs_SOCAT_minus_NA.txt",
        "StandardII_imbs_NA_STPS.txt",
        "Finalweights_SOCAT_minus_SO.txt",
        "Finalimbs_SOCAT_minus_SO.txt",
        "StandardII_imbs_SOCAT_minus_SO.txt",
        "StandardII_imbs_SO_SPSS.txt",
    ],
}

REQUIRED_GROUPS["gobm_model_weights_dii"] = [
    *[f"Finalweights_sampled16k_{model}.txt" for model in MODELS],
    *[f"Finalweights_sampled20k_{model}.txt" for model in MODELS],
    *[f"Finalimbs_sampled16k_{model}.txt" for model in MODELS],
    "Finalimbs_sampled20k_FESOM.txt",
    "Finalimbs_sampled20k_NorESM.txt",
]

REQUIRED_GROUPS["gobm_stratified_sampling_domains"] = [
    "south_dim_all.csv",
    "south_scales_all.csv",
    "north_dim_all.csv",
    "north_scales_all.csv",
    *[f"Finalweights_sampled16k_South_{model}.txt" for model in MODELS],
    *[f"Finalweights_sampled16k_North_{model}.txt" for model in MODELS],
    *[f"Finalimbs_sampled16k_South_{model}.txt" for model in MODELS],
    *[f"Finalimbs_sampled16k_North_{model}.txt" for model in MODELS],
]

REQUIRED_GROUPS["method2_generalized_imbs"] = [
    *[f"Method2GeneralizedImbs_{model}.txt" for model in MODELS],
    *[f"Method2GeneralizedImbs_{model}_toSOCAT_seed10.txt" for model in MODELS],
    *[f"Method2GeneralizedImbs_{model}_toSOCAT_seed13.txt" for model in MODELS],
]

REQUIRED_GROUPS["knn_reconstructed_scores"] = [
    "knn_weighted_scores_seed13_metrics.csv",
    "kNN_weights_models_seed13_metrics.csv",
]

REQUIRED_GROUPS["from_romy_core"] = [
    *[
        f"FromRomy/Finalweights_sampled1_seed{seed}_{model}.txt"
        for seed in SEEDS
        for model in MODELS
    ],
    *[
        f"FromRomy/Finalimbs_sampled1_seed{seed}_{model}.txt"
        for seed in SEEDS
        for model in MODELS
    ],
]

OPTIONAL_GROUPS = {
    "gobm_stratified_id_input_audit": [
        "south_dim_input.csv",
        "south_scales_input.csv",
        "north_dim_input.csv",
        "north_scales_input.csv",
    ],
    "knn_reconstructed_fold_scores": [
        "knn_weighted_scores_seed13_fold_metrics.csv",
    ],
    "unused_gobm_20k_dii": [
        "Finalimbs_sampled20k_ETHZ.txt",
        "Finalimbs_sampled20k_IPSL.txt",
        "Finalimbs_sampled20k_MRI.txt",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check provenance intermediates.")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediates"))
    args = parser.parse_args()

    any_missing = False
    for group, filenames in REQUIRED_GROUPS.items():
        missing = []
        for name in filenames:
            candidates = [
                args.intermediate_dir / name,
                args.intermediate_dir / "FromRomy" / name,
            ]
            if not any(candidate.exists() for candidate in candidates):
                missing.append(name)
        present = len(filenames) - len(missing)
        print(f"{group}: {present}/{len(filenames)} present")
        if missing:
            any_missing = True
            for name in missing:
                print(f"  missing: {name}")

    for group, filenames in OPTIONAL_GROUPS.items():
        missing = []
        for name in filenames:
            candidates = [
                args.intermediate_dir / name,
                args.intermediate_dir / "FromRomy" / name,
            ]
            if not any(candidate.exists() for candidate in candidates):
                missing.append(name)
        present = len(filenames) - len(missing)
        print(f"{group}: {present}/{len(filenames)} present (optional)")
        if missing:
            for name in missing:
                print(f"  optional missing: {name}")

    if any_missing:
        raise SystemExit(1)

    print("All checked provenance intermediates are present.")


if __name__ == "__main__":
    main()
