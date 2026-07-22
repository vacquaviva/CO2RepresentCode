#!/usr/bin/env python3
"""Verify figure scripts and their expected outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PLOT_JOBS = [
    (
        "plot_id_socat_models.py",
        ["ID_SOCAT_models_v2.png"],
        [],
    ),
    (
        "plot_socat_id_and_summary.py",
        ["ID_SOCAT_square.png", "SOCAT_three_panel_summary.png"],
        ["--weights-dir", "{intermediate_dir}"],
    ),
    (
        "plot_convergence_summary.py",
        ["combined_ID_weights_DII.png"],
        ["--weights-dir", "{intermediate_dir}"],
    ),
    (
        "plot_comparisons.py",
        [
            "GOBMs_delta_1Dxco2.png",
            "Mean_DII_SOCAT_square.png",
            "Mean_feature_weights_SOCAT_noxlabels.png",
            "DII_across_time.png",
            "Mean_featureweights_acrosstime_noxlabels.png",
        ],
        [],
    ),
    (
        "plot_gobms_vivi_romy.py",
        [
            "Weights_full_vs_SOCAT_n7_n13.png",
            "DII_with weights_full_vs_SOCAT_inline.png",
            "DII_models_vs_SOCAT_inline.png",
            "Weights_n13_models_vs_SOCAT.png",
            "Weights_n7_models_vs_SOCAT.png",
            "DII_from_models_in_SOCAT.png",
            "DII_from_models_in_SOCAT_seed10.png",
            "DII_from_models_in_SOCAT_seed13.png",
        ],
        ["--romy-dir", "{romy_dir}"],
    ),
    (
        "plot_stratified_sampling_domains.py",
        ["SamplingDomains.png"],
        ["--romy-dir", "{romy_dir}"],
    ),
    (
        "plot_id_across_time.py",
        ["ID_across_time.png"],
        [],
    ),
    (
        "plot_subset_three_panel_maps.py",
        [
            "SOCAT_Uniform_Marginals.png",
            "3panels_SOCAT_orig_unif_v2.png",
            "3panels_SOCAT_vs_NA.png",
            "3panels_SOCAT_vs_SO.png",
        ],
        [],
    ),
    (
        "plot_knn_weighted_figures.py",
        [
            "knn_weighted_scores_seed13.png",
            "kNN_weights_models_seed13.png",
        ],
        [],
    ),
]

SAMPLING_DOMAINS_SCRIPT = "plot_stratified_sampling_domains.py"


def format_args(extra_args: list[str], intermediate_dir: Path, romy_dir: Path) -> list[str]:
    return [
        arg.format(intermediate_dir=str(intermediate_dir), romy_dir=str(romy_dir))
        for arg in extra_args
    ]


def sampling_domain_args(models: list[str] | None) -> list[str]:
    if not models:
        return []
    return ["--models", *models]


def output_status(output_dir: Path, outputs: list[str], minimum_bytes: int) -> list[str]:
    problems = []
    for filename in outputs:
        path = output_dir / filename
        if not path.exists():
            problems.append(f"missing output: {path}")
        elif path.stat().st_size < minimum_bytes:
            problems.append(f"small output: {path} ({path.stat().st_size} bytes)")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and verify figure scripts.")
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediates"))
    parser.add_argument("--romy-dir", type=Path, default=Path("intermediates/FromRomy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument(
        "--check-existing",
        action="store_true",
        help="Only check currently existing figure files; do not run plotting scripts.",
    )
    parser.add_argument(
        "--minimum-bytes",
        type=int,
        default=1024,
        help="Minimum size required for a generated PNG to count as non-empty.",
    )
    parser.add_argument(
        "--sampling-domain-models",
        nargs="+",
        choices=["ETHZ", "FESOM", "NorESM", "MRI", "IPSL"],
        help=(
            "Optional model subset for SamplingDomains.png. Useful while one "
            "stratified model intermediate is intentionally unavailable."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    failures: list[str] = []

    for script_name, outputs, extra_args in PLOT_JOBS:
        if not args.check_existing:
            cmd = [
                sys.executable,
                str(scripts_dir / script_name),
                "--intermediate-dir",
                str(args.intermediate_dir),
                "--output-dir",
                str(args.output_dir),
                *format_args(extra_args, args.intermediate_dir, args.romy_dir),
                *(
                    sampling_domain_args(args.sampling_domain_models)
                    if script_name == SAMPLING_DOMAINS_SCRIPT
                    else []
                ),
            ]
            print("Running:", " ".join(cmd))
            result = subprocess.run(cmd, text=True, capture_output=True)
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0:
                failures.append(f"{script_name} failed with exit code {result.returncode}")
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                continue
            if result.stderr:
                print(result.stderr, file=sys.stderr)

        failures.extend(output_status(args.output_dir, outputs, args.minimum_bytes))

    if failures:
        print("\nVerification failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("\nVerification passed: all expected figure outputs exist and are non-empty.")


if __name__ == "__main__":
    main()
