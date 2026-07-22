# CO2RepresentCode

This repo contains code and data products associated with the paper "How well is surface ocean carbon represented in observations and ocean models?" by Viviana Acquaviva, Romina Wild, Alessandro Laio, Amanda R. Fay, Thea H. Heimdal, and Galen A. McKinley.

## Contents

- `scripts/`: figure plotting scripts and validation helpers.
- `intermediates/`: lightweight CSV/TXT inputs consumed by the plotting scripts.
- `outputs/figures/`: generated PNG outputs for all technical figures in the paper.
- `provenance_notebooks/`: notebooks showing how intermediate files can be regenerated from raw data.
- `FigureGenerationList.txt`: figure-to-script map for manuscript and supplementary figures.

The raw zarr/NetCDF inputs and large GOBM pickle files are not included in this repository because of their size (~50 Gb), but can be provided upon request.

## Quick Start

Create an environment:

```bash
conda env create -f environment.yml
conda activate co2represent
```

Check that all required lightweight intermediates are present:

```bash
python scripts/check_provenance_intermediates.py
```

Regenerate all script-produced figures:

```bash
python scripts/verify_figures.py \
  --intermediate-dir intermediates \
  --romy-dir intermediates/FromRomy \
  --output-dir outputs/figures
```

Expected result:

```text
Verification passed: all expected figure outputs exist and are non-empty.
```

## Figure Map

See `FigureGenerationList.txt` for the manuscript/supplementary figure list and the script that generates each figure or component.

## Provenance Notebooks

The numbered notebooks in `provenance_notebooks/` document how the intermediate CSV/TXT files were generated. These notebooks are optional regeneration workflows and require the raw SOCAT/LEAP zarr data, the RECCAP2 region-mask NetCDF file, and/or large upstream GOBM pickle files depending on the notebook.

The script `scripts/check_provenance_intermediates.py` groups the saved intermediate files by provenance workflow and can be used as the compact map between the cleaned notebooks and the included CSV/TXT files.

The plotting scripts do not require raw zarr or pickle files; they use the saved files in `intermediates/`.

## Raw Data

Raw inputs are intentionally excluded from version control because they are large. The cleaned provenance notebooks use local paths under `raw/` by default where applicable, for example:

- `raw/pCO2_LEAP_fco2-residual-full-dataset-preML_198201-202412.zarr`
- `raw/socat_mask_feb1982-dec2022.zarr`
- `raw/RECCAP2_region_masks_all_v20221025.nc`

Large GOBM pickle inputs are referenced by provenance notebooks using their configured storage-bucket paths or an equivalent local directory.
