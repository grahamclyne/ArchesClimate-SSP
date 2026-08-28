# Downloading source data

ArchesClimate is trained on two CMIP6 source models: **IPSL-CM6A-LR** and
**CanESM5**. Raw variables come from ESGF / input4MIPs; there is no separate
"download-only" step in this pipeline -- downloading and the first regrid to
the model's native `2.5x1.25` grid happen together, driven by
[ESMValTool](https://esmvaltool.org).

## What was pulled

Per source model, per experiment (historical, piControl, the SSPs, and for
IPSL the abrupt-2xCO2/4xCO2 idealized runs), ESMValTool fetches:

- Surface: `tas, psl, pr, uas, vas, ps, hfss, hfls, huss, rlds, rsds, rsus,
  rlus, evspsbl` (+ `net_flux`, derived from the radiative/heat fluxes above)
- Atmospheric levels: `hus, ta, ua, va, zg`
- Ocean: `thetao, so` (+ `uo, vo` for CanESM5)

The exact experiment/ensemble-member list for each scenario is declared in
the corresponding `interpolation_dataset_<exp>.yml` /
`interpolation_dataset_canesm5_<exp>.yml` recipe in this directory -- treat
those 21 files as the authoritative download manifest, not this README.

Separately, `forcings_preprocessing.py` builds reference forcing tensors
(spatial GHG concentrations, ozone) from raw CMIP6/input4MIPs forcing files.
Those outputs already exist under `reference_data/` for every experiment
currently used and do not need to be rerun -- see that script's docstring
and `reference_data/README.md`.

## Running a download

Built and tested against **ESMValTool 2.13.0** (`esmvaltool --version`). This environment is
entirely separate from the main `uv sync` one (see `README.md`) — nothing here checks that
the two stay compatible, so if recipes start failing in ways unrelated to your changes, check
this version first before debugging further.

```bash
conda activate esmvaltool-env1
esmvaltool run interpolation_dataset_<exp>.yml \
    --search_esgf=when_missing --remove_preproc_dir=True
```

`run_esmvaltool_script_cleps.sh` wraps this as an sbatch job:
`sbatch run_esmvaltool_script_cleps.sh interpolation_dataset_<exp>.yml`.

**Before running, edit the recipe's `script:` field.** ESMValTool requires
an absolute path to the diagnostic script that consumes the downloaded data
(`prepare_interpol_dataset.py` for IPSL, `prepare_interpol_dataset_canesm5.py`
for CanESM5). All 21 recipes currently hardcode this author's path
(`/home/gclyne/ArchesClimate/ArchesClimate/preprocessing/...`) -- point it at
your own checkout before running.

This is a slow, ESGF-rate-limited step (`--search_esgf=when_missing` will
download anything not already cached locally); expect it to be the long pole
in reproducing the dataset from scratch.

Once downloaded and merged, continue with [`README.md`](README.md) for the
rest of the preprocessing pipeline.
