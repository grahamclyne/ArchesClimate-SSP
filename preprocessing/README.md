# Preprocessing pipeline

Turns the raw CMIP6 data described in [`DOWNLOADING.md`](DOWNLOADING.md) into
the memmap tensors the dataloaders (`dataloader/`) read at training time.
Run in the order below; each stage's output directory is the next stage's
input. Everything writes under `$SCRATCH` unless noted.

Two source models are supported, with parallel but not identical scripts:
**IPSL-CM6A-LR** and **CanESM5**. IPSL was the original model this pipeline
was built for; CanESM5 was added later and its scripts are the ones actively
maintained -- see the per-stage notes below for what that means in practice.

## Before running anything

Every `.sh` / `.sbatch` wrapper and every `interpolation_dataset_*.yml`
recipe hardcodes this project's original cluster paths and SLURM account:

- `ARCHESCLIMATE_ROOT` (default `/home/gclyne/ArchesClimate/ArchesClimate`)
  and `SCRATCH` (default `/scratch/gclyne`) -- override via environment
  variable, or edit the defaults in each `.sh`/`.sbatch` file.
- `#SBATCH --partition=arches --account=arches` -- a cluster-specific
  allocation name; replace with your own.
- Each `interpolation_dataset_*.yml`'s `script:` field -- an absolute path,
  required by ESMValTool, that must point at your own checkout (see
  `DOWNLOADING.md`).

## Stage 0 -- reference forcing tensors (one-time, likely already done)

`forcings_preprocessing.py` regrids raw CMIP6/input4MIPs GHG and ozone
forcing files (via `cdo remapcon`) into `.pt` tensors under `reference_data/`.
Outputs already exist there for every experiment currently used -- this only
needs rerunning if you add a new experiment/scenario. Its ozone step reads
from a Jean-Zay-specific path that no longer exists; re-source the raw ozone
climatology files from ESGF/input4MIPs and update the path if you need to
extend this.

## Stage 1 -- download + merge (ESMValTool)

Covered in [`DOWNLOADING.md`](DOWNLOADING.md). Produces:

- IPSL: NetCDF files in `$SCRATCH/interpolation_project_datasets/`, via
  `prepare_interpol_dataset.py`.
- CanESM5: memmaps directly in `$SCRATCH/interpolation_canesm5_datasets/`,
  via `prepare_interpol_dataset_canesm5.py` (skips the NetCDF round-trip).

Both call into shared merge logic in `common/prepare_interpol_dataset.py`.

## Stage 2 -- gap-filling / vertical regrid

Raw CMIP6 output has missing values (e.g. under sea ice, below ocean floor)
and CanESM5's vertical levels don't match IPSL's, so this stage fills gaps
and, for CanESM5, remaps onto IPSL's pressure/depth levels for consistency
between the two source models.

- IPSL (current): `fill_in_missing_ipsl.py` (`fill_in_missing_ipsl.sbatch`) --
  reads NetCDF from `interpolation_project_datasets`, explicitly selects the
  standard 17 pressure / 10 depth levels via `.sel(..., method="nearest")`,
  and applies the same vectorized gap-fill as CanESM5
  (`common/fill_in_missing.py`) to the 5 level variables plus `thetao`.
  Replaces `fill_in_missing_data.py` (kept for reference below), which used
  xarray's per-file `interpolate_na` and didn't touch `thetao` at all. Not
  yet verified against real IPSL output -- written before any new
  experiment's Stage 1 download had finished; spot-check a resulting file
  once one has (physical ranges, permanently-masked vs. gap-filled cells).
- IPSL (superseded): `fill_in_missing_data.py` (`fill_in_missing_data.sh`) --
  xarray `interpolate_na` over NetCDFs. Its original input directory
  (`interpolation_project_datasets`) was cleaned up from disk at the time,
  but the script itself is not broken -- it just has no fresher IPSL
  replacement reason to run over `fill_in_missing_ipsl.py` now. Kept as a
  record of how the current `interpolation_project_datasets_filled_in` files
  were originally built.
- CanESM5: `fill_in_missing_canesm5.py` (`fill_in_missing_canesm5.sbatch`) --
  remaps to IPSL's 17 pressure levels and 10 ocean depth levels
  (`canesm5_levels.py`), reorders surface variables to IPSL's order, then
  applies the vectorized gap-fill in `common/fill_in_missing.py`. Supports
  SLURM array jobs (`SLURM_ARRAY_TASK_ID`/`SLURM_ARRAY_TASK_COUNT`) to
  parallelize across files.

  Note: `common/fill_in_missing.py`'s algorithm has only been verified
  against CanESM5 output -- it was never diffed against real IPSL data,
  since IPSL's pre-fill NetCDFs were already deleted by the time it was
  written.

Writes to `$SCRATCH/interpolation_{project,canesm5}_datasets_filled_in/`.

## Stage 3 -- forcing assembly into final tensors

Combines the gap-filled climate fields with the Stage 0 forcing tensors
(GHG, aerosol, ozone, solar) into the final per-experiment memmaps consumed
by training.

- **CanESM5 (current, maintained path):** `add_forcings_canesm5.py`
  (`add_forcings_canesm5.sbatch`) reads
  `interpolation_canesm5_datasets_filled_in`, assembles `spatial_forcings`
  (19 channels: 3 GHGs + 6 aerosols + 10 ozone bands) and
  `non_spatial_forcings` (6 solar bands) via `common/forcings.py`, and writes
  to `$SCRATCH/memmap_filled_in_canesm5/`.
- **IPSL (current):** `add_forcings.py` (`add_forcings_ipsl.sbatch`) reads
  `interpolation_project_datasets_filled_in` (NetCDF), stacks the named
  surface/level/`thetao` variables into `memmap_filled_in`'s tensor layout,
  assembles forcings via `common/forcings.py` (same source/logic as
  CanESM5's script), and writes directly to `$SCRATCH/memmap_filled_in/` --
  skipping the intermediate `full_ocean_dataset_forced` NetCDF the two legacy
  scripts below round-tripped through. Not yet verified against real IPSL
  output for the same reason as `fill_in_missing_ipsl.py`; run once that
  stage's output exists and spot-check a resulting memmap (physical ranges,
  `net_flux` sign, a few known grid cells) before trusting it for training.
- **IPSL (legacy):** the scripts that originally built `memmap_filled_in/`
  -- `legacy/add_forcings.py` (predates `common/forcings.py`, used the old
  6-band ozone selection and an unused 4th gas, `cfc11`) and its downstream
  NetCDF-to-memmap step `legacy/convert_xarray_to_memmap.py` -- neither runs
  anymore (their intermediate NetCDF directories were cleaned up after
  `memmap_filled_in` was produced). Kept purely as a record of how the
  existing `memmap_filled_in/` files were built; do not try to rerun them --
  use `add_forcings.py` above for any new experiment instead.

## Stage 4 -- post-hoc patches (already applied)

`fix_ozone.py` retroactively replaced IPSL's original 6-band ozone selection
in `memmap_filled_in/*` with the corrected 10-band selection (bringing it to
the same 19-channel layout CanESM5 uses). It's idempotent -- files already at
19 channels are skipped -- so it's safe to rerun if you're not sure whether
a given file has been patched.

`aerosol_patch_job.ipynb` fixed `ssp534-over`'s aerosol forcing channels:
since `ssp534-over` branches off `ssp534` in 2040 rather than starting at the
usual SSP start year, its aerosol reference tensor needed the first 25 years
sliced off and 2 trailing months zero-padded to align with the memmap's time
axis, then overwrote `spatial_forcings[3:9]` in place for the affected files.
Already applied; kept as a record of the fix, not something to rerun.

## Verification (optional, not required to reproduce the pipeline)

`verification/` holds one-off QA scripts used to validate specific steps
above: cross-checking CanESM5 vs. IPSL forcings, confirming the
`common/forcings.py` refactor was behavior-preserving, and checking the
ozone/ssp534-over patches landed correctly. None of these produce pipeline
outputs -- run them only if you want to double-check a stage's result.

## What's not part of this pipeline

- `zonal.yml` (moved to `analysis/`) is a model-evaluation recipe (zonal-mean
  bias plots vs. IPSL), not a data-prep step.
- `analysis/normalization_sanity_check.ipynb` (moved from here) sanity-checks
  the dataloader's normalize/denormalize round-trip, not raw-data prep.

## Using this pipeline for your own dataset

Everything above is written in terms of the two source models this repo
already ships with (IPSL-CM6A-LR, CanESM5), but nothing in Stages 0-4 is
actually specific to those two models -- `common/prepare_interpol_dataset.py`
and `common/forcings.py` are explicitly model-agnostic (see their module
docstrings), and CanESM5 itself was added later by following this same
pattern. Adding a third CMIP6 model, a different set of experiments, or your
own non-CMIP6 data follows the same five steps regardless of which one
you're doing:

### 1. Download: pick or write an ESMValTool recipe

Downloading and the first regrid happen together in one ESMValTool run (see
[`DOWNLOADING.md`](DOWNLOADING.md)) -- there's no separate raw-download step.
To add a model already supported by ESMValTool's CMIP6 dataset backend
(i.e. anything on ESGF):

1. Copy the nearest `interpolation_dataset_<exp>.yml` (use
   `interpolation_dataset_canesm5_<exp>.yml` as the template if your model
   needs its own native grid preserved through training, like CanESM5;
   `interpolation_dataset_<exp>.yml` if regridding straight onto IPSL's grid
   is fine, like the current IPSL recipes themselves).
2. In the `datasets:` block, change `dataset:` to your model's CMIP6 name,
   and set `ensemble:`/`grid:`/`start_year:`/`end_year:` for what you
   actually want (the commented-out blocks in each recipe show the pattern
   for adding more experiments/ensemble members to the same file).
3. In `preprocessors: process_var: regrid:`, `target_grid` is the grid every
   variable gets interpolated onto. Leave it at `"2.5x1.25"` to land your new
   model directly on IPSL's grid (simplest -- no downstream config changes
   needed for grid shape). Set it to your model's own native resolution
   instead if you want to preserve it (CanESM5's path) -- see step 5 for what
   that costs you downstream.
4. The `variables:` list under `diagnostics: regrid_and_combine:` is the
   download manifest -- add/remove entries to match what you need. Whatever
   you list here is what Stage 3 (`add_forcings*.py`'s `SURFACE_VARS`/
   `LEVEL_VARS`/`DEPTH_VARS`, or CanESM5's equivalent) must be able to find
   by name later, so keep the CMIP6 variable names.
5. Point `diagnostics: regrid_and_combine: scripts: my_diagnostic: script:`
   at your own checkout's `prepare_interpol_dataset.py` (or
   `prepare_interpol_dataset_canesm5.py` if you copied that template)
   -- an ESMValTool requirement, must be an absolute path.
6. Run it per `DOWNLOADING.md`. Repeat per experiment (one recipe = one
   experiment/scenario).

If your data isn't on ESGF at all (a non-CMIP6 source, or private/unpublished
runs), ESMValTool's recipe mechanism doesn't apply -- you'd need to produce
NetCDF files with the same variable names/dimensions
`common/prepare_interpol_dataset.py` expects (one file per variable per
experiment/ensemble member, mergeable by esmvaltool's `input_data` metadata)
and either write a minimal stand-in for Stage 1 or feed them directly into
Stage 2 gap-filling, whichever is less work for your actual file layout.

### 2. Integrate: Stage 1 merges, Stage 2 gap-fills

`prepare_interpol_dataset.py`/`prepare_interpol_dataset_canesm5.py` (called
by ESMValTool as the diagnostic `script:` from step 1 above) merge the
per-variable files ESMValTool downloaded into one NetCDF per
experiment/ensemble member and derive `net_flux` from the six raw
radiation/heat-flux variables (`common/prepare_interpol_dataset.py`'s
`merge_cubes_and_compute_net_flux`) -- this is "integration" in the sense of
combining separately-downloaded variables into one file per member, not
combining different source models together (IPSL and CanESM5 stay in
separate `memmap_filled_in*` trees end to end; a training run picks one via
`dataset_path`, it doesn't blend them).

Stage 2 (`fill_in_missing_ipsl.py` / `fill_in_missing_canesm5.py`, sharing
`common/fill_in_missing.py`'s gap-fill algorithm) then patches missing values
(sea ice, below ocean floor) and, for a model whose vertical levels don't
match IPSL's 17 pressure / 10 depth levels, regrids onto them (or your own
target levels, if you're keeping a native grid -- see `canesm5_levels.py` for
the conservative-remap template, and update `IPSL_DEPTH_LEVELS` there / your
module config's `depth_levels` together if you pick different target depths).

### 3. Memmap: Stage 3 assembles the final tensors

`add_forcings.py` / `add_forcings_canesm5.py` read the gap-filled NetCDFs,
stack the named surface/level/depth variables into one tensor per state
component, call `common/forcings.py` to build the matching
`spatial_forcings`/`non_spatial_forcings` tensors for that file's experiment
(purely a function of experiment name + the Stage 0 `reference_data/`
tensors -- not model-specific), and write the result with
`TensorDict(...).memmap()` (or equivalent) to
`$SCRATCH/memmap_filled_in<_yourmodel>/`. This is literally "how it becomes
a memmap" -- there's no separate conversion step beyond this; whichever
`add_forcings_*.py` variant you're using **is** the NetCDF-to-memmap step.
Copy `add_forcings_canesm5.py` as your template if you're adding a new
model (it's the actively-maintained one; `add_forcings.py` is the IPSL
equivalent but not yet re-verified since its last rewrite -- see its
docstring). The output directory name is up to you; it just needs to match
what you point `dataset_path` at in step 5.

### 4. Stats: regenerate normalization before training

Every module config's `norm_scheme` names a `cmip_stats_<norm_scheme>.pt`
file under `ArchesClimate/stats/` (per-channel mean/std for
surface/level/lev/spatial_forcings/non_spatial_forcings), loaded by
`dataloaders/cmip_random_lead_time.py` at dataset construction time -- a new
memmap tree needs its own stats file before you can train on it, since a
different model/variable-set/grid has different physical value
distributions:

- `utils/generate_stats.py` (`utils/generate_stats.sh`, or
  `_canesm5.sh`/`_canesm5_native.sh` for those grids)
  computes the full surface/level/lev/spatial_forcings/non_spatial_forcings
  stats from a sample of your new memmap tree via Welford's algorithm --
  point its Hydra overrides (`dataloader.dataset.path`, `module=`, `name=`,
  which becomes the output `cmip_stats_<name>.pt` filename) at your data.
- `utils/generate_forcing_stats.py` recomputes just the forcing channels
  (cheaper, useful if you only changed `spatial_forcing_variables` /
  `non_spatial_forcing_variables` and want to reuse an existing
  surface/level/lev stats file via `+base_stats_name=<existing norm_scheme>`).

### 5. Configs: what actually needs to change to train on it

Once the memmap tree and stats file exist, point a module config at them.
Copy the nearest existing module config (per `CLAUDE.md`, don't use Hydra
`defaults:` chains -- keep it a standalone file) and set:

- `dataset_path`: the directory name from step 3, under
  `${cluster.data_path}` (e.g. `memmap_filled_in_yourmodel`).
- `surface_variables` / `level_variables` / `depth_variables`: must be a
  **prefix subset, in the same order**, of whatever `add_forcings_*.py`'s
  `SURFACE_VARS`/`LEVEL_VARS`/`DEPTH_VARS` (or CanESM5's equivalent) wrote
  into the memmap -- the dataloader slices by position, not by name (see
  `dataloaders/cmip_random_lead_time.py`'s slicing comments).
- `spatial_forcing_variables` / `non_spatial_forcing_variables`: must be a
  subset of `common/forcings.py`'s master channel list, in the exact channel
  order that list defines (GHGs, then aerosols, then ozone bands for
  spatial; solar bands for non-spatial) -- see
  `dataloaders/cmip_random_lead_time.py`'s `master_list_spatial_forcing_variables`
  for the authoritative order, and its comment about ozone-band-append
  dedup if your memmap already has the bands baked in.
- `pressure_levels` / `depth_levels`: must match the levels your Stage 2
  regrid actually produced (IPSL's 17/10 by default, or your own if you
  picked different target levels in step 2).
- `norm_scheme`: the `name=` you used when generating stats in step 4.
- `train_filter` / `val_filter` / `test_filter`: which experiments (must
  match the `_interpolation.memmap` filename suffixes your Stage 3 run
  produced) go into each split.
- A matching `configs/dataloader/*.yaml` entry if you need a different
  `forcings_path` (Stage 0's `reference_data/` tree is grid-specific --
  CanESM5-native has its own `reference_data_canesm5_native/`, built by
  `regrid_reference_forcings_canesm5_native.py`, precisely because it isn't
  on IPSL's grid). Reuse `configs/dataloader/cmip_random_lead_times.yaml` if
  you regridded onto IPSL's grid in step 1; otherwise copy
  `cmip_random_lead_times_canesm5_native.yaml` as the template, including
  regridding your own `orography.pt` the same way.
- If you changed grid shape (not just source model on the same grid):
  `lat_dim`/`lon_dim` in the module config (see `canesm5_damip.yaml`'s
  comment on why/where these are threaded through
  `model/base_climate_module.py`'s `prep_model()`), and the `backbone:`
  block's `tensor_size`/`window_size` and `embedder:` block's `img_size`
  (see `canesm5_damip.yaml` for the worked 64x128 example and why
  `window_size` was picked as `[1, 8, 16]` there).

At that point `module=<your_new_config>` behaves like any other module
config for training, rollout, and the analysis pipeline.
