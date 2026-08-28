# Running ArchesClimate-SSP inference outside the training data lake

This describes exactly what a checkpoint needs at inference time, so a
model can be run (or its outputs reproduced).

## The three kinds of input a rollout needs

Everything `analysis/long_rollout.py` reads falls into one of these. Only
the first is per-checkpoint; the rest are shared across many runs/scenarios.

### 1. Model weights (per checkpoint)

- `<model_dir>/config.yaml` — the exact Hydra config saved at training launch time.
- `<model_dir>/checkpoints/checkpoint_name.ckpt` - the weights. Has both a raw and an EMA copy inside it; `inference.use_ema` selects which is used.
- `model_dir` = `<cluster.output_path>model_output/<project>/<name>/`, where
  `project` is `cmip-interpolation` (deterministic/enery_flow) or
  `cmip-interpolation-flow` (flow/generative).
- Flow models additionally need `stats/cmip_delta_std_<deterministic_model_name>` — the deterministic model's residual-magnitude stats, referenced by `delta_std_name` inside the flow model's own config.

### 2. Scenario data — initial condition + full forcing trajectory, per scenario/realization

Both come out of the same per-experiment memmap (could be useful to separate these):

- **Initial condition**: `surface`, `level`, and `lev` fields at
  `index = 120 * inference.year` inside the memmap.  the whole
  timeseries — just the starting snapshot the autoregressive model steps
  forward from. Shapes at that one timestep: `surface` `(8, 144, 144)`,
  `level` `(5, 17, 144, 144)`, `lev` `(1, 10, 144, 144)` (float32;
  channel counts and pressure-level count are IPSL-grid values, see
  `module.*_variables` in the checkpoint's `config.yaml` for the
  CanESM5-native equivalents).
- **Forcing trajectory**: the external forcings (GHG, aerosol, ozone,
  solar) for every month of the rollout horizon, conditioning each
  autoregressive step — `spatial_forcings` `(19, T, 144, 144)`,
  `non_spatial_forcings` `(6, T)`, `ozone` `(1, T, 66, 144, 144)`.
- Static orography (grid-specific, not scenario-specific, so packaged
  separately): `reference_data/orography.pt` (IPSL grid) or
  `reference_data_canesm5_native/orography.pt` (CanESM5-native grid),
  read once if `module.module.add_orography: True`.

### 3. Normalization stats

- `stats/cmip_stats_<norm_scheme>.pt` (~20 MB) — `{surface,level,lev,
  spatial_forcings,non_spatial_forcings}_{mean,std}` tensors. `norm_scheme`
  comes from the checkpoint's own `config.yaml` (e.g. `new_ozone`,
  `new_ozone_canesm5_native`).
- `stats/cmip_delta_std_<deterministic_model_name>` (~8 MB) — flow models
  only, see §1. Extension-less, so it escapes the `*.pt` gitignore rule and
  is genuinely checked into git already — no action needed for this one.

A plain `hf download gclyne/ArchesClimate-data --local-dir
<data_dir>` (§ "Uploading/downloading via Hugging Face" below) puts `cmip_stats_*.pt` under
`<data_dir>/stats/`, a directory *you* chose — it does not land in `<repo>/stats/` on its own.
Download it straight to the right place instead, using `--include` to grab just that subdir and
`--local-dir .`

```bash
uv run hf download gclyne/ArchesClimate-data --type dataset --include "stats/*" --local-dir .
```

## Data layout on disk

```
<cluster.data_path>/                          # e.g. /scratch/gclyne/
  <module.dataset_path>/                       # e.g. memmap_filled_in/
    <realization>_<experiment>_interpolation.memmap/   # TensorDict.load_memmap() directory, ~28 GB each
      meta.json                                # shapes/dtypes for surface/level/lev/spatial_forcings/non_spatial_forcings/ozone
      surface.memmap  level.memmap  lev.memmap
      spatial_forcings.memmap  non_spatial_forcings.memmap  ozone.memmap
      time
  reference_data/                              # or reference_data_canesm5_native/
    orography.pt
    (other forcing reference tensors not needed for standard inference)

/scratch/gclyne/model_output/<project>/<name>/
  config.yaml
  checkpoints/checkpoint_name.ckpt

<repo>/stats/
  cmip_stats_<norm_scheme>.pt
  cmip_delta_std_<deterministic_model_name>    # flow models only
```

Memmaps are `tensordict.TensorDict.load_memmap()` directories (raw
per-channel float32 arrays + `meta.json`), not netCDF. `.pt` stats files
are plain `torch.save` dicts of tensors.

## Minimal inference recipe

On SLURM:

```bash

sbatch analysis/long_rollout.sh \
    module=<module_config_name> name=<checkpoint_run_name> cluster=cleps \
    inference.ckpt_fname='step-step=NNNNNN.ckpt' \
    inference.target=val \                # val=ssp434, val3=ssp534-over, test=ssp585
    inference.debug=False inference.save_all_vars=True \
    inference.use_ema=True
```

Without SLURM (`long_rollout.sh` is just a thin sbatch wrapper around `long_rollout.py`, same
as `submit.py`/`main_hydra.py` for training):

```bash
uv run python -m analysis.long_rollout \
    module=<module_config_name> name=<checkpoint_run_name> cluster=<your cluster.yaml> \
    inference.ckpt_fname='step-step=NNNNNN.ckpt' inference.target=val \
    inference.debug=False inference.save_all_vars=True inference.use_ema=True
```

