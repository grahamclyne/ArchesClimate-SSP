"""Recompute normalization stats for the forcing channels only.

Regenerating full stats (surface/level/lev) requires reading the whole
atmospheric state for every sampled timestep, which is expensive because
`CMIPForecastLeadTime.__getitem__` reads the state window (prev + next +
8 pushforward steps, possibly multistep) for every single index — the
forcing arrays ('spatial_forcings' / 'non_spatial_forcings') are a small
fraction of that read volume and don't need the temporal window at all
(a single per-timestep read is enough to estimate their marginal
mean/std). This script bypasses CMIPForecastLeadTime.__getitem__ and reads
forcing channels directly off the per-file memmap (mirroring the
slicing/trimming logic in dataloaders/netcdf.py:182-217), so it can
afford to scan far more timesteps than the 1000-sample subsample
generate_stats.py uses for the full state.

It loads an existing full stats file (for the surface/level/lev entries,
which this script does not recompute) and writes out a new stats file with
only spatial_forcings_mean/std and non_spatial_forcings_mean/std replaced.

Usage (note the '+': base_stats_name/num_samples aren't in the base config
schema, so Hydra's struct mode requires '+key=value' to add them):
    python utils/generate_forcing_stats.py \
        module=forcing_dropout_no_random_lt_co2log \
        name=forcing_dropout_no_random_lt_co2log_scalar \
        +base_stats_name=new_ozone_co2log \
        [+num_samples=0]   # 0 = use every timestamp in the training set
"""

import os
import sys

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from tensordict.tensordict import TensorDict

root = os.environ.get("ARCHESCLIMATE_ROOT", "/home/gclyne/ArchesClimate/ArchesClimate")
with initialize_config_dir(config_dir=f"{root}/configs"):
    cfg = compose(config_name="config", overrides=sys.argv[1:])

L.seed_everything(cfg.seed)

base_stats_name = cfg.get("base_stats_name", None)
if base_stats_name is None:
    raise ValueError(
        "Pass +base_stats_name=<existing norm_scheme> to reuse its "
        "surface/level/lev stats (this script only recomputes forcings)."
    )
num_samples = int(cfg.get("num_samples", 0))  # 0 = every timestamp

# The dataset's __init__ loads cmip_stats_{norm_scheme}.pt unconditionally
# (to build normalization tensors we never use here), so instantiate against
# the existing base stats file instead of the one this script generates.
train = hydra.utils.instantiate(
    cfg.dataloader.dataset,
    domain="train",
    norm_scheme=base_stats_name,
)

import importlib.resources  # noqa: E402

from ArchesClimate import stats as ArchesClimate_stats  # noqa: E402

base_stats_path = (
    importlib.resources.files(ArchesClimate_stats) / f"cmip_stats_{base_stats_name}.pt"
)
base_stats = torch.load(base_stats_path, weights_only=False)

spatial_vars = train.variables["spatial_forcings"]
non_spatial_vars = train.variables["non_spatial_forcings"]

n_total = len(train.id2pt)
if num_samples and num_samples < n_total:
    indices = torch.randperm(n_total)[:num_samples].tolist()
else:
    indices = range(n_total)

means = {}
M2 = {}
counts = {}


def update(name, x):
    if name not in means:
        means[name] = torch.zeros_like(x, dtype=torch.float32)
        M2[name] = torch.zeros_like(x, dtype=torch.float32)
        counts[name] = torch.zeros_like(x, dtype=torch.int32)
    mask = ~torch.isnan(x)
    n = counts[name]
    n_new = n + mask.int()
    delta = torch.zeros_like(x, dtype=torch.float32)
    delta2 = torch.zeros_like(x, dtype=torch.float32)
    delta[mask] = x[mask] - means[name][mask]
    means[name][mask] += delta[mask] / n_new[mask].clamp_min(1)
    delta2[mask] = x[mask] - means[name][mask]
    M2[name][mask] += delta[mask] * delta2[mask]
    counts[name] = n_new


_file_cache = {}
for count, i in enumerate(indices):
    if count % 5000 == 0:
        print(f"{count} / {len(indices)}")
    file_id, line_id, _timestamp = train.id2pt[i]
    file_path = train.files[file_id]
    if file_path not in _file_cache:
        _file_cache[file_path] = TensorDict.load_memmap(file_path)
    data = _file_cache[file_path]

    # Mirrors dataloaders/netcdf.py:182-217 exactly, forcings only.
    spatial_forcings = data["spatial_forcings"][:, line_id]
    if "ozone_0" not in spatial_vars:
        spatial_forcings = spatial_forcings[:8]
    elif "ozone_1" not in spatial_vars:
        spatial_forcings = spatial_forcings[:9]
    elif "ozone_7" not in spatial_vars:
        spatial_forcings = spatial_forcings[:15]
    if "load_ASNO3M" not in spatial_vars:
        spatial_forcings = torch.cat([spatial_forcings[:3], spatial_forcings[9:]], dim=0)
    if "methane" not in spatial_vars:
        spatial_forcings = spatial_forcings[3:]

    non_spatial_forcings = (
        data["non_spatial_forcings"][:, line_id] if len(non_spatial_vars) > 0 else torch.tensor([])
    )

    mask = (spatial_forcings.abs() > 1e30) | torch.isinf(spatial_forcings)
    spatial_forcings = torch.where(mask, torch.nan, spatial_forcings)

    update("spatial_forcings", spatial_forcings)
    if non_spatial_forcings.numel() > 0:
        update("non_spatial_forcings", non_spatial_forcings)

output = dict(base_stats)
for key in means:
    mean = means[key].clone()
    var = torch.full_like(mean, float("nan"))
    std = torch.full_like(mean, float("nan"))
    mask = counts[key] > 1
    mean[~mask] = torch.nan
    var[mask] = M2[key][mask] / (counts[key][mask] - 1)
    std[mask] = torch.sqrt(var[mask])
    output[f"{key}_mean"] = mean
    output[f"{key}_std"] = std

# CMIPForecastLeadTime.__init__ reindexes spatial_forcings_mean/std against
# master_list_spatial_forcing_variables (which always ends in a phantom
# "ozone_10" slot -- the on-disk data only has 10 real ozone bands) and then
# unconditionally drops the last row (dataloaders/cmip_random_lead_time.py,
# "accidentally added orography to full_ozone normalization"). That means
# every stats file needs one extra trailing row that is never actually read
# -- pad with a duplicate of the last row rather than leaving it one short
# (which raises IndexError at dataset __init__ before training even starts).
if "spatial_forcings_mean" in output:
    output["spatial_forcings_mean"] = torch.cat(
        [output["spatial_forcings_mean"], output["spatial_forcings_mean"][-1:]], dim=0
    )
    output["spatial_forcings_std"] = torch.cat(
        [output["spatial_forcings_std"], output["spatial_forcings_std"][-1:]], dim=0
    )

output = TensorDict(output)
out_path = f"{root}/stats/cmip_stats_{cfg.name}.pt"
torch.save(output, out_path)
print(
    f"Wrote {out_path} using {len(indices)} forcing samples "
    f"(surface/level/lev copied unchanged from cmip_stats_{base_stats_name}.pt)"
)
