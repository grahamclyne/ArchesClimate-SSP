"""IPSL equivalent of add_forcings_canesm5.py -- modernized replacement for
legacy/add_forcings.py + legacy/convert_xarray_to_memmap.py combined into one
step, writing directly to memmap (skipping the intermediate
full_ocean_dataset_forced NetCDF those two legacy scripts round-tripped
through), the same way add_forcings_canesm5.py does for CanESM5.

Reads gap-filled NetCDF from $SCRATCH/interpolation_project_datasets_filled_in
(fill_in_missing_ipsl.py's output), stacks the named surface/level/lev
variables into memmap_filled_in's tensor layout, and assembles
spatial_forcings/non_spatial_forcings/ozone via common/forcings.py -- same
forcing source and branching logic as add_forcings_canesm5.py, verified
correct there (see common/forcings.py's module docstring).

NOT YET VERIFIED against real IPSL output for the same reason
fill_in_missing_ipsl.py isn't: no real gap-filled IPSL NetCDF has existed to
test against since interpolation_project_datasets_filled_in was last cleaned
up. Run this once fill_in_missing_ipsl.py has produced real output and
spot-check a resulting memmap (physical ranges, net_flux sign, a few known
grid cells) before trusting it for training.

Writes to $SCRATCH/memmap_filled_in.
"""

import glob
import os
import re

import torch
import xarray as xr
from common.forcings import EXPERIMENTS, build_forcings, load_solar_forcings
from fill_in_missing_ipsl import DEPTH_LEVELS, PRESSURE_LEVELS
from tensordict.tensordict import TensorDict

FNAME_RE = re.compile(r"^(?P<ensemble>[^_]+)_(?P<exp>.+)_interpolation\.nc$")

# Matches configs/module/*.yaml's surface_variables / level_variables /
# depth_variables (net_flux is derived at Stage 1, not a raw download --
# see common/prepare_interpol_dataset.py). huss is downloaded (DOWNLOADING.md)
# but, per fill_in_missing_canesm5.py's IPSL_SURFACE_ORDER, isn't one of the
# 8 channels memmap_filled_in's "surface" actually carries.
SURFACE_VARS = ["tas", "psl", "pr", "uas", "vas", "ps", "net_flux", "evspsbl"]
LEVEL_VARS = ["hus", "ta", "ua", "va", "zg"]
DEPTH_VARS = ["thetao"]


def _stack(ds: xr.Dataset, variables: list[str]) -> torch.Tensor:
    """[len(variables), time, ...] tensor, one channel per named variable."""
    return torch.from_numpy(ds[variables].to_array().to_numpy()).float()


def _stack_level(ds: xr.Dataset, variables: list[str]) -> torch.Tensor:
    """Same as _stack, but re-selects the pipeline's standard 17 pressure
    levels first -- fill_in_missing_ipsl.py already does this
    .sel(plev=PRESSURE_LEVELS, method="nearest") for hus/ta/ua/va/zg (mirrors
    its own docstring), but the same DAMIP files that predate the depth-level
    selection (see _stack_depth) predate this one too: their "level" stack
    came out at 19 native plevs instead of 17, which surfaced as a shape
    mismatch (19 vs 17) in utils/generate_stats.py once these files entered
    the training sample -- confirmed by checking fill_in_missing_ipsl.py's
    PRESSURE_LEVELS list has exactly 17 entries, matching every module
    config's `pressure_levels`. Applying it again here is a no-op for files
    where it was already done, same as _stack_depth.
    """
    selected = [ds[var].sel(plev=PRESSURE_LEVELS, method="nearest") for var in variables]
    return torch.from_numpy(xr.concat(selected, dim="variable").to_numpy()).float()


def _stack_depth(ds: xr.Dataset, variables: list[str]) -> torch.Tensor:
    """Same as _stack, but re-selects the pipeline's standard 10 depth levels
    first -- fill_in_missing_ipsl.py already does this .sel(lev=DEPTH_LEVELS,
    method="nearest") for "thetao" (mirrors its own docstring), but some
    interpolation_project_datasets_filled_in/*.nc files (all DAMIP
    experiments, confirmed for hist-GHG/hist-aer/hist-stratO3/1pctCO2 across
    every realization) predate that selection ever being applied and still
    carry the full raw 75-level "lev" dim. Applying it again here is a no-op
    for files where it was already done (nearest-match of the same 10 values
    onto themselves).

    Selects directly on each DataArray and stacks standalone rather than
    writing the reduced array back into `ds` (`ds[var] = ...`) -- that
    write-back re-aligns against every other variable still sharing the
    dataset's full 75-length "lev" index (e.g. lev_bnds), which forces a
    full-dataset materialization and OOMs on cpu_p1 (confirmed: killed at
    ~32GB for a single file). Selecting/stacking outside the dataset avoids
    that entirely.
    """
    selected = [ds[var].sel(lev=DEPTH_LEVELS, method="nearest") for var in variables]
    return torch.from_numpy(xr.concat(selected, dim="variable").to_numpy()).float()


def process_file(
    in_path: str,
    out_path: str,
    scratch: str,
    solar_forcings: torch.Tensor,
    forcing_cache: dict,
) -> None:
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping")
        return

    fname = os.path.basename(in_path)
    m = FNAME_RE.match(fname)
    if not m:
        print(f"skipping unrecognized filename: {fname}")
        return
    exp = m.group("exp")
    if exp not in EXPERIMENTS:
        print(f"skipping unknown experiment {exp} in {fname}")
        return

    ds = xr.open_dataset(in_path)
    n_time = ds.sizes["time"]

    surface = _stack(ds, SURFACE_VARS)
    level = _stack_level(ds, LEVEL_VARS)
    lev = _stack_depth(ds, DEPTH_VARS)
    time = ds.time.values
    ds.close()

    # forcings are experiment-driven (same for every ensemble member of the
    # same exp), so cache per exp to avoid re-reading multi-GB reference_data
    # tensors (e.g. ozone_historical.pt, ~10.7GB) once per ensemble member --
    # same caching pattern as add_forcings_canesm5.py.
    if exp not in forcing_cache:
        forcing_cache.clear()
        forcing_cache[exp] = build_forcings(scratch, exp, 10**6, solar_forcings)
    spatial_forcings_full, non_spatial_forcings_full, ozone_full = forcing_cache[exp]
    spatial_forcings = spatial_forcings_full[:, :n_time]
    non_spatial_forcings = non_spatial_forcings_full[:, :n_time]
    # ozone_bands_tensor returns [time, 66, lat, lon] -- add back the leading
    # size-1 "channel" dim (same [C, time, ...] convention "lev" uses for its
    # single DEPTH_VARS entry) so this matches what netcdf.py's __getitem__
    # expects (data['ozone'][0, line_id]) and what every pre-existing memmap
    # (historical, piControl, ssp*, ...) already has on disk.
    ozone = ozone_full[:n_time].unsqueeze(0)

    out = TensorDict(
        {
            "surface": surface,
            "level": level,
            "lev": lev,
            "spatial_forcings": spatial_forcings,
            "non_spatial_forcings": non_spatial_forcings,
            "ozone": ozone,
        }
    )
    out["time"] = time
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        import shutil

        shutil.rmtree(tmp_path)
    out.memmap(tmp_path, copy_existing=True)
    os.rename(tmp_path, out_path)
    print(f"Written {out_path}")


def main() -> None:
    scratch = os.environ["SCRATCH"]
    in_dir = f"{scratch}/interpolation_project_datasets_filled_in"
    out_dir = os.environ.get("OUTPUT_DIR", f"{scratch}/memmap_filled_in")
    os.makedirs(out_dir, exist_ok=True)

    solar_forcings = load_solar_forcings(scratch)
    forcing_cache: dict = {}

    def exp_key(path: str) -> str:
        m = FNAME_RE.match(os.path.basename(path))
        return m.group("exp") if m else ""

    # Partition files across SLURM array tasks (same pattern as
    # fill_in_missing_ipsl.py) so concurrent jobs never touch the same
    # output file/tmp path -- running the unpartitioned version as two
    # overlapping jobs previously raced on the same .tmp rename.
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    # Optional comma-separated experiment allowlist (e.g. "hist-GHG,hist-aer")
    # for targeted reruns -- e.g. after fixing a bug in one forcing source
    # that only affects specific experiments, without reprocessing every
    # other experiment's unaffected files. Mirrors OUTPUT_DIR's env-var
    # override pattern above. Unset (default) processes every experiment,
    # matching the original behavior.
    exp_filter = os.environ.get("EXP_FILTER")
    in_paths = sorted(glob.glob(f"{in_dir}/*.nc"), key=exp_key)
    if exp_filter:
        allowed = set(exp_filter.split(","))
        in_paths = [p for p in in_paths if exp_key(p) in allowed]
    in_paths = in_paths[task_id::task_count]
    print(f"task {task_id}/{task_count}: {len(in_paths)} files")
    for in_path in in_paths:
        out_name = os.path.basename(in_path).replace(".nc", ".memmap")
        out_path = f"{out_dir}/{out_name}"
        process_file(in_path, out_path, scratch, solar_forcings, forcing_cache)


if __name__ == "__main__":
    main()
