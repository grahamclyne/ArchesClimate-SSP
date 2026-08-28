"""canesm5 equivalent of add_forcings.py.

Forcings (GHG concentrations, aerosol loads, solar, ozone) are driven purely
by experiment/scenario, not by the source ESM -- add_forcings.py loads them
from reference_data/*.pt files keyed on experiment name. This script reuses
the exact same sources and branching logic, but assembles them directly into
each canesm5 TensorDict/memmap (from interpolation_canesm5_datasets_filled_in)
instead of NetCDF, matching memmap_filled_in's spatial_forcings (19 channels:
3 GHGs + 6 aerosols + 10 ozone bands) / non_spatial_forcings (6 SSI bands)
layout.

Writes to $SCRATCH/memmap_filled_in_canesm5.
"""

import glob
import os
import re

import torch
from common.forcings import EXPERIMENTS, build_forcings, load_solar_forcings
from tensordict.tensordict import TensorDict

FNAME_RE = re.compile(r"^(?P<ensemble>[^_]+)_(?P<exp>.+)_interpolation\.memmap$")


def process_file(
    in_path: str,
    out_path: str,
    scratch: str,
    solar_forcings: torch.Tensor,
    forcing_cache: dict,
    reference_data_dir: str = "reference_data",
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

    td = TensorDict.load_memmap(in_path)
    n_time = td["surface"].shape[1]

    # forcings are experiment-driven (same for every ensemble member of the
    # same exp), so cache per exp to avoid re-reading multi-GB reference_data
    # tensors (e.g. ozone_historical.pt, ~10.7GB) once per ensemble member.
    # Cache the untruncated (max-length) tensors -- keying on (exp, n_time)
    # meant any ensemble member with a different timestep count than the
    # first-seen one for that exp missed the cache and paid the full
    # multi-GB reload again. build_forcings' internal `data[:n_time]` slices
    # are a no-op when n_time is larger than the real data, so passing a
    # large sentinel here just yields the full-length tensors once per exp;
    # each file then slices down to its own n_time below.
    if exp not in forcing_cache:
        forcing_cache.clear()  # only keep one exp's forcings in memory at a time
        forcing_cache[exp] = build_forcings(scratch, exp, 10**6, solar_forcings, reference_data_dir)
    spatial_forcings_full, non_spatial_forcings_full, ozone_full = forcing_cache[exp]
    spatial_forcings = spatial_forcings_full[:, :n_time]
    non_spatial_forcings = non_spatial_forcings_full[:, :n_time]
    ozone = ozone_full[:n_time]

    out = TensorDict(
        {
            "surface": td["surface"],
            "level": td["level"],
            "lev": td["lev"],
            "spatial_forcings": spatial_forcings,
            "non_spatial_forcings": non_spatial_forcings,
            "ozone": ozone,
        }
    )
    out["time"] = td["time"]
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        import shutil

        shutil.rmtree(tmp_path)
    out.memmap(tmp_path, copy_existing=True)
    os.rename(tmp_path, out_path)
    print(f"Written {out_path}")


def main() -> None:
    scratch = os.environ["SCRATCH"]
    # Overridable so a native-grid run (e.g. CanESM5's own 64x128, see
    # preprocessing/interpolation_dataset_canesm5_*.yml's target_grid) can
    # read/write to dedicated directories instead of memmap_filled_in_canesm5,
    # which already holds 144x144 files regridded onto IPSL's grid -- writing
    # native-resolution output there would silently mix two grid shapes under
    # one directory. reference_data_dir similarly needs to point at a
    # native-resolution copy of the GHG/aerosol/ozone forcing tensors (see
    # preprocessing/regrid_reference_forcings_canesm5_native.py), since
    # reference_data/*'s spatial forcings are baked at 144x144 and can't be
    # concatenated with 64x128 surface/level/lev data as-is.
    in_dir = os.environ.get("CANESM5_IN_DIR", f"{scratch}/interpolation_canesm5_datasets_filled_in")
    out_dir = os.environ.get("CANESM5_OUT_DIR", f"{scratch}/memmap_filled_in_canesm5")
    reference_data_dir = os.environ.get("CANESM5_REFERENCE_DATA_DIR", "reference_data")
    os.makedirs(out_dir, exist_ok=True)

    solar_forcings = load_solar_forcings(scratch, reference_data_dir)
    forcing_cache: dict = {}

    # group by experiment so consecutive files reuse forcing_cache instead of
    # re-reading multi-GB reference_data tensors for every ensemble member.
    def exp_key(path: str) -> str:
        m = FNAME_RE.match(os.path.basename(path))
        return m.group("exp") if m else ""

    # Process a single named file (one job per file, for max parallelism /
    # isolating a problem file) instead of the task-partitioned sweep below.
    only_fname = os.environ.get("CANESM5_ONLY_FNAME")
    if only_fname:
        in_paths = [f"{in_dir}/{only_fname}"]
    else:
        # Partition files across SLURM array tasks (same pattern as
        # add_forcings.py) so concurrent tasks never touch the same output
        # file/tmp path.
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        in_paths = sorted(glob.glob(f"{in_dir}/*.memmap"), key=exp_key)[task_id::task_count]
        print(f"task {task_id}/{task_count}: {len(in_paths)} files")

    for in_path in in_paths:
        out_path = f"{out_dir}/{os.path.basename(in_path)}"
        try:
            process_file(
                in_path, out_path, scratch, solar_forcings, forcing_cache, reference_data_dir
            )
        except Exception:
            # Don't let one bad input file abort the rest of this task's list.
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
