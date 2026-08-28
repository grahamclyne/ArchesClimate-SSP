"""IPSL equivalent of fill_in_missing_canesm5.py -- modernized replacement for
fill_in_missing_data.py, which read/wrote the same NetCDF I/O contract but
used xarray's per-file interpolate_na instead of common.fill_in_missing's
vectorized algorithm, and didn't touch thetao at all.

Reads raw NetCDF from $SCRATCH/interpolation_project_datasets (written by
prepare_interpol_dataset.py / common/prepare_interpol_dataset.py, shared with
every interpolation_dataset_*.yml recipe -- net_flux is already derived at
that stage, nothing forcing-specific happens here), applies
common.fill_in_missing's permanent-mask + linear-gap-fill to the 5 pressure-
level variables (hus, ta, ua, va, zg) and to thetao (ocean depth levels; the
only IPSL variable that needs it besides the level vars -- surface variables
are defined everywhere on the globe and never have gaps).

Explicitly selects the pipeline's standard 17 pressure levels / 10 depth
levels via .sel(..., method="nearest") so channel count/order matches
memmap_filled_in regardless of exactly what Stage 1 returns -- mirrors the
commented-out .sel(...) calls in legacy/convert_xarray_to_memmap.py, just not
commented out.

NOT YET VERIFIED against real IPSL output: common/fill_in_missing.py's
algorithm has so far only been diffed against canesm5 data (see its
docstring) -- run this once real Stage 1 NetCDFs exist and spot-check a
resulting file before trusting it for training.

Writes to $SCRATCH/interpolation_project_datasets_filled_in.
"""

import glob
import os

import torch
import xarray as xr
from common.fill_in_missing import mask_and_fill

LEVEL_VARS = ["hus", "ta", "ua", "va", "zg"]

PRESSURE_LEVELS = [
    100000.0,
    92500.0,
    85000.0,
    70000.0,
    60000.0,
    50000.0,
    40000.0,
    30000.0,
    25000.0,
    20000.0,
    15000.0,
    10000.0,
    7000.0,
    5000.0,
    3000.0,
    2000.0,
    1000.0,
]
DEPTH_LEVELS = [
    0.50576,
    3.8562799,
    8.092519,
    13.991038,
    22.757616,
    35.740204,
    53.850636,
    77.61116,
    108.03028,
    147.40625,
]


def _fill_var(da: xr.DataArray) -> xr.DataArray:
    """da: [time, level, lat, lon] DataArray -> gap-filled DataArray, same coords."""
    arr = torch.from_numpy(da.values).float()
    filled = mask_and_fill(arr)
    return xr.DataArray(filled.numpy(), dims=da.dims, coords=da.coords, attrs=da.attrs)


def process_file(in_path: str, out_path: str) -> None:
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping")
        return

    ds = xr.open_dataset(in_path)

    for var in LEVEL_VARS:
        ds[var] = ds[var].sel(plev=PRESSURE_LEVELS, method="nearest")
        ds[var] = _fill_var(ds[var])

    ds["thetao"] = ds["thetao"].sel(lev=DEPTH_LEVELS, method="nearest")
    ds["thetao"] = _fill_var(ds["thetao"])

    ds.to_netcdf(out_path)
    ds.close()
    print(f"Written {out_path}")


def main() -> None:
    scratch = os.environ["SCRATCH"]
    in_dir = f"{scratch}/interpolation_project_datasets"
    out_dir = f"{scratch}/interpolation_project_datasets_filled_in"
    os.makedirs(out_dir, exist_ok=True)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    in_paths = sorted(glob.glob(f"{in_dir}/*.nc"))[task_id::task_count]
    print(f"task {task_id}/{task_count}: {len(in_paths)} files")
    for in_path in in_paths:
        out_path = f"{out_dir}/{os.path.basename(in_path)}"
        process_file(in_path, out_path)


if __name__ == "__main__":
    main()
