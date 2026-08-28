import faulthandler
import os
from typing import Any

from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic

# The hist-GHG/hist-aer/hist-stratO3/1pctCO2 esmvaltool runs segfaulted
# (exit -11) with no Python traceback -- MaxRSS stayed well under the job's
# requested memory each time, so it wasn't OOM. faulthandler.enable() below
# caught the actual crash: 16 dask worker threads simultaneously inside
# xarray/backends/netCDF4_.py's __setitem__ (dask's default threaded
# scheduler parallelizing to_netcdf()'s chunked write across threads that
# all touch the same open netCDF4/HDF5 file handle) -- a classic segfault
# when the underlying HDF5 build isn't compiled thread-safe, which is common
# on conda-forge netCDF4 stacks like this esmvaltool-env1 env. See
# merge_cubes_and_compute_net_flux's .compute() call below for the fix.
faulthandler.enable()

from common.prepare_interpol_dataset import (
    group_by_exp_ensemble,
    merge_cubes_and_compute_net_flux,
)


def run_my_diagnostic(cfg: Any) -> None:
    scratch = os.environ["SCRATCH"]
    cfg["work_dir"]
    group_metadata(cfg["input_data"].values(), "ensemble")

    # to_netcdf() below doesn't create its parent directory -- this used to
    # exist from an earlier run and silently kept working; once it got
    # cleaned up (see README.md's Stage 1 notes), every fresh recipe run
    # failed at the final write with a misleading PermissionError instead of
    # a clear "no such directory" (netCDF4/HDF5 reports a missing parent as
    # a permission error, not FileNotFoundError).
    os.makedirs(f"{scratch}/interpolation_project_datasets", exist_ok=True)

    grouped = group_by_exp_ensemble(cfg["input_data"].values())
    for (exp, ensemble), entries in grouped.items():
        print(exp, ensemble)
        for entry in entries:
            print(entry["filename"])
        dataset = merge_cubes_and_compute_net_flux(entries)
        # Materialize fully before writing: dataset.to_netcdf() on a
        # dask-backed Dataset writes chunks from multiple threads into the
        # same open netCDF4/HDF5 file handle concurrently, which segfaults
        # on this env's (non-thread-safe) HDF5 build -- see faulthandler
        # note above. .compute() forces everything into plain in-memory
        # numpy arrays first, so the actual to_netcdf() write happens
        # single-threaded, on this thread, touching the file handle alone.
        dataset = dataset.compute()
        dataset.to_netcdf(
            f"{scratch}/interpolation_project_datasets/{ensemble}_{exp}_interpolation.nc"
        )
        print("Merged NetCDF written to: scratch")


if __name__ == "__main__":
    with run_diagnostic() as config:
        run_my_diagnostic(config)
