"""One-off fix for memmap_filled_in/*hist-GHG*|*hist-aer*|*hist-stratO3*|*1pctCO2*.

Two independent bugs in these files, both from add_forcings.py's handling of
gap-filled DAMIP/1pctCO2 NetCDF, verified by direct shape inspection (not
re-running Stage 1/2):

1. Level count: fill_in_missing_ipsl.py's .sel(..., method="nearest")
   level-reduction never took effect -- state was written with the source
   model's raw native levels (19 pressure levels, 75 ocean depth levels)
   instead of the pipeline's standard 17 pressure levels / 10 depth levels
   that cmip_stats_new_ozone.pt's mean/std tensors are shaped for. The native
   levels are a fixed model grid, identical across every experiment/ensemble
   member (verified against interpolation_project_datasets/*.nc coord
   metadata for 1pctCO2, hist-GHG, hist-aer, hist-stratO3): the target 17
   pressure levels are exactly the first 17 (of 19) native levels in native
   order, and the target 10 depth levels are exactly every 3rd of the 75
   native depth levels (indices 0,3,6,...,27). fill_in_missing_ipsl.py's
   gap-fill ran over all native levels regardless, so the data at these
   positions is already correctly gap-filled -- this just re-selects the
   slice .sel() should have produced, positionally.

2. Time length (hist-GHG/hist-aer/hist-stratO3 only): process_file's
   n_time = ds.sizes["time"] uses the raw DAMIP NetCDF's full native range
   (1850-2020, 2052 months) for surface/level/lev, but common/forcings.py's
   HIST_SINGLE_FORCING_YEARS=165 truncation caps spatial_forcings/
   non_spatial_forcings/ozone to 1850-2014 (1980 months) -- the only range
   reference_data/*_historical.pt covers. slicing spatial_forcings_full[:,
   :n_time] with n_time=2052 on an already-1980-long tensor is a silent
   no-op (Python slicing doesn't bounds-check), so state and forcings end up
   with different time lengths in the same file. 1pctCO2 doesn't have this
   problem (self-consistently 1800 months throughout, PCT_CO2_YEARS). Fixed
   by truncating surface/level/lev to the forcings' 1980 months too --
   discards 2015-2020, which the forcings reference data can't cover anyway.

Overwrites each affected memmap in place (tmp dir + atomic rename).
"""

import glob
import os
import shutil

from tensordict.tensordict import TensorDict

PLEV_IDX = list(range(17))
LEV_IDX = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
HIST_SINGLE_FORCING_MONTHS = 165 * 12

AFFECTED_EXPS = ["hist-GHG", "hist-aer", "hist-stratO3", "1pctCO2"]
TIME_TRUNCATED_EXPS = {"hist-GHG", "hist-aer", "hist-stratO3"}


def fix_file(path: str) -> None:
    td = TensorDict.load_memmap(path)
    exp = next(e for e in AFFECTED_EXPS if f"_{e}_" in os.path.basename(path))
    n_time = HIST_SINGLE_FORCING_MONTHS if exp in TIME_TRUNCATED_EXPS else td["surface"].shape[1]

    already_correct = (
        td["level"].shape[2] == 17 and td["lev"].shape[2] == 10 and td["surface"].shape[1] == n_time
    )
    if already_correct:
        print(f"{path} already correct, skipping")
        return

    fixed = TensorDict(
        {
            "surface": td["surface"][:, :n_time].clone(),
            "level": td["level"][:, :n_time][:, :, PLEV_IDX].clone(),
            "lev": td["lev"][:, :n_time][:, :, LEV_IDX].clone(),
            "spatial_forcings": td["spatial_forcings"].clone(),
            "non_spatial_forcings": td["non_spatial_forcings"].clone(),
            "ozone": td["ozone"].clone(),
        }
    )
    fixed["time"] = td["time"][:n_time].copy()  # numpy.ndarray, not a tensor

    tmp_path = path + ".fixtmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    fixed.memmap(tmp_path, copy_existing=True)
    del td, fixed
    shutil.rmtree(path)
    os.rename(tmp_path, path)
    print(f"Fixed {path}")


def main() -> None:
    scratch = os.environ["SCRATCH"]
    in_dir = f"{scratch}/memmap_filled_in"

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    paths = sorted(
        p for exp in AFFECTED_EXPS for p in glob.glob(f"{in_dir}/*_{exp}_interpolation.memmap")
    )[task_id::task_count]
    print(f"task {task_id}/{task_count}: {len(paths)} files")
    for p in paths:
        fix_file(p)


if __name__ == "__main__":
    main()
