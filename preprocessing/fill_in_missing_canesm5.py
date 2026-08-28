"""canesm5 equivalent of fill_in_missing_data.py.

Reads raw canesm5 memmaps ($SCRATCH/interpolation_canesm5_datasets), matches
ipsl's channel layout (17 pressure levels via canesm5_levels.PLEV_KEEP, 10
ocean depth levels via canesm5_levels.conservative_vertical_regrid, 8 surface
variables in ipsl's order/count), and applies common.fill_in_missing's
permanent-mask + linear-gap-fill algorithm to the 5 level variables (hus, ta,
ua, va, zg) plus thetao (lev).

Writes to $SCRATCH/interpolation_canesm5_datasets_filled_in.
"""

import glob
import os

import numpy as np
import torch
from canesm5_levels import (
    CANESM5_LEV_BNDS,
    IPSL_DEPTH_LEVELS,
    PLEV_KEEP,
    conservative_vertical_regrid,
    guess_bounds,
)
from common.fill_in_missing import mask_and_fill
from tensordict.tensordict import TensorDict

LEVEL_VARS = ["hus", "ta", "ua", "va", "zg"]

# on-disk surface channel order (fingerprinted from real data, see plan) vs.
# ipsl's 8-variable order (memmap_filled_in surface_variables).
CANESM5_SURFACE_ORDER = [
    "tas",
    "psl",
    "pr",
    "uas",
    "vas",
    "ps",
    "net_flux",
    "evspsbl",
    "huss",
]
IPSL_SURFACE_ORDER = ["tas", "psl", "pr", "uas", "vas", "ps", "net_flux", "evspsbl"]
SURFACE_KEEP_IDX = [CANESM5_SURFACE_ORDER.index(v) for v in IPSL_SURFACE_ORDER]


def process_file(in_path: str, out_path: str) -> None:
    if os.path.exists(out_path):
        print(f"{out_path} already exists, skipping")
        return

    td = TensorDict.load_memmap(in_path)

    surface = td["surface"][SURFACE_KEEP_IDX]

    level = td["level"][:, :, PLEV_KEEP]
    filled_level = torch.stack([mask_and_fill(level[i]) for i in range(level.shape[0])], dim=0)

    lev = td["lev"][0]  # [time, depth, lat, lon], single var (thetao)
    depth_first = lev.permute(1, 0, 2, 3)  # [depth, time, lat, lon]
    tgt_bnds = guess_bounds(np.array(IPSL_DEPTH_LEVELS))
    src_bnds = np.array(CANESM5_LEV_BNDS)
    regridded = conservative_vertical_regrid(depth_first, src_bnds, tgt_bnds)
    regridded = regridded.permute(1, 0, 2, 3)  # [time, depth, lat, lon]
    filled_lev = mask_and_fill(regridded).unsqueeze(0)  # [1, time, depth, lat, lon]

    out = TensorDict(
        {
            "surface": surface,
            "level": filled_level,
            "lev": filled_lev,
        }
    )
    out["time"] = td["time"]
    out.memmap(out_path)
    print(f"Written {out_path}")


def main() -> None:
    scratch = os.environ["SCRATCH"]
    in_dir = f"{scratch}/interpolation_canesm5_datasets"
    out_dir = f"{scratch}/interpolation_canesm5_datasets_filled_in"
    os.makedirs(out_dir, exist_ok=True)

    # Process a single named file -- e.g. to fill in just ssp534-over as soon
    # as its download lands, without waiting on / re-touching whatever other
    # experiments' downloads are still in flight in the same in_dir. Same
    # pattern as add_forcings_canesm5.py's CANESM5_ONLY_FNAME.
    only_fname = os.environ.get("CANESM5_ONLY_FNAME")
    if only_fname:
        in_paths = [f"{in_dir}/{only_fname}"]
    else:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        in_paths = sorted(glob.glob(f"{in_dir}/*.memmap"))[task_id::task_count]
    print(f"processing {len(in_paths)} file(s)")
    for in_path in in_paths:
        out_path = f"{out_dir}/{os.path.basename(in_path)}"
        process_file(in_path, out_path)


if __name__ == "__main__":
    main()
