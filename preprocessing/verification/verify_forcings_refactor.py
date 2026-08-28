"""One-off check: confirm common/forcings.py's build_forcings still produces
identical output to what's already on disk in memmap_filled_in_canesm5, i.e.
that extracting add_forcings_canesm5.py's functions into common/forcings.py
was behavior-preserving.
"""

import os
import sys
from pathlib import Path

from tensordict import TensorDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.forcings import build_forcings, load_solar_forcings

scratch = os.environ["SCRATCH"]
solar = load_solar_forcings(scratch)

for exp in ["historical", "ssp245"]:
    path = f"{scratch}/memmap_filled_in_canesm5/r1i1p1f1_{exp}_interpolation.memmap"
    if exp == "ssp245":
        # r1 not available for ssp245 in canesm5; use whichever member exists
        import glob

        matches = glob.glob(f"{scratch}/memmap_filled_in_canesm5/*_{exp}_interpolation.memmap")
        path = matches[0]

    td = TensorDict.load_memmap(path)
    n_time = td["surface"].shape[1]

    spatial, non_spatial, ozone_full = build_forcings(scratch, exp, n_time, solar)

    sf_diff = (spatial - td["spatial_forcings"]).abs().max().item()
    nsf_diff = (non_spatial - td["non_spatial_forcings"]).abs().max().item()
    print(
        f"{exp} ({path}): spatial_forcings max_abs_diff={sf_diff}, "
        f"non_spatial_forcings max_abs_diff={nsf_diff}"
    )
