"""One-off reconstruction of reference_data/ozone_historical.pt from an
already-built historical memmap's "ozone" field, working around the loss of
the raw climoz source files this repo's ozone forcing pipeline was originally
built from (see forcings_preprocessing.py's docstring).

add_forcings.py (Stage 3) always writes the full, unsubsetted 66-band ozone
tensor into every memmap's "ozone" key -- not just the 10-band subset that
ends up in spatial_forcings (see common/forcings.py's ozone_bands_tensor /
ozone_subset_and_full). For the "historical" experiment specifically, that
tensor *is* torch.load(reference_data/ozone_historical.pt): ozone_bands_tensor's
"historical" branch does nothing but load that exact file, then apply its
unconditional trailing .swapdims(-1, -2) before returning. So the memmap
already holds this file's content, just post-swap and already-materialized --
this script reads it back out and undoes that one swap, landing back in the
pre-swap orientation ozone_bands_tensor (and fix_ozone.py, which loads this
same file directly with its own separate transform) expect to find on disk.

Only reconstructs the first 1980 months (1850-2014, HIST_SINGLE_FORCING_YEARS
in common/forcings.py) -- exactly what every current consumer needs:
piControl/abrupt-4xCO2/1pctCO2 only ever use the first 12 months (tiled
climatology); hist-GHG/hist-aer/hist-stratO3 need the full 165 years. This
would NOT be sufficient if some future experiment needed ozone beyond 2014.

Ozone forcing is a boundary condition and should be identical across
ensemble members of the same experiment -- spot-checked bit-identical between
r1i1p1f1 and r2i1p1f1 (max abs diff 0.0 across 30 random samples) before
writing this script, so any historical realization's memmap works as source.
"""

import os
from pathlib import Path

import numpy as np
import torch

SCRATCH = os.environ["SCRATCH"]
SOURCE_REALIZATION = os.environ.get("OZONE_SOURCE_REALIZATION", "r1i1p1f1")
MEMMAP_DIR = Path(f"{SCRATCH}/memmap_filled_in_full_ozone")
OUT_PATH = Path(f"{SCRATCH}/reference_data/ozone_historical.pt")

N_TIME, N_BANDS, N_LAT, N_LON = 1980, 66, 144, 144


def main() -> None:
    if OUT_PATH.exists():
        raise FileExistsError(f"{OUT_PATH} already exists -- refusing to overwrite")

    src_dir = MEMMAP_DIR / f"{SOURCE_REALIZATION}_historical_interpolation.memmap"
    ozone_path = src_dir / "ozone.memmap"

    # ozone.memmap has no header (raw contiguous binary dump, confirmed by
    # exact file-size match) -- read it with plain numpy rather than
    # tensordict, so this doesn't depend on the on-disk TensorDict layout.
    expected_bytes = N_TIME * N_BANDS * N_LAT * N_LON * 4  # leading dim is 1
    actual_bytes = ozone_path.stat().st_size
    assert actual_bytes == expected_bytes, (
        f"{ozone_path} is {actual_bytes} bytes, expected {expected_bytes} for "
        f"shape (1, {N_TIME}, {N_BANDS}, {N_LAT}, {N_LON}) float32 -- meta.json's "
        "recorded shape may have changed; re-check before trusting this script."
    )

    raw = np.memmap(ozone_path, dtype="float32", mode="r", shape=(1, N_TIME, N_BANDS, N_LAT, N_LON))
    # Drop the leading singleton "channel" dim (ozone is a single named
    # variable, same TensorDict convention as "lev"/thetao) -> (time, 66, lat, lon).
    post_swap = torch.from_numpy(raw[0].copy())
    pre_swap = post_swap.swapdims(-1, -2).contiguous()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pre_swap, OUT_PATH)
    print(
        f"Wrote {OUT_PATH}: shape={tuple(pre_swap.shape)}, dtype={pre_swap.dtype}, "
        f"bytes={OUT_PATH.stat().st_size} (source: {SOURCE_REALIZATION})"
    )


if __name__ == "__main__":
    main()
