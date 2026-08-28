"""Data-level verification of two historical one-off patches to memmap_filled_in:

1. fix_ozone.py / replace_ozone_layers.ipynb retroactively replaced the ozone
   channels (spatial_forcings[9:19]) of already-generated memmap_filled_in files
   with a corrected 10-band selection (OZONE_BAND_INDICES), taken from the raw
   ozone_{exp}.pt reference tensors. add_forcings.py (the script that would run
   on a *fresh* preprocessing pass) was never updated to produce this directly,
   so there's no way to tell from code alone which files actually got patched --
   fix_ozone.py's scenario list was live-edited and rerun per-experiment. This
   checks the actual data against the expected corrected values.

2. aerosol_patch_job.ipynb manually zero-padded 2 missing months at the end of
   ONE file (r1i1p1f1_ssp534-over) due to a "bug in saved file, need -2 offset"
   in the raw ssp534-over reference tensors. This checks whether every
   ssp534-over member has the same length/padding, or whether only that one
   member got fixed.

Read-only: does not modify any memmap files.
"""

import glob
import os
import re

import torch
from tensordict.tensordict import TensorDict

SCRATCH = os.environ["SCRATCH"]
MEMMAP_DIR = f"{SCRATCH}/memmap_filled_in"
REFERENCE_DIR = f"{SCRATCH}/reference_data"

FNAME_RE = re.compile(r"^(?P<ensemble>[^_]+)_(?P<exp>.+)_interpolation\.memmap$")

# Same 10 ozone-band indices baked into add_forcings_canesm5.py's
# OZONE_BAND_INDICES and fix_ozone.py's (negative-indexed) ozone_indices.
OZONE_BAND_INDICES = [65, 63, 56, 49, 45, 40, 33, 26, 12, 1]

OZONE_TOL = 1e-3  # absolute tolerance for "matches corrected patch"


_RAW_OZONE_CACHE: dict[str, torch.Tensor] = {}


def _load_raw_ozone_for_exp(exp: str) -> torch.Tensor:
    """Cache per-experiment so each multi-GB ozone_{exp}.pt is only read once,
    not once per ensemble member (there are up to ~10 members per experiment).
    """
    if exp in _RAW_OZONE_CACHE:
        return _RAW_OZONE_CACHE[exp]
    _RAW_OZONE_CACHE.clear()  # keep only one experiment's tensor in memory at a time
    if exp in ("piControl", "abrupt-4xCO2"):
        data = torch.load(f"{REFERENCE_DIR}/ozone_historical.pt", map_location="cpu")
    else:
        data = torch.load(f"{REFERENCE_DIR}/ozone_{exp}.pt", map_location="cpu")
    _RAW_OZONE_CACHE[exp] = data
    return data


def expected_ozone_for_exp(exp: str, n_time: int) -> torch.Tensor:
    """Reproduce add_forcings_canesm5.py's experiment-branching ozone load, then
    apply the corrected 10-band selection -- this is the "should be true after
    the patch" reference, independent of whichever historical script state
    actually touched a given file.
    """
    data = _load_raw_ozone_for_exp(exp)
    if exp == "piControl":
        data = torch.tile(data[:12], (250, 1, 1, 1))
    elif exp == "abrupt-4xCO2":
        data = torch.tile(data[:12], (150, 1, 1, 1))
    elif exp == "ssp534-over":
        data = data[(12 * 25) :]
    data = data.swapdims(-1, -2)  # matches the "patch for upstream bug" in add_forcings.py
    data = data[:n_time]
    return data[:, OZONE_BAND_INDICES].swapdims(0, 1)  # -> [10, time, lat, lon]


def _exp_key(path: str) -> str:
    m = FNAME_RE.match(os.path.basename(path))
    return m.group("exp") if m else ""


def check_ozone() -> None:
    print("\n=== Ozone band verification ===")
    # group by experiment so consecutive files reuse the cached raw ozone
    # tensor instead of re-reading multi-GB reference_data files per file
    paths = sorted(glob.glob(f"{MEMMAP_DIR}/*.memmap"), key=_exp_key)
    results = {"match": [], "mismatch": [], "wrong_shape": [], "error": []}

    for path in paths:
        fname = os.path.basename(path)
        m = FNAME_RE.match(fname)
        if not m:
            continue
        exp = m.group("exp")
        try:
            td = TensorDict.load_memmap(path)
            sf = td["spatial_forcings"]
            if sf.shape[0] != 19:
                results["wrong_shape"].append((fname, tuple(sf.shape)))
                continue
            n_time = sf.shape[1]
            actual_ozone = sf[9:19, :n_time]
            expected_ozone = expected_ozone_for_exp(exp, n_time)
            diff = (actual_ozone - expected_ozone).abs()
            max_diff = diff.max().item()
            if max_diff < OZONE_TOL:
                results["match"].append(fname)
            else:
                results["mismatch"].append((fname, max_diff))
        except Exception as e:  # noqa: BLE001
            results["error"].append((fname, str(e)))

    print(f"  matched (correct patch applied): {len(results['match'])}")
    print(f"  mismatched (wrong/unpatched ozone bands): {len(results['mismatch'])}")
    for fname, diff in results["mismatch"]:
        print(f"    MISMATCH {fname}: max_abs_diff={diff:.4f}")
    print(f"  wrong shape (never patched, still 15 channels?): {len(results['wrong_shape'])}")
    for fname, shape in results["wrong_shape"]:
        print(f"    WRONG_SHAPE {fname}: spatial_forcings shape={shape}")
    print(f"  errors: {len(results['error'])}")
    for fname, err in results["error"]:
        print(f"    ERROR {fname}: {err}")


def check_ssp534_over() -> None:
    print("\n=== ssp534-over length / end-padding verification ===")
    paths = sorted(glob.glob(f"{MEMMAP_DIR}/*ssp534-over*.memmap"))
    if not paths:
        print("  no ssp534-over files found")
        return

    raw_ozone = torch.load(f"{REFERENCE_DIR}/ozone_ssp534-over.pt", map_location="cpu")[(12 * 25) :]
    print(f"  raw ozone_ssp534-over.pt sliced length (12*25 onward): {raw_ozone.shape[0]} months")

    for path in paths:
        fname = os.path.basename(path)
        td = TensorDict.load_memmap(path)
        sf = td["spatial_forcings"]
        nsf = td["non_spatial_forcings"]
        n_time = sf.shape[1]
        # aerosol_patch_job.ipynb zero-padded exactly 2 months at the end,
        # across ALL FOUR forcing types (GHG, aerosol, ozone, solar/SSI), not
        # just aerosols -- check whether each is exactly zero (patched) or not
        # (unpatched / different) for the last 2 timesteps.
        last_two_ghg = sf[0:3, -2:]
        last_two_aerosol = sf[3:9, -2:]
        last_two_ozone = sf[9:19, -2:]
        last_two_ssi = nsf[:, -2:]
        ghg_is_zero = torch.allclose(last_two_ghg, torch.zeros_like(last_two_ghg))
        aerosol_is_zero = torch.allclose(last_two_aerosol, torch.zeros_like(last_two_aerosol))
        ozone_is_zero = torch.allclose(last_two_ozone, torch.zeros_like(last_two_ozone))
        ssi_is_zero = torch.allclose(last_two_ssi, torch.zeros_like(last_two_ssi))
        print(
            f"  {fname}: n_time={n_time}, raw_ozone_len={raw_ozone.shape[0]}, "
            f"length_matches_raw={n_time == raw_ozone.shape[0] + 2}, "
            f"last_2_months_zero_padded=[ghg={ghg_is_zero}, aerosol={aerosol_is_zero}, "
            f"ozone={ozone_is_zero}, ssi={ssi_is_zero}]"
        )


if __name__ == "__main__":
    check_ozone()
    check_ssp534_over()
