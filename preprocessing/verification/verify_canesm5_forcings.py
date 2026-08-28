"""Verify memmap_filled_in_canesm5 against memmap_filled_in (ipsl).

Forcings should match closely (they're driven by experiment/scenario, not
model -- see add_forcings_canesm5.py). Masking patterns won't be pixel
identical (different model orographies/ocean grids), so those are reported
as similarity metrics rather than asserted equal.

Usage: python verify_canesm5_forcings.py
"""

import glob
import os
import re

import torch
from tensordict.tensordict import TensorDict

FNAME_RE = re.compile(r"^(?P<ensemble>[^_]+)_(?P<exp>.+)_interpolation\.memmap$")
FORCING_TOL = 1e-4  # relative tolerance for forcings agreement


def index_by_exp(dir_path: str) -> dict:
    out = {}
    for path in sorted(glob.glob(f"{dir_path}/*.memmap")):
        m = FNAME_RE.match(os.path.basename(path))
        if not m:
            continue
        out.setdefault(m.group("exp"), []).append(path)
    return out


def mask_overlap(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a, b: [time, level, lat, lon] tensors (possibly differing time length)."""
    n = min(a.shape[0], b.shape[0])
    a_mask = torch.isnan(a[:n])
    b_mask = torch.isnan(b[:n])
    overlap = (a_mask == b_mask).float().mean().item()
    return {
        "canesm5_masked_frac": a_mask.float().mean().item(),
        "ipsl_masked_frac": b_mask.float().mean().item(),
        "mask_agreement": overlap,
    }


def compare_forcings(canesm5_td: TensorDict, ipsl_td: TensorDict) -> dict:
    n = min(canesm5_td["spatial_forcings"].shape[1], ipsl_td["spatial_forcings"].shape[1])
    csf = canesm5_td["spatial_forcings"][:, :n]
    isf = ipsl_td["spatial_forcings"][:, :n]
    sf_diff = (csf - isf).abs()

    cnsf = canesm5_td["non_spatial_forcings"][:, :n]
    insf = ipsl_td["non_spatial_forcings"][:, :n]
    nsf_diff = (cnsf - insf).abs()

    return {
        "spatial_forcings_max_abs_diff": sf_diff.max().item(),
        "spatial_forcings_mean_abs_diff": sf_diff.mean().item(),
        "non_spatial_forcings_max_abs_diff": nsf_diff.max().item(),
        "non_spatial_forcings_mean_abs_diff": nsf_diff.mean().item(),
        "n_timesteps_compared": n,
    }


def verify_experiment(exp: str, canesm5_path: str, ipsl_path: str) -> bool:
    print(f"\n=== {exp} ===")
    print(f"  canesm5: {canesm5_path}")
    print(f"  ipsl:    {ipsl_path}")

    canesm5_td = TensorDict.load_memmap(canesm5_path)
    ipsl_td = TensorDict.load_memmap(ipsl_path)

    forcing_report = compare_forcings(canesm5_td, ipsl_td)
    forcings_ok = (
        forcing_report["spatial_forcings_max_abs_diff"] < FORCING_TOL
        and forcing_report["non_spatial_forcings_max_abs_diff"] < FORCING_TOL
    )
    print(f"  forcings: {'PASS' if forcings_ok else 'FAIL'} -- {forcing_report}")

    level_names = ["hus", "ta", "ua", "va", "zg"]
    canesm5_level = canesm5_td["level"]
    ipsl_level = ipsl_td["level"]
    for i, name in enumerate(level_names):
        report = mask_overlap(canesm5_level[i], ipsl_level[i])
        print(f"  level[{name}] mask: {report}")

    lev_report = mask_overlap(canesm5_td["lev"][0], ipsl_td["lev"][0])
    print(f"  lev[thetao] mask: {lev_report}")

    return forcings_ok


def main() -> None:
    scratch = os.environ["SCRATCH"]
    canesm5_dir = f"{scratch}/memmap_filled_in_canesm5"
    ipsl_dir = f"{scratch}/memmap_filled_in"

    canesm5_by_exp = index_by_exp(canesm5_dir)
    ipsl_by_exp = index_by_exp(ipsl_dir)

    shared_exps = sorted(set(canesm5_by_exp) & set(ipsl_by_exp))
    if not shared_exps:
        print("No shared experiments found between the two datasets.")
        return

    all_pass = True
    for exp in shared_exps:
        canesm5_path = canesm5_by_exp[exp][0]
        ipsl_path = ipsl_by_exp[exp][0]
        ok = verify_experiment(exp, canesm5_path, ipsl_path)
        all_pass = all_pass and ok

    print(f"\n{'ALL EXPERIMENTS PASSED' if all_pass else 'SOME EXPERIMENTS FAILED'}")


if __name__ == "__main__":
    main()
