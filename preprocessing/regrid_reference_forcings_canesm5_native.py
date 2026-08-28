"""Downsample reference_data/*'s spatial forcing tensors (baked at 144x144,
IPSL's grid) onto CanESM5's own native 2.8125x2.8125 grid (64x128), so
add_forcings_canesm5.py can build a shape-consistent memmap for a native-grid
CanESM5 run (see preprocessing/interpolation_dataset_canesm5_*.yml's
target_grid).

This is a plain downsample (144x144 -> 64x128, more source pixels per target
pixel), unlike the upsampling regrid that produced the pole row-duplication
bug fixed in interpolation_dataset_canesm5_*.yml -- adaptive average pooling
is well-defined in this direction for any resolution ratio, no artifact
class to worry about. These are large-scale, spatially smooth forcing
fields (GHG concentrations, aerosol loads, ozone bands), so a plain spatial
average is a reasonable approximation; nothing here claims to reproduce
CanESM5's own forcing dataset, just to make the existing IPSL-grid forcings
usable at CanESM5's resolution.

Only regrids the files needed for the experiments passed via --experiments.
Note piControl/1pctCO2/hist-GHG/hist-aer/hist-stratO3/abrupt-4xCO2 all fall
back to reference_data/*_historical.pt (see preprocessing/common/forcings.py)
-- pass "historical" explicitly too when regridding for one of those.
solar_forcings_monthly_interpolated.pt is non-spatial and copied unchanged.

Usage:
    python preprocessing/regrid_reference_forcings_canesm5_native.py --experiments ssp534-over
"""

import argparse
import os
import shutil

import torch
import torch.nn.functional as F

SPATIAL_GASES = ["methane", "carbon", "nitrous"]

# exp -> ghg/aerosol/ozone filename suffix used by preprocessing/common/forcings.py
# ("ssp534-over" GHG files are named "..._ssp534.pt", not "..._ssp534-over.pt" --
# see spatial_ghg_tensor's ssp534-over branch).
GHG_SUFFIX = {"ssp534-over": "ssp534"}


def regrid(t: torch.Tensor, out_hw: tuple) -> torch.Tensor:
    """Adaptive-average-pool the last two (lat, lon) dims of t to out_hw."""
    orig_shape = t.shape
    x = t.reshape(-1, 1, orig_shape[-2], orig_shape[-1]).float()
    x = F.adaptive_avg_pool2d(x, out_hw)
    return x.reshape(*orig_shape[:-2], *out_hw).to(t.dtype)


def regrid_file(src: str, dst: str, out_hw: tuple) -> None:
    if os.path.exists(dst):
        print(f"{dst} already exists, skipping")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    data = torch.load(src)
    regridded = regrid(data, out_hw)
    torch.save(regridded, dst)
    print(f"{src} {tuple(data.shape)} -> {dst} {tuple(regridded.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="e.g. ssp534-over. historical is always included (fallback source).",
    )
    parser.add_argument("--lat", type=int, default=64)
    parser.add_argument("--lon", type=int, default=128)
    parser.add_argument("--out-subdir", default="reference_data_canesm5_native")
    args = parser.parse_args()
    out_hw = (args.lat, args.lon)

    scratch = os.environ["SCRATCH"]
    src_root = f"{scratch}/reference_data"
    dst_root = f"{scratch}/{args.out_subdir}"

    exps = sorted(set(args.experiments))

    solar_src = f"{src_root}/solar_forcings_monthly_interpolated.pt"
    solar_dst = f"{dst_root}/solar_forcings_monthly_interpolated.pt"
    os.makedirs(dst_root, exist_ok=True)
    if not os.path.exists(solar_dst):
        shutil.copy2(solar_src, solar_dst)
        print(f"copied {solar_src} -> {solar_dst} (non-spatial, unchanged)")

    # orography.pt: static (no per-experiment variant), (lat, lon) already --
    # no swapdims patch needed (that's only for spatial_ghg_tensor/
    # ozone_bands_tensor's upstream-bug workaround). Regrid once regardless
    # of --experiments.
    orography_src = f"{src_root}/orography.pt"
    orography_dst = f"{dst_root}/orography.pt"
    if os.path.exists(orography_src):
        regrid_file(orography_src, orography_dst, out_hw)

    # preprocessing/common/forcings.py's spatial_ghg_tensor and
    # ozone_bands_tensor both apply a final `.swapdims(-1, -2)` ("patch for
    # upstream bug") that was a no-op shape-wise on the old square 144x144
    # grid but isn't on a rectangular native grid -- pre-swap our target
    # (lat, lon) here so the *runtime* swapdims lands back on (lat, lon).
    # aerosol_tensor has no such swap, so it regrids straight to (lat, lon).
    swapped_hw = (out_hw[1], out_hw[0])

    for exp in exps:
        ghg_suffix = GHG_SUFFIX.get(exp, exp)
        for gas in SPATIAL_GASES:
            fname = f"{gas}_{ghg_suffix}.pt"
            src = f"{src_root}/processed_spatial_monthly_ghg_forcings/{fname}"
            if os.path.exists(src):
                regrid_file(
                    src, f"{dst_root}/processed_spatial_monthly_ghg_forcings/{fname}", swapped_hw
                )

        fname = f"ozone_{exp}.pt"
        src = f"{src_root}/{fname}"
        if os.path.exists(src):
            regrid_file(src, f"{dst_root}/{fname}", swapped_hw)

        for prefix in ("aerosol",):
            fname = f"{prefix}_{exp}.pt"
            src = f"{src_root}/{fname}"
            if os.path.exists(src):
                regrid_file(src, f"{dst_root}/{fname}", out_hw)


if __name__ == "__main__":
    main()
