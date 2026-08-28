"""Vertical-grid constants and conservative regridding for the canesm5 pipeline.

canesm5's atmospheric pressure levels are a strict superset of ipsl's (see
PLEV_KEEP). canesm5's ocean depth levels use a different native grid than
ipsl's NEMO-derived grid, so thetao needs a conservative (bounds-overlap
weighted) vertical remap onto ipsl's depth levels instead of a simple index
subset.
"""

import numpy as np
import torch

# canesm5's 19 plev values (Pa) are ipsl's 17 plev values plus two extra
# levels at the top of the atmosphere (500 Pa, 100 Pa). Confirmed against
# /scratch/gclyne/interpolation_dataset_canesm5_ssp119_20260704_110227/preproc/
# regrid_and_combine/ta/CMIP6_CanESM5_Amon_ssp119_r5i1p1f1_ta_gn_2015-2100.nc
PLEV_KEEP = slice(0, 17)

# ipsl's 10 target ocean depth levels (m), from any current module config's
# depth_levels (e.g. configs/module/deterministic_damip_pf4_fixed.yaml).
IPSL_DEPTH_LEVELS = [
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

# canesm5's native thetao layer bounds (m), lev_bnds from a real CanESM5
# Omon file (same on every ensemble/experiment for the "gn" grid).
CANESM5_LEV_BNDS = [
    [0.0, 6.194242477416992],
    [6.194242477416992, 12.839149475097656],
    [12.839149475097656, 20.044540405273438],
    [20.044540405273438, 27.946292877197266],
    [27.946292877197266, 36.71212387084961],
    [36.71212387084961, 46.548465728759766],
    [46.548465728759766, 57.70850372314453],
    [57.70850372314453, 70.50135803222656],
    [70.50135803222656, 85.30240631103516],
    [85.30240631103516, 102.5643081665039],
    [102.5643081665039, 122.8283920288086],
    [122.8283920288086, 146.7351531982422],
    [146.7351531982422, 175.0327606201172],
    [175.0327606201172, 208.58119201660156],
    [208.58119201660156, 248.34959411621094],
    [248.34959411621094, 295.4039001464844],
    [295.4039001464844, 350.8814697265625],
    [350.8814697265625, 415.9508972167969],
    [415.9508972167969, 491.7565002441406],
    [491.7565002441406, 579.35009765625],
    [579.35009765625, 679.6165771484375],
    [679.6165771484375, 793.201904296875],
    [793.201904296875, 920.4561767578125],
    [920.4561767578125, 1061.4005126953125],
    [1061.4005126953125, 1215.72509765625],
    [1215.72509765625, 1382.8184814453125],
    [1382.8184814453125, 1561.822021484375],
    [1561.822021484375, 1751.7010498046875],
    [1751.7010498046875, 1951.318603515625],
    [1951.318603515625, 2159.505859375],
    [2159.505859375, 2375.119384765625],
    [2375.119384765625, 2597.08203125],
    [2597.08203125, 2824.411865234375],
    [2824.411865234375, 3056.234619140625],
    [3056.234619140625, 3291.7880859375],
    [3291.7880859375, 3530.4189453125],
    [3530.4189453125, 3771.573974609375],
    [3771.573974609375, 4014.78955078125],
    [4014.78955078125, 4259.68115234375],
    [4259.68115234375, 4505.9326171875],
    [4505.9326171875, 4753.283203125],
    [4753.283203125, 5001.52197265625],
    [5001.52197265625, 5250.47607421875],
    [5250.47607421875, 5500.00634765625],
    [5500.00634765625, 5750.0],
]


def guess_bounds(centers: np.ndarray) -> np.ndarray:
    """Derive (n, 2) cell bounds from monotonically increasing cell centers.

    Bounds are midpoints between consecutive centers, with the top edge at 0
    and the bottom edge extrapolated using the last interior half-width --
    the standard convention (e.g. iris' Coord.guess_bounds()) for grids
    whose real bounds aren't available.
    """
    centers = np.asarray(centers, dtype=np.float64)
    midpoints = (centers[:-1] + centers[1:]) / 2
    lower = np.concatenate([[0.0], midpoints])
    upper = np.concatenate([midpoints, [centers[-1] + (centers[-1] - midpoints[-1])]])
    return np.stack([lower, upper], axis=1)


def conservative_vertical_regrid(
    values: torch.Tensor, src_bnds: np.ndarray, tgt_bnds: np.ndarray
) -> torch.Tensor:
    """Conservative (bounds-overlap weighted) remap along the level axis.

    Args:
        values: tensor shaped [levels, ...] (levels is the first axis).
        src_bnds: (n_src, 2) array of source layer bounds, matching values.shape[0].
        tgt_bnds: (n_tgt, 2) array of target layer bounds.

    Returns:
        tensor shaped [n_tgt, ...], each output layer is the overlap-weighted
        average of the source layers it intersects.
    """
    n_src = values.shape[0]
    src_bnds = np.asarray(src_bnds, dtype=np.float64)[:n_src]
    tgt_bnds = np.asarray(tgt_bnds, dtype=np.float64)

    weights = np.zeros((len(tgt_bnds), n_src), dtype=np.float64)
    for t, (t_lo, t_hi) in enumerate(tgt_bnds):
        overlap_lo = np.maximum(src_bnds[:, 0], t_lo)
        overlap_hi = np.minimum(src_bnds[:, 1], t_hi)
        overlap = np.clip(overlap_hi - overlap_lo, a_min=0.0, a_max=None)
        weights[t] = overlap
    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = np.nan
    weights = weights / row_sums

    w = torch.from_numpy(weights).to(dtype=values.dtype)
    flat = values.reshape(n_src, -1)
    out = torch.einsum("ts,s...->t...", w, flat)
    return out.reshape(len(tgt_bnds), *values.shape[1:])
