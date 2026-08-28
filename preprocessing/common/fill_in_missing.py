"""Permanent-mask + linear-gap-fill for atmospheric/ocean level variables.

Vectorized over time and lat*lon (loops only over the small level axis).
Intended to match fill_in_missing_data.py's xarray-based
`interpolate_na(method="linear")` behavior: interior gaps get filled, a
level that's NaN at every timestep stays permanently masked, and
leading/trailing NaN runs along a level are left unfilled (no
extrapolation). This is currently only verified against canesm5 output --
fill_in_missing_data.py's raw pre-fill IPSL data no longer exists on disk
(interpolation_project_datasets/ was cleaned up after conversion to memmap),
so this hasn't been diffed against a real IPSL file. Don't assume it's a
drop-in replacement for fill_in_missing_data.py without that verification.
"""

import torch
import torch.nn.functional as F


def _local_neighborhood_mean(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """x: [T, H, W]. kernel_size x kernel_size neighborhood mean at every
    point, per timestep. Longitude (W) wraps (circular pad, date line);
    latitude (H) does not (replicate pad -- no pole wrap)."""
    pad = kernel_size // 2
    x4 = x.unsqueeze(1)
    x4 = F.pad(x4, (pad, pad, 0, 0), mode="circular")
    x4 = F.pad(x4, (0, 0, pad, pad), mode="replicate")
    return F.avg_pool2d(x4, kernel_size, stride=1).squeeze(1)


def detect_persistent_spatial_outliers(
    arr: torch.Tensor,
    kernel_size: int = 5,
    anomaly_threshold: float = 10.0,
    min_bad_frac: float = 0.9,
    direction: str = "both",
) -> torch.Tensor:
    """Flag grid cells that read anomalously far from their local
    neighborhood at nearly every timestep -- e.g. the persistent -30 to
    -45K single-cell cold spots found in CanESM5's raw tas at a handful of
    fixed Antarctic interior cells (~82S, verified against the raw source
    NetCDF, not a regridding artifact). Unlike a plain per-timestep outlier
    test, requiring the anomaly at *most* timesteps builds one permanent
    mask, the same idea as mask_and_fill's per-level permanent-NaN mask, so
    a genuine one-off weather extreme (real, transient) isn't flagged --
    only a cell that's *always* out of step with its surroundings is.

    Args:
        arr: [T, H, W] float tensor (single variable/level).
        kernel_size: neighborhood window (odd) used both to define "local"
            and, later in fill_spatial_outliers, to source the fill value.
        anomaly_threshold: |value - neighborhood_mean| above this (same
            units as arr, e.g. Kelvin) counts as anomalous at that timestep.
        min_bad_frac: fraction of timesteps a cell must be anomalous in to
            be permanently masked.
        direction: "both" flags cells anomalous in either sign; "low" only
            flags cells reading colder than their neighborhood (the observed
            CanESM5 case), "high" only warmer.

    Returns:
        [H, W] bool mask, True = permanently anomalous cell.
    """
    assert direction in ("both", "low", "high")
    local_mean = _local_neighborhood_mean(arr, kernel_size)
    anomaly = arr - local_mean
    if direction == "both":
        is_bad_t = anomaly.abs() > anomaly_threshold
    elif direction == "low":
        is_bad_t = anomaly < -anomaly_threshold
    else:
        is_bad_t = anomaly > anomaly_threshold
    return is_bad_t.float().mean(dim=0) >= min_bad_frac


def fill_spatial_outliers(
    arr: torch.Tensor,
    bad_mask: torch.Tensor,
    kernel_size: int = 5,
    n_passes: int = 3,
) -> torch.Tensor:
    """Replace cells flagged by detect_persistent_spatial_outliers with a
    neighbor-average inpaint, at every timestep. A masked cell's own
    (anomalous) value never contributes to any fill -- neighbors are
    averaged only over not-currently-masked cells -- and multi-cell blobs
    resolve from their edges inward: a pass fills whichever masked cells
    currently have >=1 unmasked neighbor, then the next pass can use those
    newly-filled cells as valid neighbors for the remaining ones.

    Args:
        arr: [T, H, W] float tensor (same variable/level bad_mask was
            computed from).
        bad_mask: [H, W] bool, True = cell to fill (same mask at every
            timestep -- see detect_persistent_spatial_outliers).
        kernel_size: neighborhood window used to source each fill (odd).
        n_passes: max inpainting passes; safe to over-provision, a pass
            that fills nothing exits early.

    Returns:
        [T, H, W] tensor, unmasked cells unchanged.
    """
    T, H, W = arr.shape
    pad = kernel_size // 2
    filled = arr.clone()
    still_bad = bad_mask.clone()

    for _ in range(n_passes):
        if not still_bad.any():
            break
        valid = (~still_bad).float()
        valid_t = valid[None, None].expand(T, 1, H, W)
        vals_t = (filled * valid[None]).unsqueeze(1)

        valid_t = F.pad(valid_t, (pad, pad, 0, 0), mode="circular")
        valid_t = F.pad(valid_t, (0, 0, pad, pad), mode="replicate")
        vals_t = F.pad(vals_t, (pad, pad, 0, 0), mode="circular")
        vals_t = F.pad(vals_t, (0, 0, pad, pad), mode="replicate")

        neighbor_sum = F.avg_pool2d(vals_t, kernel_size, stride=1).squeeze(1) * kernel_size**2
        neighbor_count = F.avg_pool2d(valid_t, kernel_size, stride=1).squeeze(1) * kernel_size**2
        neighbor_count_2d = neighbor_count[0]  # bad_mask/neighbor_count don't vary over T

        can_fill_now = still_bad & (neighbor_count_2d > 0.5)
        if not can_fill_now.any():
            break
        neighbor_mean = neighbor_sum / neighbor_count.clamp(min=1e-6)
        filled = torch.where(can_fill_now[None], neighbor_mean, filled)
        still_bad = still_bad & ~can_fill_now

    return filled


def mask_and_fill_spatial_outliers(
    arr: torch.Tensor,
    kernel_size: int = 5,
    anomaly_threshold: float = 10.0,
    min_bad_frac: float = 0.9,
    direction: str = "both",
    n_fill_passes: int = 3,
) -> torch.Tensor:
    """Convenience wrapper: detect_persistent_spatial_outliers then
    fill_spatial_outliers. arr: [T, H, W]. See both for parameter meaning.
    """
    bad_mask = detect_persistent_spatial_outliers(
        arr, kernel_size, anomaly_threshold, min_bad_frac, direction
    )
    return fill_spatial_outliers(arr, bad_mask, kernel_size, n_fill_passes)


def _fill_2d(x: torch.Tensor) -> torch.Tensor:
    """x: [T, L] float32, NaN = missing. Vectorized linear fill along L, no
    extrapolation (matches xarray's interpolate_na: interior gaps filled,
    leading/trailing NaNs left as-is)."""
    T, L = x.shape
    valid = ~torch.isnan(x)
    idx = torch.arange(L, dtype=torch.long, device=x.device).expand(T, L)

    idx_or_neg1 = torch.where(valid, idx, idx.new_full((), -1))
    prev_idx = torch.cummax(idx_or_neg1, dim=-1).values

    idx_or_big = torch.where(valid, idx, idx.new_full((), L))
    next_idx = torch.cummin(idx_or_big.flip(-1), dim=-1).values.flip(-1)

    x_prev = torch.gather(x, -1, prev_idx.clamp(min=0))
    x_next = torch.gather(x, -1, next_idx.clamp(max=L - 1))

    denom = (next_idx - prev_idx).clamp(min=1).float()
    frac = (idx - prev_idx).float() / denom
    interp_val = x_prev + (x_next - x_prev) * frac

    can_fill = (~valid) & (prev_idx >= 0) & (next_idx <= L - 1)
    return torch.where(can_fill, interp_val, x)


def mask_and_fill(arr: torch.Tensor) -> torch.Tensor:
    """arr: [time, level, lat, lon] float tensor."""
    T, P, H, W = arr.shape
    L = H * W
    x = arr.reshape(T, P, L).float()
    x = torch.where(x < 1e30, x, x.new_full((), float("nan")))

    out = torch.empty_like(x)
    for p in range(P):
        xp = x[:, p, :]
        mask_p = torch.isnan(xp).all(dim=0)  # permanently-missing this level
        filled = _fill_2d(xp)
        out[:, p, :] = torch.where(mask_p.unsqueeze(0), xp.new_full((), float("nan")), filled)

    return out.reshape(T, P, H, W)
