"""Compare generated rollouts against a target CMIP scenario (and MESMER-M).

Given a YAML config listing generated-run output folders (under
``{scratch}/generated_data/``) and a target scenario, computes:

  - a global-mean (lat-weighted) annual trend figure, one line per run plus
    the target and (if configured) MESMER-M, each shown as an ensemble
    mean +/- spread when the run has more than one member;
  - a stats table with, per run (and MESMER-M if available): RMSE computed
    two ways (on monthly data, and on non-overlapping decade-mean maps),
    spatial standard deviation, ensemble spread (nan for single-member /
    deterministic runs), and detrended inter-annual variability (IAV).

Usage:
    python analysis/compare_runs.py --config analysis/configs/compare_runs_example.yaml
"""

import argparse
import glob
import os
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook

# Standardized rollout filename: rollout_surface_{chunk}_{ens_member}_{seed}.pt.
# "det" is the reserved seed value for a deterministic/no-noise rollout, "tf"
# for a teacher_force=True diagnostic rollout (see model/forecast.py's
# generate_rollouts). The trailing seed group stays optional here so files
# from before this standardization (bare "rollout_surface_{chunk}_{ens_member}.pt",
# no seed field at all) still parse.
ROLLOUT_RE = re.compile(r"rollout_surface_(\d+)_(\d+)(?:_(det|tf|\d+))?\.pt$")
SENTINEL = 1e20
GRID_LATS = torch.linspace(-85.5, 85.5, 144)


# --------------------------------------------------------------------------
# Metric primitives
# --------------------------------------------------------------------------


def nanstd(tensor, dim, keepdim=False, correction=1):
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    var = torch.nanmean((tensor - mean) ** 2, dim=dim, keepdim=keepdim)
    if correction > 0:
        n = torch.sum(~torch.isnan(tensor), dim=dim, keepdim=keepdim)
        var = var * n / (n - correction)
    return torch.sqrt(var)


def lat_weighted_mean(map_2d, lats=GRID_LATS):
    """Lat-weighted spatial mean of a (..., lat, lon) map, NaN-aware."""
    weights = torch.cos(torch.deg2rad(lats)).view(*([1] * (map_2d.dim() - 2)), -1, 1)
    weights = weights.expand_as(map_2d)
    mask = ~torch.isnan(map_2d)
    w = weights * mask
    return (torch.nan_to_num(map_2d) * w).sum(dim=(-1, -2)) / w.sum(dim=(-1, -2))


def rmse_map(pred, target, time_dim):
    """Pixel-wise RMSE (time-RMS of the error) -- returns a (..., lat, lon) map."""
    err2 = (pred - target) ** 2
    return torch.sqrt(torch.nanmean(err2, dim=time_dim))


def get_iav(tensor):
    """Detrended inter-annual variability for [member, time, lat, lon].

    Linear-detrends each pixel's time series (per member), then returns the
    std of the residual over time -- shape [member, lat, lon].
    """
    m, t, lat, lon = tensor.shape
    x = torch.linspace(0, t - 1, t, dtype=tensor.dtype).view(1, t, 1, 1)
    x_mean = x.mean()
    y_mean = torch.nanmean(tensor, dim=1, keepdim=True)
    slope = ((x - x_mean) * (tensor - y_mean)).nansum(dim=1, keepdim=True) / (
        (x - x_mean) ** 2
    ).sum(dim=1, keepdim=True)
    intercept = y_mean - slope * x_mean
    detrended = tensor - (slope * x + intercept)
    return torch.std(detrended, dim=1)


def to_annual(monthly):
    """[..., T, lat, lon] -> [..., T // 12, lat, lon], trimming any remainder."""
    t = monthly.shape[-3]
    n_years = t // 12
    trimmed = monthly[..., : n_years * 12, :, :]
    shape = trimmed.shape
    return trimmed.view(*shape[:-3], n_years, 12, *shape[-2:]).mean(dim=-3)


def decade_means(annual, decade_years):
    """[..., n_years, lat, lon] -> [..., n_decades, lat, lon], dropping the remainder."""
    n_years = annual.shape[-3]
    n_decades = n_years // decade_years
    if n_decades == 0:
        return None
    trimmed = annual[..., : n_decades * decade_years, :, :]
    shape = trimmed.shape
    return trimmed.view(*shape[:-3], n_decades, decade_years, *shape[-2:]).mean(dim=-3)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_run(run_dir, var_idx, data_mean, data_std):
    """Load every member of a generated-run folder, denormalized, for one variable.

    Returns a tensor of shape (n_members, T, 144, 144), aligned so that
    result[i] corresponds to target[i] (the same convention paper_figures_*.py
    and every other caller already assume) -- see the alignment note below.
    T is the minimum length across members (chunks are truncated to the
    shortest member so they stack cleanly) plus a 2-step NaN-padded lead-in.

    Alignment note: dataloaders/cmip_random_lead_time.py shifts any
    requested index forward by one step (load_prev=1 for both model
    families), so dataset[0]'s state is calendar month 1, not 0. From there
    the two `generate_rollouts` implementations diverge: DCPPDiffusion
    (flow) explicitly prepends that t0 state as its saved row 0 before any
    prediction (model/diffusion.py, "put in t0 for sanity check"), so its
    row 0 is ground truth at month 1 and its first real prediction (month 2)
    is row 1. ForecastModuleWithCond (deterministic) has the equivalent
    prepend line commented out (model/forecast.py), so its row 0 is already
    its first prediction, also month 2. Net effect, uncorrected: flow's
    saved row i is month (1+i); deterministic's saved row i is month (2+i)
    -- a 1-month and 2-month lead over target index i respectively. This
    was previously uncorrected here (confirmed via a lag scan: RMSE dropped
    sharply at lag -1 for flow and lag -2 for deterministic). A second,
    long-standing loader in this repo (analysis_utils.py's
    load_cmip_experiment) already encodes the same asymmetry via different
    chunk-filename index arithmetic (i-1 for flow, i+1 for deterministic).
    Fixed here by dropping flow's spurious row 0 and left-padding both
    series with 2 NaN rows, so every caller's existing target[i] comparison
    is correct for both model families without further changes -- the
    NaN lead-in is handled by the NaN-tolerant masking every RMSE/mean
    computation in this codebase already does.
    """
    files = glob.glob(os.path.join(run_dir, "rollout_surface_*.pt"))
    if not files:
        raise FileNotFoundError(f"No rollout_surface_*.pt files found in {run_dir}")

    is_flow = "flow" in os.path.basename(os.path.normpath(run_dir)).lower()

    members = {}
    for f in files:
        m = ROLLOUT_RE.search(os.path.basename(f))
        if not m:
            continue
        idx = int(m.group(1))
        member_key = (m.group(2), m.group(3))  # (ens, seed-or-None)
        members.setdefault(member_key, []).append((idx, f))

    member_tensors = []
    for member_key in sorted(members):
        chunks = sorted(members[member_key], key=lambda t: t[0])
        pieces = []
        for _, path in chunks:
            data = torch.load(path, map_location="cpu")
            # (1, T_chunk, n_vars, 1, 144, 144) -> (T_chunk, 144, 144)
            data = data.squeeze(0).squeeze(-3)[:, var_idx]
            data = torch.where(data.abs() > SENTINEL, torch.nan, data)
            pieces.append(data)
        series = torch.cat(pieces, dim=0)
        if is_flow:
            series = series[1:]  # drop the prepended t0 ground-truth row
        lead_in = torch.full((2, *series.shape[1:]), float("nan"))
        member_tensors.append(torch.cat([lead_in, series], dim=0))

    t_min = min(m.shape[0] for m in member_tensors)
    stacked = torch.stack([m[:t_min] for m in member_tensors], dim=0)
    # Grid shape comes from the data itself (144x144 for IPSL, 64x128 for
    # CanESM5 native) rather than a hardcoded 144x144, so this works for
    # both -- data_mean/data_std are already shaped for whichever grid
    # train_dataset was built from.
    denorm = stacked * data_std[var_idx].view(1, 1, 1, 1) + data_mean[var_idx].view(
        1, 1, *stacked.shape[-2:]
    )
    return denorm


def load_run_all_vars(run_dir, data_mean, data_std):
    """Like load_run, but keeps the variable axis instead of indexing var_idx.

    Returns a tensor of shape (n_members, T, n_vars, 144, 144), denormalized.
    """
    files = glob.glob(os.path.join(run_dir, "rollout_surface_*.pt"))
    if not files:
        raise FileNotFoundError(f"No rollout_surface_*.pt files found in {run_dir}")

    members = {}
    for f in files:
        m = ROLLOUT_RE.search(os.path.basename(f))
        if not m:
            continue
        idx = int(m.group(1))
        member_key = (m.group(2), m.group(3))
        members.setdefault(member_key, []).append((idx, f))

    member_tensors = []
    for member_key in sorted(members):
        chunks = sorted(members[member_key], key=lambda t: t[0])
        pieces = []
        for _, path in chunks:
            data = torch.load(path, map_location="cpu")
            # (1, T_chunk, n_vars, 1, 144, 144) -> (T_chunk, n_vars, 144, 144)
            data = data.squeeze(0).squeeze(-3)
            data = torch.where(data.abs() > SENTINEL, torch.nan, data)
            pieces.append(data)
        member_tensors.append(torch.cat(pieces, dim=0))

    t_min = min(m.shape[0] for m in member_tensors)
    stacked = torch.stack([m[:t_min] for m in member_tensors], dim=0)
    denorm = stacked * data_std.view(1, 1, -1, 1, 1) + data_mean.view(1, 1, -1, 144, 144)
    return denorm


def load_run_lev(run_dir, data_mean_lev, data_std_lev):
    """Ocean 'lev' (thetao) rollouts, all depth levels.

    Returns a tensor of shape (n_members, T, n_depth, 144, 144), denormalized.
    """
    files = glob.glob(os.path.join(run_dir, "rollout_lev_*.pt"))
    if not files:
        raise FileNotFoundError(f"No rollout_lev_*.pt files found in {run_dir}")

    members = {}
    for f in files:
        m = ROLLOUT_RE.search(os.path.basename(f).replace("rollout_lev", "rollout_surface"))
        if not m:
            continue
        idx = int(m.group(1))
        member_key = (m.group(2), m.group(3))
        members.setdefault(member_key, []).append((idx, f))

    member_tensors = []
    for member_key in sorted(members):
        chunks = sorted(members[member_key], key=lambda t: t[0])
        pieces = []
        for _, path in chunks:
            data = torch.load(path, map_location="cpu")
            # (1, T_chunk, 1, n_depth, 144, 144) -> (T_chunk, n_depth, 144, 144)
            data = data.squeeze(0).squeeze(1)
            data = torch.where(data.abs() > SENTINEL, torch.nan, data)
            pieces.append(data)
        member_tensors.append(torch.cat(pieces, dim=0))

    t_min = min(m.shape[0] for m in member_tensors)
    stacked = torch.stack([m[:t_min] for m in member_tensors], dim=0)
    denorm = stacked * data_std_lev.view(1, 1, -1, 1, 1) + data_mean_lev.view(1, 1, -1, 144, 144)
    return denorm


def load_run_level(run_dir, var_idx, plevel_idx, data_mean_level, data_std_level):
    """Atmospheric 'level' rollouts (e.g. ta/ua/va/zg/hus), one variable at one pressure level.

    Parallels load_run_lev (ocean 'lev'/thetao) above, but for
    rollout_level_*.pt files, which carry an extra pressure-level axis.

    Returns a tensor of shape (n_members, T, 144, 144), denormalized.
    """
    files = glob.glob(os.path.join(run_dir, "rollout_level_*.pt"))
    if not files:
        raise FileNotFoundError(f"No rollout_level_*.pt files found in {run_dir}")

    members = {}
    for f in files:
        m = ROLLOUT_RE.search(os.path.basename(f).replace("rollout_level", "rollout_surface"))
        if not m:
            continue
        idx = int(m.group(1))
        member_key = (m.group(2), m.group(3))
        members.setdefault(member_key, []).append((idx, f))

    member_tensors = []
    for member_key in sorted(members):
        chunks = sorted(members[member_key], key=lambda t: t[0])
        pieces = []
        for _, path in chunks:
            data = torch.load(path, map_location="cpu")
            # (1, T_chunk, n_vars, n_plevels, 144, 144) -> (T_chunk, 144, 144)
            data = data.squeeze(0)[:, var_idx, plevel_idx]
            data = torch.where(data.abs() > SENTINEL, torch.nan, data)
            pieces.append(data)
        member_tensors.append(torch.cat(pieces, dim=0))

    t_min = min(m.shape[0] for m in member_tensors)
    stacked = torch.stack([m[:t_min] for m in member_tensors], dim=0)
    denorm = stacked * data_std_level[var_idx, plevel_idx].view(1, 1, 1, 1) + data_mean_level[
        var_idx, plevel_idx
    ].view(1, 1, 144, 144)
    return denorm


def load_run_level_all(run_dir, data_mean_level, data_std_level):
    """Like load_run_level, but keeps the variable and pressure-level axes.

    data_mean_level/data_std_level: (n_vars, n_plevels, 144, 144) and
    (n_vars, n_plevels, 1, 1) respectively (i.e. train_dataset.data_mean/
    data_std["level"] as-is -- their trailing (1, 1) / (144, 144) dims
    broadcast fine against the extra leading (member, T) axes below without
    any reshaping).

    Returns a tensor of shape (n_members, T, n_vars, n_plevels, 144, 144), denormalized.
    """
    files = glob.glob(os.path.join(run_dir, "rollout_level_*.pt"))
    if not files:
        raise FileNotFoundError(f"No rollout_level_*.pt files found in {run_dir}")

    members = {}
    for f in files:
        m = ROLLOUT_RE.search(os.path.basename(f).replace("rollout_level", "rollout_surface"))
        if not m:
            continue
        idx = int(m.group(1))
        member_key = (m.group(2), m.group(3))
        members.setdefault(member_key, []).append((idx, f))

    member_tensors = []
    for member_key in sorted(members):
        chunks = sorted(members[member_key], key=lambda t: t[0])
        pieces = []
        for _, path in chunks:
            data = torch.load(path, map_location="cpu")
            # (1, T_chunk, n_vars, n_plevels, 144, 144) -> (T_chunk, n_vars, n_plevels, 144, 144)
            data = data.squeeze(0)
            data = torch.where(data.abs() > SENTINEL, torch.nan, data)
            pieces.append(data)
        member_tensors.append(torch.cat(pieces, dim=0))

    t_min = min(m.shape[0] for m in member_tensors)
    stacked = torch.stack([m[:t_min] for m in member_tensors], dim=0)
    return stacked * data_std_level + data_mean_level


def load_target(scratch, realization, scenario, var_idx, dataset_path="memmap_filled_in"):
    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    surface = td["surface"][var_idx]  # (T, 144, 144)
    surface = torch.where(surface.abs() > SENTINEL, torch.nan, surface)
    time = pd.DatetimeIndex(np.asarray(td["time"])) if "time" in td.keys() else None
    return surface, time


def load_mesmer(path, variable, time_index):
    """Load MESMER-M 'tas' data and align it to the given monthly DatetimeIndex.

    Returns (n_realisations, T, lat, lon) or None if unavailable / variable
    isn't tas / the requested window isn't covered.
    """
    if variable != "tas":
        warnings.warn(
            f"MESMER-M comparison only supports 'tas', got variable={variable!r}; skipping."
        )
        return None
    if not os.path.exists(path):
        warnings.warn(f"MESMER file not found at {path}; skipping MESMER-M row.")
        return None

    ds = xr.open_dataset(path)
    da = ds["tas"]
    if "variable" in da.dims:
        da = da.isel(variable=0)
    if "member" in da.dims:
        da = da.isel(member=0)

    start, end = time_index[0], time_index[-1]
    try:
        da = da.sel(time=slice(start, end))
    except Exception as e:  # noqa: BLE001 -- report and bail, don't crash the run
        warnings.warn(f"Could not slice MESMER time range {start}..{end}: {e}")
        return None
    if da.sizes.get("time", 0) != len(time_index):
        warnings.warn(
            f"MESMER time coverage ({da.sizes.get('time', 0)} months) doesn't match "
            f"requested window ({len(time_index)} months); truncating to the shorter one."
        )
        n = min(da.sizes.get("time", 0), len(time_index))
        if n == 0:
            return None
        da = da.isel(time=slice(0, n))

    arr = da.transpose("realisation", "time", "lat", "lon").values
    return torch.tensor(arr, dtype=torch.float32)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def compute_metrics(pred_monthly, target_monthly, decade_years):
    """pred_monthly: (member, T, 144, 144); target_monthly: (T, 144, 144)."""
    n_members = pred_monthly.shape[0]

    pred_annual = to_annual(pred_monthly)
    target_annual = to_annual(target_monthly)

    # RMSE, monthly resolution: per-member pixel-wise time-RMS vs target, then
    # averaged across members.
    monthly_map = rmse_map(pred_monthly, target_monthly[None], time_dim=1)
    rmse_monthly = lat_weighted_mean(monthly_map).mean().item()

    # RMSE, decadal resolution: chunk annual data into non-overlapping
    # decade_years-year blocks, then pixel-wise time-RMS across those blocks.
    pred_decades = decade_means(pred_annual, decade_years)
    target_decades = decade_means(target_annual, decade_years)
    if pred_decades is None or target_decades is None:
        rmse_decadal = float("nan")
    else:
        decadal_map = rmse_map(pred_decades, target_decades[None], time_dim=1)
        rmse_decadal = lat_weighted_mean(decadal_map).mean().item()

    # Spatial std: nanstd across space, per member & year, averaged over both.
    sp_std = nanstd(pred_annual, dim=(-1, -2)).mean().item()

    # Ensemble spread: std across members of the global-mean annual series,
    # averaged over years. NaN for n_members == 1 (deterministic / no ensemble).
    global_mean_annual = lat_weighted_mean(pred_annual)  # (member, n_years)
    spread = global_mean_annual.std(dim=0, unbiased=True).mean().item()

    # Detrended IAV: per member & pixel, averaged over members + space.
    iav = get_iav(pred_annual).nanmean().item()

    return {
        "rmse_monthly": rmse_monthly,
        "rmse_decadal": rmse_decadal,
        "spatial_std": sp_std,
        "ensemble_spread": spread,
        "iav": iav,
        "n_members": n_members,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a comparison YAML config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    decade_years = cfg.get("decade_years", 10)
    output_dir = Path(cfg.get("output_dir", "analysis/outputs/compare_runs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    surface_variables = list(hydra_cfg.module.surface_variables)
    variable = cfg.variable
    var_idx = surface_variables.index(variable)

    data_mean_all = train_dataset.data_mean["surface"].squeeze(1)  # (n_vars, 144, 144)
    data_std_all = train_dataset.data_std["surface"].view(-1)  # (n_vars,)

    realization = cfg.target.get("realization", "r1i1p1f1")
    scenario = cfg.target.scenario
    target_monthly, target_time = load_target(
        scratch, realization, scenario, var_idx, dataset_path=hydra_cfg.module.dataset_path
    )

    # MESMER-M emulations are land-only (NaN over ocean). Loaded up front, if
    # configured, so its land mask can be applied to every other series --
    # otherwise MESMER's land-only mean would be compared against everyone
    # else's full-globe (land+ocean) mean, which is not an apples-to-apples
    # comparison (matches the masking practice in
    # analysis/global_mean_stats.ipynb).
    mesmer_monthly = None
    land_mask = None
    if "mesmer" in cfg:
        if target_time is None:
            warnings.warn(
                "Target memmap has no 'time' coordinate; cannot align MESMER-M, skipping."
            )
        else:
            window_time = target_time[1:]  # full available window; load_mesmer truncates as needed
            mesmer_monthly = load_mesmer(cfg.mesmer.path, variable, window_time)
            if mesmer_monthly is not None:
                land_mask = torch.isnan(mesmer_monthly[0, 0])  # (144, 144), True over ocean
                target_monthly = torch.where(land_mask, torch.nan, target_monthly)

    results = {}
    trend_series = {}

    for run_cfg in cfg.runs:
        name = run_cfg.name
        label = run_cfg.get("label", name)
        run_dir = os.path.join(scratch, "generated_data", name)
        pred_monthly = load_run(run_dir, var_idx, data_mean_all, data_std_all)
        if land_mask is not None:
            pred_monthly = torch.where(land_mask, torch.nan, pred_monthly)

        t_common = min(pred_monthly.shape[1], target_monthly.shape[0] - 1)
        pred_monthly = pred_monthly[:, :t_common]
        # rollout index k (0-based) predicts target memmap row k+1 (row 0 is
        # the initial condition fed into the model, never itself predicted).
        target_aligned = target_monthly[1 : 1 + t_common]

        results[label] = compute_metrics(pred_monthly, target_aligned, decade_years)
        trend_series[label] = lat_weighted_mean(to_annual(pred_monthly))  # (member, n_years)

    mesmer_series = None
    if mesmer_monthly is not None:
        target_aligned = target_monthly[1 : 1 + mesmer_monthly.shape[1]]
        results["MESMER-M"] = compute_metrics(mesmer_monthly, target_aligned, decade_years)
        mesmer_series = lat_weighted_mean(to_annual(mesmer_monthly))

    target_series = lat_weighted_mean(to_annual(target_monthly))

    # --- Figure: global-mean annual trend ---------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    start_year = target_time[0].year if target_time is not None else 0
    for label, series in trend_series.items():
        years = np.arange(start_year + 1, start_year + 1 + series.shape[1])
        mean = series.mean(dim=0).numpy()
        ax.plot(years, mean, label=label, lw=1.3)
        if series.shape[0] > 1:
            std = series.std(dim=0, unbiased=True).numpy()
            ax.fill_between(years, mean - std, mean + std, alpha=0.2)
    if mesmer_series is not None:
        years = np.arange(start_year + 1, start_year + 1 + mesmer_series.shape[1])
        mean = mesmer_series.mean(dim=0).numpy()
        std = mesmer_series.std(dim=0, unbiased=True).numpy()
        ax.plot(years, mean, label="MESMER-M", lw=1.3, linestyle="--", color="black")
        ax.fill_between(years, mean - std, mean + std, alpha=0.15, color="black")
    # target's annual bins start from row 0 (calendar Jan of start_year), one
    # calendar year "ahead" of the model/MESMER bins, which start from row 1
    # (they never predict the initial condition itself).
    years = np.arange(start_year, start_year + target_series.shape[0])
    ax.plot(years, target_series.numpy(), label=scenario, lw=1.5, linestyle=":", color="red")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Global mean {variable} (K)" if variable == "tas" else f"Global mean {variable}")
    ax.set_title(f"Global mean {variable} trend -- target {scenario}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = output_dir / f"global_mean_trend_{variable}_{scenario}.png"
    fig.savefig(fig_path, dpi=200)
    print(f"Saved figure to {fig_path}")

    # --- Table ---------------------------------------------------------------
    table = pd.DataFrame(results).T[
        ["rmse_monthly", "rmse_decadal", "spatial_std", "ensemble_spread", "iav", "n_members"]
    ]
    table.columns = [
        "RMSE (monthly)",
        f"RMSE ({decade_years}yr avg)",
        "Spatial Std",
        "Ensemble Spread",
        "IAV",
        "N members",
    ]
    csv_path = output_dir / f"stats_table_{variable}_{scenario}.csv"
    table.to_csv(csv_path)
    print(f"Saved stats table to {csv_path}")
    print()
    with pd.option_context("display.float_format", "{:.4f}".format):
        print(table)


if __name__ == "__main__":
    main()
