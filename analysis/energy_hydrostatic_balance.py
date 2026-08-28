"""Hydrostatic and energy-balance diagnostics for a CMIP rollout checkpoint.

Loads a predicted rollout (surface/level/ocean fields) and the corresponding
ground-truth CMIP target for ssp585 and ssp434, then computes:

  1. Hydrostatic consistency of the (predicted or target) level fields
     (see analysis/hydrostatic_consistency.ipynb).
  2. Global atmosphere + ocean energy conservation: the rate of change of
     column atmospheric energy plus ocean heat content should track the
     net TOA/surface flux (`net_flux`).
  3. Clausius-Clapeyron consistency of moisture (`hus`) vs. temperature
     (`ta`): (a) a strict supersaturation bound (hus should never exceed
     saturation specific humidity), and (b) the ~7%/K scaling of
     specific humidity with temperature in the lower troposphere.
  4. Pred-vs-target MAE (area-weighted, see energy_component_mae) for the
     column-integrated atmospheric energy (`e_atm`), ocean heat content
     (`e_oce`), and the four physical terms decomposed out of `e_atm`
     (sensible, potential, kinetic, latent -- see
     atmospheric_column_energy_decomposed).

Predicted level ("ta","zg","ua","va","hus") and ocean ("lev"/thetao) rollout
files are only written once `generate_rollouts` in model/forecast.py has the
corresponding `torch.save(out["level"], ...)` / `torch.save(out["lev"], ...)`
calls uncommented -- as of this writing only `rollout_surface_*.pt` is saved,
so predicted-side hydrostatic/energy diagnostics will raise FileNotFoundError
until those rollouts are regenerated. Target-side diagnostics work today
since ground truth is read directly from the CMIP memmap files.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
from tensordict.tensordict import TensorDict

from analysis.plot_style import (
    AC_SSP_LABEL,
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TARGET_COLOR,
    TICK_FONTSIZE,
)
from analysis.plot_style import MODEL_COLORS as _MODEL_COLORS
from ArchesClimate.analysis.analysis_utils import initialize_notebook

_AC_SSP_COLOR = _MODEL_COLORS[AC_SSP_LABEL]

CP = 1004.6  # J/kg/K, specific heat of dry air at constant pressure
G = 9.80665  # m/s^2
LV = 2.5e6  # J/kg, latent heat of vaporization
RD = 287.05  # J/kg/K, dry air gas constant
RHO_W = 1026.0  # kg/m^3, seawater density
CPW = 3996.0  # J/kg/K, seawater specific heat capacity
RV = 461.5  # J/kg/K, water vapor gas constant
R_EARTH = 6.371e6  # m
AVG_SECONDS_PER_MONTH = 365.25 / 12 * 86400  # average calendar month length
CC_TARGET_PRESSURE = 85000.0  # Pa, lower-troposphere level used for the 7%/K scaling check

SCENARIO_DOMAIN = {"ssp585": "test", "ssp434": "val", "ssp534-over": "val3"}


def infer_config_module(experiment_name: str) -> str:
    """Recover the Hydra module config name from a generated_data dir name.

    The dir name is `{module}_{num_inference_steps}_{num_members}_..._{ckpt}`,
    so the module name is everything before the first purely-numeric token.
    """
    tokens = experiment_name.split("_")
    for idx, tok in enumerate(tokens):
        if tok.replace(".", "", 1).isdigit():
            return "_".join(tokens[:idx])
    raise ValueError(f"could not infer config module from {experiment_name!r}")


def experiment_dir_for_scenario(base_experiment_name: str, scenario: str) -> str:
    """Swap the val/test marker in a rollout dir name for the given scenario."""
    domain = SCENARIO_DOMAIN[scenario]
    if "_val3_" in base_experiment_name:
        return re.sub("_val3_", f"_{domain}_", base_experiment_name)
    if "_val_" in base_experiment_name:
        return re.sub("_val_", f"_{domain}_", base_experiment_name)
    if "_test_" in base_experiment_name:
        return re.sub("_test_", f"_{domain}_", base_experiment_name)
    raise ValueError(
        "experiment_name must contain '_val_', '_val3_', or '_test_' to "
        "identify the rollout's original target domain"
    )


def load_predicted_field(exp_dir: Path, field: str, member: int, seed: int = 0) -> torch.Tensor:
    """Load and concatenate rollout chunks for one field.

    Rollout chunks are named `rollout_{field}_{i}_{member}_{seed}.pt`, matching
    the naming used by both forecast.py (always seed 0) and diffusion.py
    (one file per ensemble seed).

    Returns the raw (normalized) tensor of shape [T, nvar, levels_or_1, lat, lon].
    """
    pattern = re.compile(rf"rollout_{field}_(\d+)_{member}_{seed}\.pt$")
    files = [
        (int(m.group(1)), f)
        for f in exp_dir.glob(f"rollout_{field}_*_{member}_{seed}.pt")
        if (m := pattern.match(f.name))
    ]
    if not files:
        raise FileNotFoundError(
            f"no rollout_{field}_*_{member}_{seed}.pt files in {exp_dir}. "
            "level/lev rollouts are only saved once the corresponding "
            "torch.save calls are uncommented in model/forecast.py."
        )
    files.sort(key=lambda x: x[0])
    chunks = [torch.load(f, map_location="cpu") for _, f in files]
    data = torch.cat(chunks, dim=1)[0]  # drop batch dim
    return torch.where(data > 1e20, torch.nan, data)


def load_predicted(exp_dir: Path, dataset: Any, member: int = 0) -> dict[str, torch.Tensor]:
    """Load and denormalize predicted surface/level/lev fields for one member."""
    surface = load_predicted_field(exp_dir, "surface", member)
    surface = surface * dataset.data_std["surface"] + dataset.data_mean["surface"]
    surface = surface.squeeze(-3)  # [T, nvar, lat, lon]

    level = load_predicted_field(exp_dir, "level", member)
    level = (
        level * dataset.data_std["level"] + dataset.data_mean["level"]
    )  # [T, nvar, levels, lat, lon]

    lev = load_predicted_field(exp_dir, "lev", member)
    lev = lev * dataset.data_std["lev"] + dataset.data_mean["lev"]
    lev = lev.squeeze(1)  # [T, depth_levels, lat, lon] (single ocean var: thetao)

    return dict(surface=surface, level=level, lev=lev)


def load_target(
    scenario: str,
    start: int,
    stop: int,
    realization: str = "r1i1p1f1",
    dataset_path: str = "memmap_filled_in",
) -> dict[str, torch.Tensor]:
    """Load ground-truth CMIP fields for one scenario directly from the memmap store.

    Only months [start, stop) are read off disk to avoid materializing the
    full ~85-year memmap (~1032 months) into RAM. Fields are stored already
    in physical units (no denormalization needed).

    dataset_path: subdirectory under $SCRATCH holding the memmap files --
    pass the training module config's own `dataset_path` (e.g.
    "memmap_filled_in_full_ozone"), since it varies per model family and
    isn't always the default.
    """
    scratch = Path(os.environ.get("SCRATCH", "/scratch/gclyne"))
    path = scratch / dataset_path / f"{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(str(path))
    surface = td["surface"][:, start:stop].permute(1, 0, 2, 3).clone()  # [T, nvar, lat, lon]
    level = td["level"][:, start:stop].permute(1, 0, 2, 3, 4).clone()  # [T, nvar, levels, lat, lon]
    lev = td["lev"][0, start:stop].clone()  # [T, depth_levels, lat, lon]
    return dict(surface=surface, level=level, lev=lev)


def hydrostatic_residual(
    level: torch.Tensor,
    pressure_levels: list[float],
    level_vars: list[str],
    return_scale: bool = False,
) -> torch.Tensor:
    """Hydrostatic residual: dz_geo - dz_hydro, see hydrostatic_consistency.ipynb.

    Computed per grid cell (no spatial/temporal averaging here) -- callers
    that want a zonal or time mean average this local residual afterward
    (see plot_hydrostatic_residual), rather than averaging ta/zg first and
    computing the residual on an already-smoothed field.

    Returns tensor of shape [T, levels-1, lat, lon]. If return_scale, also
    returns dz_hydro (same shape) -- the hydrostatic layer-thickness scale
    each residual is measured against, useful for a relative/percentage
    error (residual / dz_hydro).
    """
    ta = level[:, level_vars.index("ta")]
    zg = level[:, level_vars.index("zg")]

    p = torch.tensor(pressure_levels, dtype=level.dtype)
    dz_geo = zg[:, 1:] - zg[:, :-1]
    t_avg = (ta[:, 1:] + ta[:, :-1]) / 2
    d_ln_p = torch.log(p[:-1] / p[1:]).view(1, -1, 1, 1)
    dz_hydro = (RD * t_avg / G) * d_ln_p
    residual = dz_geo - dz_hydro
    if return_scale:
        return residual, dz_hydro
    return residual


def atmospheric_column_energy_decomposed(
    level: torch.Tensor,
    surface: torch.Tensor,
    pressure_levels: list[float],
    level_vars: list[str],
    surface_vars: list[str],
) -> dict[str, torch.Tensor]:
    """Vertically-integrated atmospheric energy per unit area [J/m^2], split by term.

    Same mass-weighted vertical integral (dp/g, clipped to surface pressure
    `ps` so levels below ground contribute zero) as atmospheric_column_energy,
    but keeps the four physical terms of E = cp*ta + g*zg + 0.5*(ua^2+va^2) +
    Lv*hus separate instead of summing them first:
      - sensible: cp*ta
      - potential: g*zg
      - kinetic: 0.5*(ua^2+va^2)
      - latent: Lv*hus

    Returns a dict of tensors, each [T, lat, lon]: "sensible", "potential",
    "kinetic", "latent", and "total" (== atmospheric_column_energy's output).
    """
    ta = level[:, level_vars.index("ta")]
    zg = level[:, level_vars.index("zg")]
    ua = level[:, level_vars.index("ua")]
    va = level[:, level_vars.index("va")]
    hus = level[:, level_vars.index("hus")]
    ps = surface[:, surface_vars.index("ps")]  # [T, lat, lon]

    terms = {
        "sensible": CP * ta,
        "potential": G * zg,
        "kinetic": 0.5 * (ua**2 + va**2),
        "latent": LV * hus,
    }  # each [T, levels, lat, lon]

    p = torch.tensor(pressure_levels, dtype=level.dtype)
    mid = (p[:-1] + p[1:]) / 2
    upper = torch.cat([mid, p.new_zeros(1)])  # top-of-layer pressure (0 Pa at TOA)
    lower = torch.cat([p.new_tensor([float("inf")]), mid])  # bottom-of-layer pressure

    lower_clipped = torch.minimum(lower.view(1, -1, 1, 1), ps.unsqueeze(1))
    thickness = (lower_clipped - upper.view(1, -1, 1, 1)).clamp(min=0)  # [T, levels, lat, lon]

    out = {}
    total = None
    for name, e in terms.items():
        contribution = torch.where(thickness > 0, e * thickness, torch.zeros_like(e))
        integrated = (contribution / G).sum(dim=1)  # [T, lat, lon]
        out[name] = integrated
        total = integrated if total is None else total + integrated
    out["total"] = total
    return out


def atmospheric_column_energy(
    level: torch.Tensor,
    surface: torch.Tensor,
    pressure_levels: list[float],
    level_vars: list[str],
    surface_vars: list[str],
) -> torch.Tensor:
    """Vertically-integrated atmospheric energy per unit area [J/m^2].

    Integrates E = cp*ta + g*zg + 0.5*(ua^2+va^2) + Lv*hus over mass (dp/g),
    clipping each level's layer thickness to the surface pressure `ps` so
    that levels below ground contribute zero. See
    atmospheric_column_energy_decomposed for the per-term breakdown.

    Returns tensor of shape [T, lat, lon].
    """
    return atmospheric_column_energy_decomposed(
        level, surface, pressure_levels, level_vars, surface_vars
    )["total"]


def ocean_heat_content(lev: torch.Tensor, depth_levels: list[float]) -> torch.Tensor:
    """Vertically-integrated ocean heat content per unit area [J/m^2].

    Integrates rho_w * cpw * thetao over depth using layer thicknesses derived
    from midpoints between successive depth levels.

    Returns tensor of shape [T, lat, lon].
    """
    d = torch.tensor(depth_levels, dtype=lev.dtype)
    mid = (d[:-1] + d[1:]) / 2
    edges = torch.cat([d.new_zeros(1), mid, (d[-1] + (d[-1] - mid[-1])).view(1)])
    dz = (edges[1:] - edges[:-1]).view(1, -1, 1, 1)
    return (RHO_W * CPW * lev * dz).sum(dim=1)


def grid_cell_area(n_lat: int = 144, n_lon: int = 144) -> torch.Tensor:
    """Approximate lat/lon grid cell areas [m^2] on a regular 144x144 grid.

    Pole rows collapse to zero area (cos(+-90deg) == 0), a known artifact of
    including pole points directly in a regular lat/lon grid.
    """
    lat = np.linspace(-90, 90, n_lat)
    dlat = np.deg2rad(180 / n_lat)
    dlon = np.deg2rad(360 / n_lon)
    area_per_lat = (R_EARTH**2) * np.cos(np.deg2rad(lat)) * dlat * dlon
    area = np.broadcast_to(area_per_lat[:, None], (n_lat, n_lon))
    return torch.tensor(area.copy(), dtype=torch.float64)


def global_sum(field: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """Area-weighted global sum of a per-unit-area field. field: [T, lat, lon] -> [T]."""
    weighted = field.double() * area
    return torch.nansum(weighted, dim=(-2, -1))


def area_weighted_mean(field: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
    """Area-weighted global mean of a field, ignoring NaN cells. field: [T, lat, lon] -> [T]."""
    valid = ~torch.isnan(field)
    field64 = field.double()
    weighted = torch.where(valid, field64 * area, torch.zeros_like(field64))
    area_valid = torch.where(valid, area.expand_as(field64), torch.zeros_like(field64))
    return weighted.sum(dim=(-2, -1)) / area_valid.sum(dim=(-2, -1))


def saturation_specific_humidity(ta: torch.Tensor, pressure_pa: torch.Tensor) -> torch.Tensor:
    """Saturation specific humidity via the Magnus-Tetens approximation.

    Args:
        ta: Temperature [K].
        pressure_pa: Pressure [Pa], broadcastable against `ta`.
    """
    p_hpa = pressure_pa / 100.0
    es = 6.112 * torch.exp(17.67 * (ta - 273.15) / (ta - 29.65))  # hPa
    return 0.622 * es / (p_hpa - 0.378 * es)


def supersaturation_diagnostic(
    level: torch.Tensor, pressure_levels: list[float], level_vars: list[str]
) -> dict[str, torch.Tensor]:
    """Frequency and magnitude of hus > q_sat (a hard thermodynamic violation).

    Returns per-level violation frequency [levels] and the excess field
    hus - q_sat [T, levels, lat, lon] (positive where supersaturated).
    """
    ta = level[:, level_vars.index("ta")]
    hus = level[:, level_vars.index("hus")]
    p = torch.tensor(pressure_levels, dtype=level.dtype).view(1, -1, 1, 1)

    q_sat = saturation_specific_humidity(ta, p)
    excess = hus - q_sat

    valid = ~torch.isnan(hus) & ~torch.isnan(ta)
    violation = (excess > 0) & valid

    per_level_frequency = violation.float().sum(dim=(0, 2, 3)) / valid.float().sum(
        dim=(0, 2, 3)
    ).clamp(min=1)
    overall_frequency = violation.float().sum() / valid.float().sum().clamp(min=1)
    mean_violation_magnitude = torch.where(
        violation, excess, torch.zeros_like(excess)
    ).sum() / violation.float().sum().clamp(min=1)

    return dict(
        excess=excess,
        per_level_frequency=per_level_frequency,
        overall_frequency=overall_frequency,
        mean_violation_magnitude=mean_violation_magnitude,
    )


def clausius_clapeyron_scaling(
    level: torch.Tensor,
    pressure_levels: list[float],
    level_vars: list[str],
    area: torch.Tensor,
    target_pressure: float = CC_TARGET_PRESSURE,
) -> dict[str, Any]:
    """Fractional scaling of lower-troposphere specific humidity with temperature.

    Fits d(ln hus)/d(ta) across the rollout's monthly global means at the
    pressure level nearest `target_pressure`, and compares it to the C-C
    prediction Lv / (Rv * ta^2) (~0.07 /K near 285 K).
    """
    level_index = int(np.argmin(np.abs(np.array(pressure_levels) - target_pressure)))
    ta = level[:, level_vars.index("ta"), level_index]  # [T, lat, lon]
    hus = level[:, level_vars.index("hus"), level_index]

    ta_mean = area_weighted_mean(ta, area)
    hus_mean = area_weighted_mean(hus, area)
    log_hus_mean = torch.log(hus_mean)

    slope, intercept = np.polyfit(ta_mean.numpy(), log_hus_mean.numpy(), 1)
    theoretical_slope = LV / (RV * ta_mean.mean().item() ** 2)

    return dict(
        level_index=level_index,
        ta_mean=ta_mean,
        hus_mean=hus_mean,
        log_hus_mean=log_hus_mean,
        slope=float(slope),
        intercept=float(intercept),
        theoretical_slope=theoretical_slope,
    )


def energy_component_mae(
    pred: torch.Tensor, target: torch.Tensor, area: torch.Tensor
) -> torch.Tensor:
    """Area-weighted MAE between predicted and target per-unit-area energy fields.

    pred, target: [T, lat, lon], same T (caller truncates to the common
    overlap when pred/target rollouts differ in length). Returns per-timestep
    MAE [T] (area-weighted mean of |pred - target| over lat/lon); take
    `.mean()` for a single scalar over the whole trajectory.
    """
    return area_weighted_mean((pred - target).abs(), area)


def conservation_check(
    e_atm_global: torch.Tensor,
    e_oce_global: torch.Tensor,
    net_flux_global: torch.Tensor,
    dt_seconds: float,
) -> torch.Tensor:
    """Residual of d(E_atm + E_oce)/dt - F_net, per step. Returns [T-1]."""
    e_total = e_atm_global + e_oce_global
    de_dt = (e_total[1:] - e_total[:-1]) / dt_seconds
    f_avg = (net_flux_global[1:] + net_flux_global[:-1]) / 2
    return de_dt - f_avg


def _hydrostatic_zonal_mean_and_pct(
    residual: torch.Tensor, dz_hydro: torch.Tensor | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Zonal/time-mean residual [levels-1, lat] and, if dz_hydro given, its
    percentage-of-layer-thickness field, from LOCAL per-grid-cell inputs."""
    zm = residual.nanmean(dim=(0, 3)).cpu().numpy()
    pct = None
    if dz_hydro is not None:
        zm_scale = dz_hydro.abs().nanmean(dim=(0, 3)).cpu().numpy()
        pct = np.abs(zm) / zm_scale * 100
    return zm, pct


def _area_weighted_rms_pct(pct: np.ndarray, lat: np.ndarray) -> float:
    """cos(lat)-area-weighted RMS of a [levels-1, lat] percentage-error field."""
    weights = np.cos(np.deg2rad(lat))[None, :]
    weights = np.broadcast_to(weights, pct.shape)
    valid = np.isfinite(pct)
    return float(np.sqrt(np.average(pct[valid] ** 2, weights=weights[valid])))


def _setup_hydrostatic_axes(ax, p_mid_hpa: np.ndarray) -> None:
    """Shared axis styling: log-pressure (surface at bottom) x linear latitude."""
    ax.set_yscale("log")
    ax.invert_yaxis()
    candidate_ticks = [1000, 850, 700, 500, 300, 200, 100, 50, 20, 10]
    tick_p = [t for t in candidate_ticks if p_mid_hpa.min() <= t <= p_mid_hpa.max()]
    ax.set_yticks(tick_p)
    ax.set_yticklabels([str(t) for t in tick_p], fontsize=TICK_FONTSIZE)
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    lat_ticks_deg = [-90, -60, -30, 0, 30, 60, 90]
    lat_tick_labels = ["90S", "60S", "30S", "EQ", "30N", "60N", "90N"]
    ax.set_xticks(lat_ticks_deg)
    ax.set_xticklabels(lat_tick_labels, fontsize=TICK_FONTSIZE)
    for lt in lat_ticks_deg:
        ax.axvline(lt, color="0.6", lw=0.5, ls=":", zorder=0)
    ax.set_xlabel("Latitude", fontsize=AXIS_LABEL_FONTSIZE)


def plot_hydrostatic_residual(
    residual: torch.Tensor,
    pressure_levels: list[float],
    title: str,
    out_path: Path,
    dz_hydro: torch.Tensor | None = None,
    pct_contour_levels: tuple[float, ...] = (1, 5),
) -> None:
    """Zonal-mean hydrostatic residual vs. log-pressure and linear latitude.

    `residual` (and `dz_hydro`, if given) are the LOCAL, per-grid-cell values
    returned by hydrostatic_residual ([T, levels-1, lat, lon]) -- the
    zonal/time mean is taken here, after the residual itself was computed
    per grid cell, not on an already-zonally-averaged ta/zg field.

    y-axis: actual pressure in hPa, log-scaled and inverted so the surface
    (highest pressure) is at the bottom and TOA at the top, matching the
    standard atmospheric-profile convention.
    x-axis: linear latitude in degrees, with faint dotted vertical
    reference lines at the standard 90S/60S/30S/EQ/30N/60N/90N latitudes.

    If `dz_hydro` is given, overlays black contour lines of the relative
    error |zonal-mean residual| / |zonal-mean dz_hydro| * 100% at
    `pct_contour_levels`, labeled directly on the lines, and annotates the
    area-weighted root-mean-square (RMS) of that percentage field in the
    corner -- the single number a reader would otherwise have to eyeball
    from the contours, giving one summary of how large the residual is
    relative to the local layer thickness, not just its absolute size in
    meters.

    `title` is accepted for backward compatibility with existing call sites
    but deliberately not rendered -- no in-figure titles, see
    analysis/README.md; convey scenario/model identity via the caption
    instead.
    """
    del title
    p = np.array(pressure_levels)
    p_mid_hpa = (p[:-1] + p[1:]) / 2 / 100.0  # hPa, same order as pressure_levels (surface-first)
    zm, pct = _hydrostatic_zonal_mean_and_pct(residual, dz_hydro)
    lat = np.linspace(-90, 90, zm.shape[-1])
    X, Y = np.meshgrid(lat, p_mid_hpa)

    v_limit = np.nanmax(np.abs(zm))
    fig, ax = plt.subplots(figsize=(8, 5))
    cf = ax.contourf(X, Y, zm, levels=20, cmap="RdBu_r", extend="both", vmin=-v_limit, vmax=v_limit)

    if pct is not None:
        # Light smoothing (latitude axis only -- 144 points vs. 16 pressure
        # levels, so noise is mostly cell-to-cell jitter along latitude) so
        # the contour lines read as a few clean curves instead of many small
        # noisy closed loops; the underlying color fill (zm, above) is left
        # unsmoothed.
        pct_smooth = scipy.ndimage.gaussian_filter1d(pct, sigma=2.0, axis=-1, mode="nearest")
        cs = ax.contour(X, Y, pct_smooth, levels=list(pct_contour_levels), colors="black", linewidths=1.0)
        ax.clabel(cs, inline=True, fontsize=9, fmt="%g%%")
        rms_pct = _area_weighted_rms_pct(pct, lat)
        # Top-right corner (near TOA, away from the crowded surface contours
        # at the bottom): annotate against the axes in data space, but pin
        # to the y-axis top since pressure is log-scaled/inverted.
        ax.text(
            0.98,
            0.97,
            f"RMS: {rms_pct:.1f}%",
            transform=ax.transAxes,
            fontsize=TICK_FONTSIZE,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.6"),
        )

    _setup_hydrostatic_axes(ax, p_mid_hpa)
    ax.set_ylabel("Pressure (hPa)", fontsize=AXIS_LABEL_FONTSIZE)
    cbar = fig.colorbar(cf, ax=ax, label="Hydrostatic residual (m)")
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label("Hydrostatic residual (m)", fontsize=AXIS_LABEL_FONTSIZE)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_hydrostatic_residual_decades(
    residual: torch.Tensor,
    pressure_levels: list[float],
    out_path: Path,
    dz_hydro: torch.Tensor | None = None,
    pct_contour_levels: tuple[float, ...] = (1, 5, 10),
    n_panels: int = 4,
    window_months: int = 120,
    panel_labels: list[str] | None = None,
) -> None:
    """Small-multiples version of plot_hydrostatic_residual across a rollout.

    Splits `residual` (and `dz_hydro`) into `n_panels` non-overlapping,
    evenly-spaced `window_months`-long windows (first window starts at the
    rollout's first month, last window ends at its last month, with any
    remainder split as a gap between windows) and plots one
    zonal/time-mean-in-that-window panel per window, side by side on a
    shared color scale -- so the reader can see whether the residual grows
    or drifts over a long autoregressive rollout rather than only a single
    whole-rollout average.
    """
    p = np.array(pressure_levels)
    p_mid_hpa = (p[:-1] + p[1:]) / 2 / 100.0
    n_t = residual.shape[0]
    window_months = min(window_months, n_t // n_panels)
    if n_panels > 1:
        starts = np.linspace(0, n_t - window_months, n_panels).astype(int)
    else:
        starts = [0]

    panels = []
    for s in starts:
        r_window = residual[s : s + window_months]
        dz_window = dz_hydro[s : s + window_months] if dz_hydro is not None else None
        zm, pct = _hydrostatic_zonal_mean_and_pct(r_window, dz_window)
        panels.append((zm, pct))

    lat = np.linspace(-90, 90, panels[0][0].shape[-1])
    x = np.sin(np.deg2rad(lat))
    X, Y = np.meshgrid(x, p_mid_hpa)
    v_limit = max(np.nanmax(np.abs(zm)) for zm, _ in panels)

    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5), sharey=True)
    axes = np.atleast_1d(axes)
    if panel_labels is None:
        panel_labels = [f"Months {s}-{s + window_months}" for s in starts]

    for ax, (zm, pct), label in zip(axes, panels, panel_labels):
        cf = ax.contourf(X, Y, zm, levels=20, cmap="RdBu_r", extend="both", vmin=-v_limit, vmax=v_limit)
        if pct is not None:
            cs = ax.contour(X, Y, pct, levels=list(pct_contour_levels), colors="black", linewidths=0.8)
            ax.clabel(cs, inline=True, fontsize=7, fmt="%g%%")
            rms_pct = _area_weighted_rms_pct(pct, lat)
            ax.text(
                0.02,
                0.03,
                f"RMS: {rms_pct:.1f}%",
                transform=ax.transAxes,
                fontsize=TICK_FONTSIZE - 1,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.6"),
            )
        _setup_hydrostatic_axes(ax, p_mid_hpa)
        ax.set_title(label, fontsize=TICK_FONTSIZE)

    axes[0].set_ylabel("Pressure (hPa)", fontsize=AXIS_LABEL_FONTSIZE)
    cbar = fig.colorbar(cf, ax=list(axes), label="Hydrostatic residual (m)", shrink=0.9)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label("Hydrostatic residual (m)", fontsize=AXIS_LABEL_FONTSIZE)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_energy_budget(diag: dict[str, torch.Tensor], title: str, out_path: Path) -> None:
    """Global E_atm / E_oce time series and the dE/dt - F_net conservation residual."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    t = np.arange(diag["e_atm_global"].shape[0])
    axes[0].plot(t, diag["e_atm_global"].numpy(), label="$E_{atm}$")
    axes[0].plot(t, diag["e_oce_global"].numpy(), label="$E_{oce}$")
    axes[0].set_ylabel("Global energy (J)")
    axes[0].legend()
    axes[0].set_title(title)

    axes[1].axhline(0, color="k", linewidth=0.5)
    axes[1].plot(t[:-1], diag["budget_residual"].numpy(), color="tab:red")
    axes[1].set_ylabel(r"$dE/dt - F_{net}$ (W)")
    axes[1].set_xlabel("Month")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_energy_series_comparison(
    pred_series: torch.Tensor,
    target_series: torch.Tensor,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Single-panel predicted-vs-target global time series (E_atm, E_oce, or residual)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    t = np.arange(pred_series.shape[0])
    if "residual" in ylabel.lower() or "dE/dt" in ylabel:
        ax.axhline(0, color="k", linewidth=0.5)
    ax.plot(t, target_series.numpy(), color="tab:gray", label="target")
    ax.plot(t, pred_series.numpy(), color="tab:blue", label="pred")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Month")
    ax.legend()
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_energy_budget_comparison(
    pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor], title: str, out_path: Path
) -> None:
    """Overlay predicted vs. target global energy trajectories and budget residual.

    Deprecated in favor of plot_energy_series_comparison (one file per
    series, see plot_e_atm_decomposition_comparison's sibling calls in
    main()) -- kept only for any external callers still importing it.

    `title` is accepted for backward compatibility but deliberately not
    rendered -- no in-figure titles, see analysis/README.md. Colors/labels
    use the shared cross-figure convention from analysis/plot_style.py
    (IPSL target always black, AC-SSP always the same red) instead of the
    generic "target"/"pred" this used to say.
    """
    del title
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    t = np.arange(pred["e_atm_global"].shape[0])
    axes[0].plot(t, target["e_atm_global"].numpy(), color=TARGET_COLOR, label="IPSL target")
    axes[0].plot(t, pred["e_atm_global"].numpy(), color=_AC_SSP_COLOR, label=AC_SSP_LABEL)
    axes[0].set_ylabel("$E_{atm}$ (J)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].legend(fontsize=LEGEND_FONTSIZE)
    axes[0].tick_params(axis="both", labelsize=TICK_FONTSIZE)

    axes[1].plot(t, target["e_oce_global"].numpy(), color=TARGET_COLOR, label="IPSL target")
    axes[1].plot(t, pred["e_oce_global"].numpy(), color=_AC_SSP_COLOR, label=AC_SSP_LABEL)
    axes[1].set_ylabel("$E_{oce}$ (J)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].tick_params(axis="both", labelsize=TICK_FONTSIZE)

    axes[2].axhline(0, color="k", linewidth=0.5)
    axes[2].plot(t[:-1], target["budget_residual"].numpy(), color=TARGET_COLOR, label="IPSL target")
    axes[2].plot(t[:-1], pred["budget_residual"].numpy(), color=_AC_SSP_COLOR, label=AC_SSP_LABEL)
    axes[2].set_ylabel(r"$dE/dt - F_{net}$ (W)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[2].set_xlabel("Month", fontsize=AXIS_LABEL_FONTSIZE)
    axes[2].tick_params(axis="both", labelsize=TICK_FONTSIZE)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_e_atm_decomposition_comparison(
    pred_components: dict[str, torch.Tensor],
    target_components: dict[str, torch.Tensor],
    title: str,
    out_path: Path,
) -> None:
    """4-panel predicted-vs-target global time series for E_atm's physical decomposition.

    sensible/potential/kinetic/latent, one panel per term.
    """
    terms = ["sensible", "potential", "kinetic", "latent"]
    fig, axes = plt.subplots(len(terms), 1, figsize=(8, 10), sharex=True)

    t = np.arange(pred_components["sensible"].shape[0])
    for ax, term in zip(axes, terms):
        ax.plot(t, target_components[term].numpy(), color="tab:gray", label="target")
        ax.plot(t, pred_components[term].numpy(), color="tab:blue", label="pred")
        ax.set_ylabel(f"$E_{{{term}}}$ (J)")
    axes[0].legend()
    axes[0].set_title(title)
    axes[-1].set_xlabel("Month")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_spatial_bias(
    atm_pred: np.ndarray,
    atm_target: np.ndarray,
    oce_pred: np.ndarray,
    oce_target: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """2x3 grid: (atmosphere, ocean) x (pred, target, pred-target bias)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    rows = [
        ("$E_{atm}$", atm_pred, atm_target, "J/m$^2$"),
        ("$E_{oce}$", oce_pred, oce_target, "J/m$^2$"),
    ]

    for row, (label, pred_map, target_map, units) in enumerate(rows):
        vmin = min(np.nanmin(pred_map), np.nanmin(target_map))
        vmax = max(np.nanmax(pred_map), np.nanmax(target_map))
        diff = pred_map - target_map
        diff_limit = np.nanmax(np.abs(diff))

        for col, (name, data, cmap, vlo, vhi) in enumerate(
            [
                ("pred", pred_map, "viridis", vmin, vmax),
                ("target", target_map, "viridis", vmin, vmax),
                ("pred - target", diff, "RdBu_r", -diff_limit, diff_limit),
            ]
        ):
            ax = axes[row, col]
            im = ax.imshow(
                data, origin="lower", extent=[0, 360, -90, 90], cmap=cmap, vmin=vlo, vmax=vhi
            )
            ax.set_title(f"{label} {name}")
            if col == 0:
                ax.set_ylabel("Latitude (deg)")
            if row == 1:
                ax.set_xlabel("Longitude (deg)")
            fig.colorbar(im, ax=ax, label=units, shrink=0.8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_supersaturation_comparison(
    pred: dict[str, Any],
    target: dict[str, Any],
    pressure_levels: list[float],
    title: str,
    out_path: Path,
) -> None:
    """Supersaturation frequency (hus > q_sat) vs. pressure level, pred vs. target."""
    p_hpa = np.array(pressure_levels) / 100.0
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(
        target["per_level_frequency"].numpy() * 100,
        p_hpa,
        color="tab:gray",
        marker="o",
        label="target",
    )
    ax.plot(
        pred["per_level_frequency"].numpy() * 100, p_hpa, color="tab:blue", marker="o", label="pred"
    )
    ax.invert_yaxis()
    ax.set_xlabel("Supersaturated cells (%)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_cc_scaling_comparison(
    pred: dict[str, Any], target: dict[str, Any], title: str, out_path: Path
) -> None:
    """ln(hus) vs. ta at the lower-troposphere level, with fitted vs. theoretical C-C slopes."""
    fig, ax = plt.subplots(figsize=(6, 5))

    for label, diag, color in (("target", target, "tab:gray"), ("pred", pred, "tab:blue")):
        ta = diag["ta_mean"].numpy()
        log_hus = diag["log_hus_mean"].numpy()
        ax.scatter(
            ta,
            log_hus,
            s=8,
            color=color,
            alpha=0.5,
            label=f"{label} (fit slope={diag['slope']:.3f}/K)",
        )
        fit_x = np.linspace(ta.min(), ta.max(), 50)
        ax.plot(fit_x, diag["slope"] * fit_x + diag["intercept"], color=color, linewidth=2)

    ax.set_xlabel("Global-mean $T_a$ at 850 hPa (K)")
    ax.set_ylabel(r"$\ln(\overline{hus})$")
    ax.set_title(f"{title} (theoretical slope $\\approx${target['theoretical_slope']:.3f}/K)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def run_diagnostics(
    fields: dict[str, torch.Tensor],
    pressure_levels: list[float],
    depth_levels: list[float],
    level_vars: list[str],
    surface_vars: list[str],
    area: torch.Tensor,
) -> dict[str, torch.Tensor]:
    residual, dz_hydro = hydrostatic_residual(
        fields["level"], pressure_levels, level_vars, return_scale=True
    )

    e_atm_components = atmospheric_column_energy_decomposed(
        fields["level"], fields["surface"], pressure_levels, level_vars, surface_vars
    )
    e_atm = e_atm_components["total"]
    e_oce = ocean_heat_content(fields["lev"], depth_levels)
    net_flux = fields["surface"][:, surface_vars.index("net_flux")]

    e_atm_global = global_sum(e_atm, area)
    e_oce_global = global_sum(e_oce, area)
    net_flux_global = global_sum(net_flux, area)
    e_atm_components_global = {
        component: global_sum(field, area)
        for component, field in e_atm_components.items()
        if component != "total"
    }

    budget_residual = conservation_check(
        e_atm_global, e_oce_global, net_flux_global, AVG_SECONDS_PER_MONTH
    )

    supersaturation = supersaturation_diagnostic(fields["level"], pressure_levels, level_vars)
    cc_scaling = clausius_clapeyron_scaling(fields["level"], pressure_levels, level_vars, area)

    return dict(
        hydrostatic_residual=residual,
        hydrostatic_dz_hydro=dz_hydro,
        e_atm=e_atm,
        e_atm_components=e_atm_components,
        e_atm_components_global=e_atm_components_global,
        e_oce=e_oce,
        e_atm_global=e_atm_global,
        e_oce_global=e_oce_global,
        net_flux_global=net_flux_global,
        budget_residual=budget_residual,
        supersaturation=supersaturation,
        cc_scaling=cc_scaling,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_name",
        help="rollout dir name, e.g. "
        "forcing_dropout_no_random_lt_no_aero_0_12_10_1_1020_1_val_0_step-step=050000.ckpt_ema",
    )
    parser.add_argument(
        "--config-module",
        default=None,
        help="Hydra module config name; inferred from experiment_name if omitted",
    )
    parser.add_argument("--member", type=int, default=0, help="ensemble member index to load")
    parser.add_argument(
        "--default-months",
        type=int,
        default=120,
        help="months of target-only diagnostics to compute when predicted rollouts are missing",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="optional .pt path to save computed diagnostics"
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="if set, save hydrostatic/energy-budget plots to this directory",
    )
    parser.add_argument(
        "--last-decade-months",
        type=int,
        default=120,
        help="number of trailing months to average for the spatial pred/target/bias maps",
    )
    args = parser.parse_args()

    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)

    config_module = args.config_module or infer_config_module(args.experiment_name)
    print(f"Using config module: {config_module}")
    _, scratch, cfg, dataset = initialize_notebook(domain="val", config_module=config_module)

    pressure_levels = cfg.module.pressure_levels
    depth_levels = cfg.module.depth_levels
    level_vars = list(cfg.module.level_variables)
    surface_vars = list(cfg.module.surface_variables)
    area = grid_cell_area()

    results = {}
    for scenario in SCENARIO_DOMAIN:
        exp_dir = (
            Path(scratch)
            / "generated_data"
            / experiment_dir_for_scenario(args.experiment_name, scenario)
        )

        try:
            pred = load_predicted(exp_dir, dataset, member=args.member)
            n_t = pred["surface"].shape[0]
        except FileNotFoundError as exc:
            print(f"[{scenario}] skipping predicted diagnostics: {exc}")
            pred = None
            n_t = args.default_months

        # predicted step i corresponds to ground truth month i+1 (lead_time_months=1)
        target = load_target(scenario, start=1, stop=1 + n_t, dataset_path=cfg.module.dataset_path)

        for label, fields in (("target", target), ("pred", pred)):
            if fields is None:
                continue
            diag = run_diagnostics(
                fields, pressure_levels, depth_levels, level_vars, surface_vars, area
            )
            print(
                f"[{scenario}/{label}] hydrostatic residual MAE: "
                f"{diag['hydrostatic_residual'].abs().nanmean():.2f} m"
            )
            print(
                f"[{scenario}/{label}] energy budget residual: "
                f"mean={diag['budget_residual'].mean():.3e} W, "
                f"std={diag['budget_residual'].std():.3e} W"
            )
            print(
                f"[{scenario}/{label}] supersaturation (hus > q_sat): "
                f"{diag['supersaturation']['overall_frequency'].item() * 100:.3f}% of cells, "
                f"mean excess where violated="
                f"{diag['supersaturation']['mean_violation_magnitude'].item():.2e} kg/kg"
            )
            print(
                f"[{scenario}/{label}] C-C scaling at "
                f"{pressure_levels[diag['cc_scaling']['level_index']] / 100:.0f} hPa: "
                f"fit slope={diag['cc_scaling']['slope']:.4f}/K, "
                f"theoretical={diag['cc_scaling']['theoretical_slope']:.4f}/K"
            )
            results[f"{scenario}_{label}"] = diag

            if args.plot_dir:
                plot_hydrostatic_residual(
                    diag["hydrostatic_residual"],
                    pressure_levels,
                    title=f"{scenario} ({label})",
                    out_path=args.plot_dir / f"hydrostatic_{scenario}_{label}.png",
                    dz_hydro=diag["hydrostatic_dz_hydro"],
                )
                plot_energy_budget(
                    diag,
                    title=f"{scenario} ({label})",
                    out_path=args.plot_dir / f"energy_budget_{scenario}_{label}.png",
                )

        if f"{scenario}_pred" in results and f"{scenario}_target" in results:
            pred_diag = results[f"{scenario}_pred"]
            target_diag = results[f"{scenario}_target"]

            n_t = min(pred_diag["e_atm"].shape[0], target_diag["e_atm"].shape[0])
            mae = {
                "e_atm": energy_component_mae(
                    pred_diag["e_atm"][:n_t], target_diag["e_atm"][:n_t], area
                ),
                "e_oce": energy_component_mae(
                    pred_diag["e_oce"][:n_t], target_diag["e_oce"][:n_t], area
                ),
            }
            for component in ("sensible", "potential", "kinetic", "latent"):
                mae[f"e_atm_{component}"] = energy_component_mae(
                    pred_diag["e_atm_components"][component][:n_t],
                    target_diag["e_atm_components"][component][:n_t],
                    area,
                )
            results[f"{scenario}_mae"] = mae
            for name, series in mae.items():
                print(f"[{scenario}] {name} MAE (pred vs target): {series.mean():.3e} J/m^2")

        if args.plot_dir and f"{scenario}_pred" in results and f"{scenario}_target" in results:
            pred_diag = results[f"{scenario}_pred"]
            target_diag = results[f"{scenario}_target"]

            plot_energy_series_comparison(
                pred_diag["e_atm_global"],
                target_diag["e_atm_global"],
                ylabel="$E_{atm}$ (J)",
                title=scenario,
                out_path=args.plot_dir / f"energy_budget_eatm_{scenario}.png",
            )
            plot_energy_series_comparison(
                pred_diag["e_oce_global"],
                target_diag["e_oce_global"],
                ylabel="$E_{oce}$ (J)",
                title=scenario,
                out_path=args.plot_dir / f"energy_budget_eoce_{scenario}.png",
            )
            plot_energy_series_comparison(
                pred_diag["budget_residual"],
                target_diag["budget_residual"],
                ylabel=r"$dE/dt - F_{net}$ (W)",
                title=scenario,
                out_path=args.plot_dir / f"energy_budget_residual_{scenario}.png",
            )
            plot_e_atm_decomposition_comparison(
                pred_diag["e_atm_components_global"],
                target_diag["e_atm_components_global"],
                title=f"{scenario}: $E_{{atm}}$ decomposition",
                out_path=args.plot_dir / f"energy_budget_eatm_decomposition_{scenario}.png",
            )

            plot_supersaturation_comparison(
                pred_diag["supersaturation"],
                target_diag["supersaturation"],
                pressure_levels,
                title=f"{scenario}: supersaturation frequency",
                out_path=args.plot_dir / f"supersaturation_{scenario}.png",
            )

            plot_cc_scaling_comparison(
                pred_diag["cc_scaling"],
                target_diag["cc_scaling"],
                title=f"{scenario}: Clausius-Clapeyron scaling",
                out_path=args.plot_dir / f"cc_scaling_{scenario}.png",
            )

            n_decade = min(args.last_decade_months, pred_diag["e_atm"].shape[0])
            plot_spatial_bias(
                pred_diag["e_atm"][-n_decade:].mean(dim=0).numpy(),
                target_diag["e_atm"][-n_decade:].mean(dim=0).numpy(),
                pred_diag["e_oce"][-n_decade:].mean(dim=0).numpy(),
                target_diag["e_oce"][-n_decade:].mean(dim=0).numpy(),
                title=f"{scenario}: last {n_decade} months mean",
                out_path=args.plot_dir / f"spatial_bias_{scenario}.png",
            )

    if args.out:
        torch.save(results, args.out)
        print(f"Saved diagnostics to {args.out}")


if __name__ == "__main__":
    main()
