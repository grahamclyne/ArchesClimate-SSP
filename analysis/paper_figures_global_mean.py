r"""Generate the two global-mean summary figures comparing SSP4-3.4 vs SSP5-3.4.

Adapted from analysis/global_mean_figure.ipynb:
  - cell 14: "Climate_Variables_Triple_Layer" -- mean/max/min 5-year-binned
    time series over the Tropics band (lat 56-86), for tas (land-masked),
    ta@50hPa, net_surface_flux, and pr. Solid=IPSL, dotted=model, dashed=
    MESMER-M (tas panel only); one color per scenario.
  - cell 18: "decadal_difference_2090" -- Robinson-projection maps of the
    final-decade mean difference (MESMER-IPSL for tas, model-IPSL for all
    four variables), one row per scenario.

Both figures are inherently a side-by-side SSP4-3.4/SSP5-3.4 comparison, so
(unlike paper_figures_rmse_plot.py, which is per-scenario) this script needs
BOTH scenario keys "434" and "534" present in the config's `scenarios` list.
Only the first model declared under each scenario's `models` is plotted, to
match the notebook's single-model-per-scenario design.

Usage:
    python analysis/paper_figures_global_mean.py \\
        --config analysis/configs/paper_figures_table_example.yaml
"""

import argparse
import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_mesmer, load_run, load_run_level

SCENARIO_COLORS = {"434": "darkgreen", "534": "blue"}
LAT_MIN, LAT_MAX = 56, 86  # "Tropics" band, matches paper_figures_rmse_plot.py's REGIONS
PLOT_VARS = [
    # (key, display title, unit)
    ("tas_land", r"$\mathit{tas}$ (Land Only)", "K"),
    ("ta50", r"$\mathit{ta}$ (50 hPa)", "K"),
    ("net_flux", r"$\mathit{net\_surface\_flux}$", "W/m$^2$"),
    ("pr", r"$\mathit{pr}$", "kg m$^{-2}$ s$^{-1}$"),
]


def monthly_to_yearly(t, time_dim=-3):
    """(..., T, lat, lon) -> (..., T // 12, lat, lon), trimming the remainder."""
    T = t.shape[time_dim]
    n_years = T // 12
    idx = [slice(None)] * t.ndim
    idx[time_dim] = slice(None, n_years * 12)
    t = t[tuple(idx)]
    shape = list(t.shape)
    shape[time_dim : time_dim + 1] = [n_years, 12]
    return t.reshape(shape).mean(time_dim + 1)


def monthly_to_5yr(t, time_dim=-3):
    """(..., T, lat, lon) -> (..., T // 60, lat, lon), trimming the remainder."""
    T = t.shape[time_dim]
    n_bins = T // 60
    idx = [slice(None)] * t.ndim
    idx[time_dim] = slice(None, n_bins * 60)
    t = t[tuple(idx)]
    shape = list(t.shape)
    shape[time_dim : time_dim + 1] = [n_bins, 60]
    return t.reshape(shape).mean(time_dim + 1)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_target_multivar_monthly(
    scratch, realization, scenario, ta_var_idx, ta_plevel_idx, dataset_path="memmap_filled_in"
):
    """Returns ({"tas_land", "ta50", "net_flux", "pr"} -> (T, 144, 144)), time index."""
    import numpy as np
    import pandas as pd

    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    surface = td["surface"]  # (n_vars, T, 144, 144)
    level = td["level"]  # (n_vars, T, n_plevels, 144, 144)
    surface = torch.where(surface.abs() < 1e30, surface, torch.nan)
    level = torch.where(level.abs() < 1e30, level, torch.nan)
    time = pd.DatetimeIndex(np.asarray(td["time"])) if "time" in td.keys() else None
    data = {
        "tas_land": surface[0],
        "ta50": level[ta_var_idx, :, ta_plevel_idx],
        "net_flux": surface[6],
        "pr": surface[2],
    }
    return data, time


def _cat_members(tensors):
    t_min = min(t.shape[1] for t in tensors)
    return torch.cat([t[:, :t_min] for t in tensors], dim=0)


def load_model_multivar_monthly(
    scratch,
    run_names,
    ta_var_idx,
    ta_plevel_idx,
    data_mean_surface,
    data_std_surface,
    data_mean_level,
    data_std_level,
):
    """Returns {"tas_land", "ta50", "net_flux", "pr"} -> (n_members, T, 144, 144)."""
    run_dirs = [os.path.join(scratch, "generated_data", name) for name in run_names]
    return {
        "tas_land": _cat_members(
            [load_run(d, 0, data_mean_surface, data_std_surface) for d in run_dirs]
        ),
        "net_flux": _cat_members(
            [load_run(d, 6, data_mean_surface, data_std_surface) for d in run_dirs]
        ),
        "pr": _cat_members([load_run(d, 2, data_mean_surface, data_std_surface) for d in run_dirs]),
        "ta50": _cat_members(
            [
                load_run_level(d, ta_var_idx, ta_plevel_idx, data_mean_level, data_std_level)
                for d in run_dirs
            ]
        ),
    }


def load_scenario(
    scenario_cfg,
    scratch,
    ta_var_idx,
    ta_plevel_idx,
    data_mean_surface,
    data_std_surface,
    data_mean_level,
    data_std_level,
    dataset_path="memmap_filled_in",
):
    target_monthly, time_index = load_target_multivar_monthly(
        scratch,
        scenario_cfg.target.get("realization", "r1i1p1f1"),
        scenario_cfg.target.scenario,
        ta_var_idx,
        ta_plevel_idx,
        dataset_path=dataset_path,
    )
    model_key = next(iter(scenario_cfg.models))
    model_cfg = scenario_cfg.models[model_key]
    model_monthly = load_model_multivar_monthly(
        scratch,
        list(model_cfg.runs),
        ta_var_idx,
        ta_plevel_idx,
        data_mean_surface,
        data_std_surface,
        data_mean_level,
        data_std_level,
    )

    mesmer_tas = None
    if "mesmer" in scenario_cfg and time_index is not None:
        mesmer_tas = load_mesmer(scenario_cfg.mesmer.path, "tas", time_index)

    if mesmer_tas is not None:
        # Land mask from MESMER-M's own nan pattern (True = ocean); matches
        # paper_figures_rmse_plot.py's per-timestep-loop land-masking, kept
        # the same way here since direct 3-D boolean broadcasting was found
        # unreliable in that script.
        land_mask = mesmer_tas[0, 0].isnan()
        for t in range(target_monthly["tas_land"].shape[0]):
            target_monthly["tas_land"][t][land_mask] = float("nan")
        for m in range(model_monthly["tas_land"].shape[0]):
            for t in range(model_monthly["tas_land"].shape[1]):
                model_monthly["tas_land"][m, t][land_mask] = float("nan")

    return {
        "label": scenario_cfg.label,
        "model_label": model_cfg.get("label", model_key),
        "target_monthly": target_monthly,
        "model_monthly": model_monthly,
        "mesmer_tas": mesmer_tas,
        "time_index": time_index,
    }


# --------------------------------------------------------------------------
# Figure 1: triple-layer mean/max/min over the Tropics band
# --------------------------------------------------------------------------


def _plot_band(ax_mean, ax_max, ax_min, band, years, color, label, linestyle):
    """band: (n_members_or_realisations, n_bins, lat_band, lon)."""
    spatial = band.nanmean(dim=(-1, -2))  # (n, n_bins)
    mean = spatial.nanmean(dim=0)
    std = spatial.std(dim=0)
    n = years[: mean.shape[0]]
    ax_mean.plot(n, mean.numpy(), color=color, linestyle=linestyle, lw=2, label=label, zorder=3)
    ax_mean.fill_between(
        n, (mean - std).numpy(), (mean + std).numpy(), color=color, alpha=0.2, lw=0
    )

    n_bins = band.shape[1]
    flat = band.permute(1, 0, 2, 3).reshape(n_bins, -1)  # (n_bins, n * lat_band * lon)
    filled = torch.where(flat.isnan(), torch.nanmean(flat, dim=1, keepdim=True), flat)
    vmax = filled.max(dim=1).values
    vmin = filled.min(dim=1).values
    ax_max.plot(n, vmax.numpy(), color=color, linestyle=linestyle, lw=2, alpha=0.85)
    ax_min.plot(n, vmin.numpy(), color=color, linestyle=linestyle, lw=2, alpha=0.85)


def plot_triple_layer(per_scenario, output_dir):
    fig, axes = plt.subplots(3, len(PLOT_VARS), figsize=(7 * len(PLOT_VARS), 9), sharex=True)

    for col, (key, title, unit) in enumerate(PLOT_VARS):
        ax_mean, ax_max, ax_min = axes[0, col], axes[1, col], axes[2, col]
        ax_mean.set_title(title, fontsize=16, pad=14)

        for scen_key in ("434", "534"):
            scen = per_scenario[scen_key]
            color = SCENARIO_COLORS[scen_key]

            target = scen["target_monthly"][key][None]  # (1, T, lat, lon)
            model = scen["model_monthly"][key]  # (n_members, T, lat, lon)
            t_min = min(target.shape[1], model.shape[1])
            target, model = target[:, :t_min], model[:, :t_min]

            start_year = scen["time_index"].year[0] if scen["time_index"] is not None else 2015
            target_5yr = monthly_to_5yr(target)[..., LAT_MIN:LAT_MAX, :]
            model_5yr = monthly_to_5yr(model)[..., LAT_MIN:LAT_MAX, :]
            years = start_year + 5 * np.arange(target_5yr.shape[1])

            _plot_band(
                ax_mean, ax_max, ax_min, target_5yr, years, color, f"IPSL {scen['label']}", "-"
            )
            _plot_band(
                ax_mean,
                ax_max,
                ax_min,
                model_5yr,
                years,
                color,
                f"{scen['model_label']} {scen['label']}",
                ":",
            )

            if key == "tas_land" and scen["mesmer_tas"] is not None:
                mesmer = scen["mesmer_tas"][:, :t_min]
                mesmer_5yr = monthly_to_5yr(mesmer)[..., LAT_MIN:LAT_MAX, :]
                _plot_band(
                    ax_mean,
                    ax_max,
                    ax_min,
                    mesmer_5yr,
                    years,
                    color,
                    f"MESMER-M {scen['label']}",
                    "--",
                )

        ax_mean.set_ylabel(f"Mean\n({unit})", fontsize=13)
        ax_max.set_ylabel(f"Max\n({unit})", fontsize=13)
        ax_min.set_ylabel(f"Min\n({unit})", fontsize=13)
        for ax in (ax_mean, ax_max, ax_min):
            ax.grid(alpha=0.3, linestyle=":")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.tick_params(axis="both", labelsize=12)
        ax_min.set_xlabel("Year", fontsize=13)

    color_handles = [
        mlines.Line2D([], [], color=SCENARIO_COLORS[k], lw=3, label=per_scenario[k]["label"])
        for k in ("434", "534")
    ]
    model_label = per_scenario["434"]["model_label"]
    style_handles = [
        mlines.Line2D([], [], color="black", linestyle="-", lw=2, label="IPSL"),
        mlines.Line2D([], [], color="black", linestyle=":", lw=2, label=model_label),
        mlines.Line2D([], [], color="black", linestyle="--", lw=2, label="MESMER-M"),
    ]
    leg1 = fig.legend(
        handles=color_handles,
        title="Scenario",
        loc="upper left",
        bbox_to_anchor=(0.05, 1.08),
        ncol=2,
        fontsize=13,
        title_fontsize=14,
    )
    leg2 = fig.legend(
        handles=style_handles,
        title="Model",
        loc="upper right",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        fontsize=13,
        title_fontsize=14,
    )
    fig.add_artist(leg1)
    del leg2  # kept alive by the figure; local ref unused beyond this point

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "global_mean_triple_layer.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved triple-layer figure to {out_path}")


# --------------------------------------------------------------------------
# Figure 2: decadal difference maps (Robinson projection)
# --------------------------------------------------------------------------


def _final_decade_field(scen, source, var_key, final_decade_years):
    if source == "mesmer":
        arr = scen["mesmer_tas"]
        if arr is None:
            return None
    elif source == "target":
        arr = scen["target_monthly"][var_key][None]
    elif source == "model":
        arr = scen["model_monthly"][var_key]
    else:
        raise ValueError(source)
    annual = monthly_to_yearly(arr)  # (member, n_years, 144, 144)
    return annual[:, -final_decade_years:].nanmean(dim=(0, 1))


def _plot_robinson(ax, data, vmin, vmax):
    import matplotlib.colors as mcolors
    import numpy.ma as ma

    lat = np.linspace(-90, 90, 144)
    lon = np.linspace(-2, 358, 144)
    lon2d, lat2d = np.meshgrid(lon, lat)
    masked_data = ma.masked_invalid(data)

    limit = max(abs(vmin), abs(vmax)) or 1.0
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit)
    levels = np.linspace(-limit, limit, 13)

    import cartopy.crs as ccrs

    cs = ax.contourf(
        lon2d,
        lat2d,
        masked_data,
        levels=levels,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        norm=norm,
        extend="both",
    )
    ax.coastlines()
    return cs


def plot_decadal_difference(per_scenario, output_dir, final_decade_years):
    import cartopy.crs as ccrs

    col_defs = [
        ("tas (MESMER-M - IPSL)", "mesmer", "target", "tas_land", "K"),
        ("tas (model - IPSL)", "model", "target", "tas_land", "K"),
        ("ta (50 hPa, model - IPSL)", "model", "target", "ta50", "K"),
        ("net_surface_flux (model - IPSL)", "model", "target", "net_flux", "W/m$^2$"),
        ("pr (model - IPSL)", "model", "target", "pr", "kg m$^{-2}$ s$^{-1}$"),
    ]
    scen_keys = ("434", "534")

    fig, axes = plt.subplots(
        2,
        len(col_defs),
        figsize=(6 * len(col_defs), 10),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
    )

    for row, scen_key in enumerate(scen_keys):
        scen = per_scenario[scen_key]
        for col, (title, src_a, src_b, var_key, unit) in enumerate(col_defs):
            ax = axes[row, col]
            a = _final_decade_field(scen, src_a, var_key, final_decade_years)
            b = _final_decade_field(scen, src_b, var_key, final_decade_years)
            if a is None or b is None:
                ax.set_visible(False)
                continue
            diff = (a - b).numpy()
            finite = diff[np.isfinite(diff)]
            vlim = np.abs(finite).max() if finite.size else 1.0
            cs = _plot_robinson(ax, diff, -vlim, vlim)

            if row == 0:
                ax.set_title(title, fontsize=13, pad=12)
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    scen["label"],
                    va="center",
                    ha="center",
                    rotation="vertical",
                    fontsize=14,
                    transform=ax.transAxes,
                )

            cbar = fig.colorbar(cs, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06)
            cbar.set_label(unit, fontsize=9)
            cbar.ax.tick_params(labelsize=8)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    out_path = output_dir / "global_mean_decadal_difference.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved decadal-difference figure to {out_path}")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    final_decade_years = cfg.get("final_decade_years", 10)
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfgs = {str(s.key): s for s in cfg.scenarios}
    missing = [k for k in ("434", "534") if k not in scenario_cfgs]
    if missing:
        raise ValueError(
            f"paper_figures_global_mean.py needs both '434' and '534' scenarios in "
            f"the config (these two figures are a side-by-side SSP4-3.4/SSP5-3.4 "
            f"comparison); missing {missing}."
        )

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    level_variables = list(hydra_cfg.module.level_variables)
    pressure_levels = list(hydra_cfg.module.pressure_levels)
    ta_var_idx = level_variables.index("ta")
    ta_plevel_idx = min(range(len(pressure_levels)), key=lambda i: abs(pressure_levels[i] - 5000.0))

    data_mean_surface = train_dataset.data_mean["surface"].squeeze(1)
    data_std_surface = train_dataset.data_std["surface"].view(-1)
    data_mean_level = train_dataset.data_mean["level"]  # (n_vars, n_plevels, 144, 144)
    data_std_level = train_dataset.data_std["level"].view(
        len(level_variables), len(pressure_levels)
    )

    per_scenario = {
        key: load_scenario(
            scenario_cfgs[key],
            scratch,
            ta_var_idx,
            ta_plevel_idx,
            data_mean_surface,
            data_std_surface,
            data_mean_level,
            data_std_level,
            dataset_path=hydra_cfg.module.dataset_path,
        )
        for key in ("434", "534")
    }

    plot_triple_layer(per_scenario, output_dir)
    plot_decadal_difference(per_scenario, output_dir, final_decade_years)


if __name__ == "__main__":
    main()
