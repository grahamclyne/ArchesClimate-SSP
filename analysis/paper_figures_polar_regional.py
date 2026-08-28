r"""Regional-accuracy figures focused on the poles.

Antarctic/Tropics/Arctic temporal trends, plus early-vs-late-rollout spatial
bias maps.

Companion to paper_figures_rmse_plot.py (region-banded RMSE over time) and
paper_figures_global_mean.py (Robinson decadal-difference maps) -- built for
the polar-dynamics comparison across the flatloss/spherepad/flatpf (det) and
flatloss/halfloss/amploss (flow) intervention runs, but generic to any
`models` dict in the usual paper-figures YAML schema (see
paper_figures_table.py for the schema).

For one scenario (--scenario, matching a `key` in the config), for the
variable set by `variable` in the config (default "tas"), produces:

  - polar_regional_timeseries_{scenario}.pdf: one column per region (REGIONS
    below -- Antarctic 0-65, Tropics 65-85, Arctic 85-144, lat indices),
    annual-mean regional value over time, IPSL target vs. each model
    (ensemble mean +/- spread).
  - polar_spatial_diff_{scenario}.pdf: Robinson-projection grid, one row for
    the first `window_months` months of the rollout and one row for the last
    `window_months`, one column per model -- time-mean(model) -
    time-mean(target) over that window.

Usage:
    python analysis/paper_figures_polar_regional.py \\
        --config analysis/configs/paper_figures_table_polar_comparison.yaml \\
        --scenario 434
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

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run, load_target
from analysis.paper_figures_rmse_plot import monthly_to_yearly

# MODEL_COLORS (paper_figures_rmse_plot.py) only has 6 entries -- this
# comparison can have more models than that (currently 8: 4 det + 4 flow),
# and cycling MODEL_COLORS with `% len(MODEL_COLORS)` would silently give two
# different models the same color. Extend locally rather than growing the
# shared list, which other figures rely on staying at exactly 6.
_EXTRA_COLORS = ["saddlebrown", "deeppink", "olive", "slategray"]

REGIONS = {
    "Antarctic (0-65)": slice(0, 65),
    "Tropics (65-85)": slice(65, 85),
    "Arctic (85-144)": slice(85, 144),
}


def _align_to_target(pred_monthly, target_monthly):
    """pred_monthly: (n_members, T, lat, lon); target_monthly: (T_tgt, lat, lon).

    Rollout index k (0-based) predicts target row k+1 (row 0 is the initial
    condition fed into the model, never itself predicted) -- same convention
    as compare_runs.py::main() and paper_figures_585_analysis.py.
    """
    t_common = min(pred_monthly.shape[1], target_monthly.shape[0] - 1)
    return pred_monthly[:, :t_common], target_monthly[1 : 1 + t_common]


def plot_regional_timeseries(
    scenario_cfg,
    target_annual,
    target_first,
    model_annuals,
    model_colors,
    model_labels,
    variable,
    unit,
    output_dir,
):
    region_names = list(REGIONS.keys())
    fig, axes = plt.subplots(1, len(region_names), figsize=(6 * len(region_names), 5), sharex=True)

    start_year = target_first.year if target_first is not None else 2015

    for col, region_name in enumerate(region_names):
        ax = axes[col]
        region_slice = REGIONS[region_name]

        tgt_region = target_annual[..., region_slice, :].nanmean(dim=(-1, -2))  # (n_years,)
        years_tgt = np.arange(start_year, start_year + tgt_region.shape[0])
        ax.plot(years_tgt, tgt_region.numpy(), color="black", lw=2, linestyle="-", zorder=3)

        for key, annual in model_annuals.items():
            region = annual[..., region_slice, :].nanmean(dim=(-1, -2))  # (n_members, n_years)
            mean = region.mean(dim=0).numpy()
            std = (
                region.std(dim=0, unbiased=True).numpy()
                if region.shape[0] > 1
                else np.zeros_like(mean)
            )
            years = start_year + 1 + np.arange(mean.shape[0])
            ax.plot(years, mean, color=model_colors[key], lw=2)
            ax.fill_between(years, mean - std, mean + std, color=model_colors[key], alpha=0.2, lw=0)

        ax.set_title(region_name, fontsize=15)
        ax.grid(alpha=0.3, linestyle=":")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.set_xlabel("Year", fontsize=13)
        if col == 0:
            ax.set_ylabel(f"{variable} ({unit})", fontsize=13)

    handles = [mlines.Line2D([], [], color="black", lw=2, label=f"IPSL {scenario_cfg.label}")]
    handles += [
        mlines.Line2D([], [], color=model_colors[key], lw=2, label=model_labels[key])
        for key in model_annuals
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=12,
        frameon=True,
        bbox_to_anchor=(0.5, 1.08),
    )
    plt.tight_layout()

    out_path = output_dir / f"polar_regional_timeseries_{scenario_cfg.key}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved regional timeseries figure to {out_path}")


def _plot_robinson(ax, data, vlim):
    import cartopy.crs as ccrs
    import matplotlib.colors as mcolors
    import numpy.ma as ma

    lat = np.linspace(-90, 90, 144)
    lon = np.linspace(-2, 358, 144)
    lon2d, lat2d = np.meshgrid(lon, lat)
    masked_data = ma.masked_invalid(data)

    vlim = vlim or 1.0
    norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)
    levels = np.linspace(-vlim, vlim, 13)

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


def plot_spatial_diff(
    scenario_cfg,
    target_monthly,
    model_monthlies,
    model_labels,
    variable,
    unit,
    window_months,
    output_dir,
):
    import cartopy.crs as ccrs

    model_keys = list(model_monthlies.keys())
    row_defs = [("First", slice(0, window_months)), ("Last", slice(-window_months, None))]

    fig, axes = plt.subplots(
        2,
        len(model_keys),
        figsize=(5.5 * len(model_keys), 8.5),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
    )
    if len(model_keys) == 1:
        axes = axes.reshape(2, 1)

    # Pre-compute every cell's diff first so all panels can share one vlim/
    # colorbar -- otherwise each panel's own min/max makes the color scale
    # incomparable across models (a small bias can look as "red" as a large
    # one in a different panel).
    diffs = {}
    for row_label, window in row_defs:
        target_window_mean = target_monthly[window].nanmean(dim=0)  # (144, 144)
        for key in model_keys:
            pred = model_monthlies[key]  # (n_members, T, 144, 144)
            pred_window_mean = pred[:, window].nanmean(dim=(0, 1))  # (144, 144)
            diffs[(row_label, key)] = (pred_window_mean - target_window_mean).numpy()

    all_finite = np.concatenate([d[np.isfinite(d)] for d in diffs.values()])
    vlim = np.abs(all_finite).max() if all_finite.size else 1.0

    cs = None
    for row, (row_label, window) in enumerate(row_defs):
        for col, key in enumerate(model_keys):
            ax = axes[row, col]
            cs = _plot_robinson(ax, diffs[(row_label, key)], vlim)

            if row == 0:
                ax.set_title(model_labels[key], fontsize=14, pad=10)
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    f"{row_label} {window_months}mo",
                    va="center",
                    ha="center",
                    rotation="vertical",
                    fontsize=13,
                    transform=ax.transAxes,
                )

    fig.suptitle(f"{scenario_cfg.label}: model - IPSL, {variable}", fontsize=15, y=1.02)
    plt.subplots_adjust(wspace=0.05, hspace=0.15, bottom=0.18)
    cbar_ax = fig.add_axes([0.3, 0.04, 0.4, 0.025])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"{variable} bias ({unit})", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    out_path = output_dir / f"polar_spatial_diff_{scenario_cfg.key}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved spatial diff figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    parser.add_argument("--scenario", required=True, help="Scenario key to plot (e.g. '434').")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    window_months = cfg.get("polar_window_months", 120)
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfg = next(s for s in cfg.scenarios if str(s.key) == str(args.scenario))

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    surface_variables = list(hydra_cfg.module.surface_variables)
    variable = cfg.get("variable", "tas")
    unit = "K" if variable == "tas" else ""
    var_idx = surface_variables.index(variable)
    data_mean = train_dataset.data_mean["surface"].squeeze(1)
    data_std = train_dataset.data_std["surface"].view(-1)

    realization = scenario_cfg.target.get("realization", "r1i1p1f1")
    target_monthly, target_time = load_target(
        scratch,
        realization,
        scenario_cfg.target.scenario,
        var_idx,
        dataset_path=hydra_cfg.module.dataset_path,
    )
    target_first = target_time[0] if target_time is not None else None

    from analysis.paper_figures_rmse_plot import MODEL_COLORS

    palette = MODEL_COLORS + _EXTRA_COLORS
    if len(scenario_cfg.models) > len(palette):
        raise ValueError(
            f"{len(scenario_cfg.models)} models but only {len(palette)} colors in "
            "MODEL_COLORS + _EXTRA_COLORS -- add more colors rather than let them cycle "
            "and collide."
        )
    model_colors = {key: palette[i] for i, key in enumerate(scenario_cfg.models)}
    model_labels = {key: scenario_cfg.models[key].get("label", key) for key in scenario_cfg.models}

    model_monthlies, model_annuals = {}, {}
    for key, model_cfg in scenario_cfg.models.items():
        members = [
            load_run(f"{scratch}/generated_data/{name}", var_idx, data_mean, data_std)
            for name in model_cfg.runs
        ]
        t_min = min(m.shape[1] for m in members)
        raw = torch.cat([m[:, :t_min] for m in members], dim=0)  # (n_members, T, 144, 144)
        pred_aligned, target_aligned = _align_to_target(raw, target_monthly)
        model_monthlies[key] = pred_aligned
        model_annuals[key] = monthly_to_yearly(pred_aligned, time_dim=1)

    t_common = min(m.shape[1] for m in model_monthlies.values())
    target_monthly_aligned = target_monthly[1 : 1 + t_common]
    model_monthlies = {k: v[:, :t_common] for k, v in model_monthlies.items()}
    target_annual = monthly_to_yearly(target_monthly[None], time_dim=1)[0]

    plot_regional_timeseries(
        scenario_cfg,
        target_annual,
        target_first,
        model_annuals,
        model_colors,
        model_labels,
        variable,
        unit,
        output_dir,
    )
    plot_spatial_diff(
        scenario_cfg,
        target_monthly_aligned,
        model_monthlies,
        model_labels,
        variable,
        unit,
        window_months,
        output_dir,
    )


if __name__ == "__main__":
    main()
