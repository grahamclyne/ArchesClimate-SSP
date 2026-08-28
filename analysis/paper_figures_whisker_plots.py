r"""Forcing-ablation "delta from baseline" whisker plots across pressure level and region.

Per ablated model.

Same idea as paper_figures_forced_response.py's global-mean tas delta plot
(each ablation's trajectory minus that same model's own all-forcings
baseline), but resolved across the atmospheric column (pressure level) and
region instead of collapsed to a single global-mean time series. For one
scenario (--scenario) and one atmospheric level variable (--variable,
default "ta"), computes, per ensemble member of each ablated model, the
area-weighted mean of (member's final-decade time-mean $-$ the baseline
model's own ensemble-mean final-decade time-mean), at every pressure level,
within each of three latitude regions (North/Tropics/South, matching
paper_figures_rmse_plot.py's REGIONS). Plots mean +/- std (across members)
per model/pressure-level as an errorbar (or an "x" marker if either exceeds
100, matching the whisker-plot overflow-marker convention), one subplot per
region. The model dict entry named "full" (see config schema below) is
treated as the all-forcings baseline and used only as the reference, not
plotted as its own row.

Config schema: identical scenarios/models/runs structure as
paper_figures_table_example.yaml; requires one model key literally named
"full" per scenario to serve as the baseline.

Usage:
    python analysis/paper_figures_whisker_plots.py \\
        --config analysis/configs/paper_figures_table_example.yaml \\
        --scenario 534 --variable ta
"""

import argparse
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run_level_all
from analysis.paper_figures_rmse_plot import REGIONS
from analysis.plot_style import ABLATION_COLORS

OVERFLOW_THRESHOLD = 100  # matches the notebook's "x" marker cutoff for runaway mean/std


def load_model_member_last_decade_means(
    scratch, run_names, var_idx, data_mean_level, data_std_level, final_decade_months
):
    """Final-decade time-mean per ensemble member, pooled across every run in `run_names`.

    Returns a list of (n_plevels, 144, 144) tensors, one per member.
    """
    means = []
    for name in run_names:
        run_dir = os.path.join(scratch, "generated_data", name)
        raw = load_run_level_all(
            run_dir, data_mean_level, data_std_level
        )  # (n_members, T, n_vars, n_plevels, 144, 144)
        var_raw = raw[:, :, var_idx]  # (n_members, T, n_plevels, 144, 144)
        member_means = var_raw[:, -final_decade_months:].nanmean(
            dim=1
        )  # (n_members, n_plevels, 144, 144)
        means.extend(member_means)
    return means


def region_weighted_delta(pred_map, baseline_map, lat_slice):
    """pred_map/baseline_map: (n_plevels, 144, 144) -> (n_plevels,).

    Lat-weighted signed mean of (pred - baseline) over the region -- the
    atmospheric-column analogue of paper_figures_forced_response.py's
    "delta from all-forcings baseline".
    """
    lat_all = torch.linspace(-90, 90, 144)
    lats = lat_all[lat_slice]
    weights = torch.cos(torch.deg2rad(lats)).view(1, -1, 1)

    pred_region = pred_map[:, lat_slice, :]
    baseline_region = baseline_map[:, lat_slice, :]
    diff = pred_region - baseline_region
    mask = ~torch.isnan(diff)
    effective_weights = weights * mask.float()
    weighted_diff = torch.nan_to_num(diff, nan=0.0) * effective_weights
    return weighted_diff.sum(dim=(-2, -1)) / effective_weights.sum(dim=(-2, -1))  # (n_plevels,)


def compute_model_region_delta(member_means, baseline_mean, lat_slice):
    """member_means: list of (n_plevels, 144, 144) -> (n_plevels, n_members)."""
    deltas = [region_weighted_delta(m, baseline_mean, lat_slice) for m in member_means]
    return torch.stack(deltas, dim=1)


def make_whisker_df(tensors_by_model, pressure_levels):
    """tensors_by_model: {model: (n_plevels, n_samples)} -> long-form DataFrame."""
    rows = []
    for model, tensor in tensors_by_model.items():
        for i, plevel in enumerate(pressure_levels):
            p_name = str(int(plevel / 100))
            values = tensor[i].flatten().numpy()
            rows.append(pd.DataFrame({"Value": values, "Pressure Level": p_name, "Model": model}))
    return pd.concat(rows)


def plot_profile(ax, df, model_colors, models):
    """Vertical-profile style: pressure on y (surface at bottom), delta on x.

    One line per model with a shaded +/-std band instead of per-level errorbars.
    """
    plevels = list(df["Pressure Level"].unique())
    y = np.arange(len(plevels))
    for model in models:
        means, stds = [], []
        for level in plevels:
            subset = df[(df["Pressure Level"] == level) & (df["Model"] == model)]["Value"]
            means.append(subset.mean())
            stds.append(subset.std())
        means, stds = np.array(means), np.array(stds)
        ax.plot(means, y, color=model_colors[model], lw=2, marker="o", markersize=4)
        ax.fill_betweenx(y, means - stds, means + stds, color=model_colors[model], alpha=0.2, lw=0)

    ax.axvline(0, color="slategray", linestyle="--", linewidth=1.5, alpha=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(plevels)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(alpha=0.3, linestyle=":")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_heatmap(fig, ax, df, models, model_labels, vmax=None):
    """Pressure level x model grid, color = signed delta mean (std not shown)."""
    plevels = list(df["Pressure Level"].unique())
    grid = np.zeros((len(plevels), len(models)))
    for i, level in enumerate(plevels):
        for j, model in enumerate(models):
            subset = df[(df["Pressure Level"] == level) & (df["Model"] == model)]["Value"]
            grid[i, j] = subset.mean()

    if vmax is None:
        vmax = np.abs(grid).max()
    # origin="lower": index 0 (surface/1000hPa, first entry in plevels) at
    # the bottom, matching plot_profile/plot_whiskers (regular ax.plot/
    # errorbar, y increasing upward) and standard meteorological convention.
    # imshow defaults to origin="upper", which would put the surface at the
    # top instead.
    im = ax.imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_yticks(range(len(plevels)))
    ax.set_yticklabels(plevels)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([model_labels[m] for m in models], rotation=45, ha="right")
    ax.tick_params(axis="both", which="major", labelsize=11)
    return im


def plot_whiskers(ax, df, model_colors, models, offsets):
    plevels = df["Pressure Level"].unique()
    for i, level in enumerate(plevels):
        for j, model in enumerate(models):
            subset = df[(df["Pressure Level"] == level) & (df["Model"] == model)]["Value"]
            if subset.empty:
                continue
            mean, std = subset.mean(), subset.std()
            y_pos = i + offsets[j]
            if abs(mean) > OVERFLOW_THRESHOLD or std > OVERFLOW_THRESHOLD:
                ax.plot(
                    0, y_pos, marker="x", color=model_colors[model], markersize=8, markeredgewidth=2
                )
            else:
                ax.errorbar(
                    mean,
                    y_pos,
                    xerr=std,
                    fmt="o",
                    color=model_colors[model],
                    capsize=4,
                    elinewidth=2,
                    markersize=7,
                    markeredgecolor="white",
                )

    ax.axvline(0, color="slategray", linestyle="--", linewidth=1.5, alpha=0.6, zorder=0)
    ax.set_yticks(range(len(plevels)))
    ax.set_yticklabels(plevels)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.grid(axis="x", linestyle=":", color="#ecf0f1", zorder=0)
    for spine in ("top", "right", "left", "bottom"):
        ax.spines[spine].set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    parser.add_argument("--scenario", required=True, help="Scenario key to plot (e.g. '534').")
    parser.add_argument(
        "--variable", default="ta", help="Level variable to plot (e.g. 'ta', 'hus', 'zg')."
    )
    parser.add_argument(
        "--style",
        default="whisker",
        choices=["whisker", "profile", "heatmap"],
        help="Plot style: 'whisker' (original errorbar-per-level), 'profile' "
        "(line + shaded std band, pressure on y), or 'heatmap' (level x model grid).",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    final_decade_years = cfg.get("final_decade_years", 10)
    final_decade_months = final_decade_years * 12
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfg = next(s for s in cfg.scenarios if str(s.key) == str(args.scenario))

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    level_vars = list(hydra_cfg.module.level_variables)
    pressure_levels = [float(p) for p in hydra_cfg.module.pressure_levels]
    var_idx = level_vars.index(args.variable)
    data_mean_level = train_dataset.data_mean["level"]  # (n_vars, n_plevels, 144, 144)
    data_std_level = train_dataset.data_std["level"]  # (n_vars, n_plevels, 1, 1)

    if "full" not in scenario_cfg.models:
        raise ValueError(
            "scenario config must declare a model key literally named 'full' "
            "(the all-forcings baseline) to serve as the delta reference"
        )
    ablation_keys = [k for k in scenario_cfg.models.keys() if k != "full"]
    # Keyed by the config's own ablation key (e.g. "no_co2"), not
    # positionally -- so "No CO2" gets the same color here as it does in
    # paper_figures_forced_response.py's figures, regardless of the order
    # this config happens to list its models in. See analysis/plot_style.py.
    model_colors = {k: ABLATION_COLORS[k] for k in ablation_keys}
    model_labels = {k: scenario_cfg.models[k].get("label", k) for k in ablation_keys}

    model_member_means = {
        key: load_model_member_last_decade_means(
            scratch,
            list(model_cfg.runs),
            var_idx,
            data_mean_level,
            data_std_level,
            final_decade_months,
        )
        for key, model_cfg in scenario_cfg.models.items()
    }
    baseline_mean = torch.stack(model_member_means["full"], dim=0).mean(
        dim=0
    )  # (n_plevels, 144, 144)

    region_names = list(REGIONS.keys())
    dfs = {}
    for region_name in region_names:
        region_slice = REGIONS[region_name]
        tensors_by_model = {
            key: compute_model_region_delta(model_member_means[key], baseline_mean, region_slice)
            for key in ablation_keys
        }
        dfs[region_name] = make_whisker_df(tensors_by_model, pressure_levels)

    if args.style == "heatmap":
        # shared color scale across regions so panels are directly comparable
        vmax = max(
            np.abs(dfs[r].groupby(["Pressure Level", "Model"])["Value"].mean()).max()
            for r in region_names
        )
        fig, axes = plt.subplots(
            1, len(region_names), figsize=(5 * len(region_names), 8), sharey=True
        )
        im = None
        for ax, region_name in zip(axes, region_names):
            im = plot_heatmap(fig, ax, dfs[region_name], ablation_keys, model_labels, vmax=vmax)
        axes[0].set_ylabel("hPa", fontsize=18)
        fig.colorbar(
            im,
            ax=axes,
            label=f"$\\Delta$ {args.variable} vs. all-forcings baseline",
            shrink=0.8,
            pad=0.02,
        )
        out_path = output_dir / f"heatmap_delta_plots_{scenario_cfg.key}_{args.variable}.pdf"
        fig.savefig(out_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved heatmap delta plot to {out_path}")
        return

    offsets = np.linspace(-0.15, 0.15, len(ablation_keys))
    fig, axes = plt.subplots(1, len(region_names), figsize=(16, 10), sharey=True)

    for ax, region_name in zip(axes, region_names):
        if args.style == "profile":
            plot_profile(ax, dfs[region_name], model_colors, ablation_keys)
        else:
            plot_whiskers(ax, dfs[region_name], model_colors, ablation_keys, offsets)
        ax.set_xlabel(f"$\\Delta$ {args.variable} vs. all-forcings baseline", fontsize=20)
    axes[0].set_ylabel("hPa", fontsize=20)

    legend_bars = [
        mpatches.Patch(color=model_colors[m], label=model_labels[m]) for m in ablation_keys
    ]
    fig.legend(
        handles=legend_bars,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=len(ablation_keys),
        frameon=False,
        fontsize=20,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    style_tag = "profile" if args.style == "profile" else "whisker"
    out_path = output_dir / f"{style_tag}_delta_plots_{scenario_cfg.key}_{args.variable}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {args.style} delta plot to {out_path}")


if __name__ == "__main__":
    main()
