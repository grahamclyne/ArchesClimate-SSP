"""Abrupt-4xCO2 global-mean tas/pr/ocean heat content: does the energy-score
model keep the slow, ocean-mediated transient ramp?

Compares deterministic_damip_pf2_fixed vs.
deterministic_damip_pf4_energy_score_w050_80_10_0. Both rollouts must have
been generated with inference.save_all_vars=True so their "lev" (thetao)
chunks exist on disk (the default abrupt-4xCO2 rollouts only save
"surface") -- see analysis/long_rollout.py.

Usage:
    python -m analysis.abrupt4xco2_det_vs_es
"""

import matplotlib.pyplot as plt
import torch
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run, load_run_lev
from analysis.energy_hydrostatic_balance import grid_cell_area, ocean_heat_content
from analysis.plot_style import MODEL_COLORS

# Larger than the shared plot_style.py defaults -- this figure is a
# 1-textwidth-wide, 3-panel spread, so the shared defaults (tuned for
# smaller/denser multi-panel figures) read too small at print size.
AXIS_LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18

PLOT_VARS = [
    ("tas", "Global-mean tas", "K"),
    ("pr", "Global-mean pr", "kg m$^{-2}$ s$^{-1}$"),
]
OCEAN_PANEL_LABEL = "Upper-ocean heat content"
OCEAN_PANEL_UNIT = "J m$^{-2}$"

# Same target realization used elsewhere in the paper for abrupt-4xCO2
# (fig:abrupt4xco2's IPSL comparison).
TARGET_REALIZATION = "r2i1p1f1"
TARGET_LABEL = "IPSL target"
MAX_YEARS = 86  # matches fig:abrupt4xco2's convention; ~model rollout length

# Display labels/colors from the shared cross-figure convention (analysis/
# plot_style.py) -- no internal config/checkpoint names in the plot itself
# (those live in the paper's caption text, not here).
RUNS = {
    "Deterministic": (
        "deterministic_damip_pf2_fixed_12_10_1_1020_1_abrupt_0_step-step=022000.ckpt_ema",
        MODEL_COLORS["Deterministic"],
    ),
    "AC-SSP": (
        "deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_1020_1_abrupt_0_step-step=040000.ckpt_ema",
        MODEL_COLORS["AC-SSP"],
    ),
}


def area_weighted_mean(field, lats):
    w = torch.cos(torch.deg2rad(lats)).view(1, 1, -1, 1).expand_as(field)
    mask = ~torch.isnan(field)
    w_eff = w * mask.float()
    return (torch.nan_to_num(field) * w_eff).sum(dim=(-2, -1)) / w_eff.sum(dim=(-2, -1))


def to_annual_series(x):
    n_years = x.shape[-1] // 12
    return x[..., : n_years * 12].reshape(*x.shape[:-1], n_years, 12).mean(dim=-1)


def _plot_ocean_panel(ax, target_td, hydra_cfg, scratch, train_dataset):
    """Global-mean upper-ocean heat content time series, same convention as
    the surface-variable panels above (target in black, each run's ensemble
    mean +/- std band). Requires each run's rollout to have been generated
    with inference.save_all_vars=True (rollout_lev_*.pt chunks on disk)."""
    depth_levels = [float(d) for d in hydra_cfg.module.depth_levels]
    data_mean_lev = train_dataset.data_mean["lev"].squeeze(0)
    data_std_lev = train_dataset.data_std["lev"].view(-1)
    lats = torch.linspace(-90, 90, hydra_cfg.module.get("lat_dim", 144))

    target_lev = target_td["lev"][0]  # (T, n_depth, lat, lon)
    target_lev = torch.where(target_lev.abs() < 1e30, target_lev, torch.nan)
    target_ohc = ocean_heat_content(target_lev, depth_levels)[None]  # (1, T, lat, lon)
    target_annual = to_annual_series(area_weighted_mean(target_ohc, lats))[0][:MAX_YEARS]
    target_years = list(range(1, target_annual.shape[0] + 1))
    ax.plot(target_years, target_annual, color="black", lw=2.2, label=TARGET_LABEL, zorder=5)

    for label, (run_name, color) in RUNS.items():
        run_dir = f"{scratch}/generated_data/{run_name}"
        lev = load_run_lev(run_dir, data_mean_lev, data_std_lev)  # (n_members, T, n_depth, lat, lon)
        ohc = torch.stack([ocean_heat_content(m, depth_levels) for m in lev], dim=0)  # (n_members, T, lat, lon)
        gm_annual = to_annual_series(area_weighted_mean(ohc, lats))  # (n_members, n_years)
        years = list(range(1, gm_annual.shape[-1] + 1))
        mean = gm_annual.mean(dim=0)
        ax.plot(years, mean, color=color, lw=1.8, label=label)
        if gm_annual.shape[0] > 1:
            std = gm_annual.std(dim=0)
            ax.fill_between(years, mean - std, mean + std, color=color, alpha=0.15, lw=0)

    ax.set_xlabel("Rollout year", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f"{OCEAN_PANEL_LABEL} ({OCEAN_PANEL_UNIT})", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(alpha=0.3, linestyle=":")


def main():
    scratch = "/scratch/gclyne"
    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train",
        config_module="deterministic_damip_pf4_energy_score_w050",
        dataloader="cmip_random_lead_times",
    )
    surface_variables = list(hydra_cfg.module.surface_variables)
    data_mean = train_dataset.data_mean["surface"].squeeze(1)
    data_std = train_dataset.data_std["surface"].view(-1)
    lats = torch.linspace(-90, 90, hydra_cfg.module.get("lat_dim", 144))
    dataset_path = hydra_cfg.module.dataset_path

    target_path = f"{scratch}/{dataset_path}/{TARGET_REALIZATION}_abrupt-4xCO2_interpolation.memmap"
    target_td = TensorDict.load_memmap(target_path)

    n_panels = len(PLOT_VARS) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    for ax, (var, title, unit) in zip(axes, PLOT_VARS):
        var_idx = surface_variables.index(var)

        target_surface = target_td["surface"][var_idx][None]  # (1, T, lat, lon)
        target_surface = torch.where(target_surface.abs() < 1e30, target_surface, torch.nan)
        target_annual = to_annual_series(area_weighted_mean(target_surface, lats))[0][:MAX_YEARS]
        target_years = list(range(1, target_annual.shape[0] + 1))
        ax.plot(target_years, target_annual, color="black", lw=2.2, label=TARGET_LABEL, zorder=5)

        for label, (run_name, color) in RUNS.items():
            run_dir = f"{scratch}/generated_data/{run_name}"
            field = load_run(run_dir, var_idx, data_mean, data_std)  # (n_members, T, lat, lon)
            gm_annual = to_annual_series(area_weighted_mean(field, lats))  # (n_members, n_years)
            years = list(range(1, gm_annual.shape[-1] + 1))
            mean = gm_annual.mean(dim=0)
            ax.plot(years, mean, color=color, lw=1.8, label=label)
            if gm_annual.shape[0] > 1:
                std = gm_annual.std(dim=0)
                ax.fill_between(years, mean - std, mean + std, color=color, alpha=0.15, lw=0)
        ax.set_xlabel("Rollout year", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f"{title} ({unit})", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.grid(alpha=0.3, linestyle=":")

    _plot_ocean_panel(axes[-1], target_td, hydra_cfg, scratch, train_dataset)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=LEGEND_FONTSIZE,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.08),
    )
    fig.tight_layout(w_pad=3.5)
    fig.subplots_adjust(wspace=0.35)

    out_dir = "plots/es_vs_det_abrupt_transient"
    import os

    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/abrupt4xco2_tas_pr_det_vs_es.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    png_path = f"{out_dir}/abrupt4xco2_tas_pr_det_vs_es.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"Saved to {out_path} and {png_path}")


if __name__ == "__main__":
    main()
