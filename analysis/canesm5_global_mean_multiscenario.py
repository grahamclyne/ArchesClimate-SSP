r"""Global-mean tas rollout, CanESM5 native-grid model vs. CanESM5 target, multi-scenario.

For multiple scenarios (SSP4-3.4, SSP5-3.4-over, SSP5-8.5) on one figure.

Unlike paper_figures_585_analysis.py / paper_figures_global_mean.py (both
IPSL-grid, and the latter hardcoded to exactly {434, 534}), this is a
standalone script for the CanESM5 native-grid (64x128) model family --
config_module must resolve to a canesm5_native module (lat_dim=64,
lon_dim=128) and dataset_path memmap_filled_in_canesm5_native.

Plots each scenario's model ensemble as a mean +/- 1 std filled band, and
the CanESM5 target as a solid line, all on one shared axes (annual global
mean tas, 2015-2100 for 434/585, 2040-2100 for 534).

Usage:
    python -m analysis.canesm5_global_mean_multiscenario \\
        --config-module canesm5_damip_pf4_energy_score_w050 \\
        --scratch /scratch/gclyne
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run, to_annual
from analysis.plot_style import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
)

SCENARIOS = {
    "434": {
        "label": "SSP4-3.4",
        "target_scenario": "ssp434",
        "run_dir": (
            "canesm5_damip_pf4_energy_score_w050_0_12_10_1_1020_1_val_0_step-step=022000.ckpt_ema"
        ),
        "color": "darkgreen",
    },
    "534": {
        "label": "SSP5-3.4-over",
        "target_scenario": "ssp534-over",
        "run_dir": (
            "canesm5_damip_pf4_energy_score_w050_0_12_10_1_730_1_val3_0_step-step=022000.ckpt_ema"
        ),
        "color": "steelblue",
    },
    "585": {
        "label": "SSP5-8.5",
        "target_scenario": "ssp585",
        "run_dir": (
            "canesm5_damip_pf4_energy_score_w050_0_12_10_1_1020_1_test_0_step-step=022000.ckpt_ema"
        ),
        "color": "firebrick",
    },
}


def load_target_annual(scratch, dataset_path, realization, scenario, var_idx):
    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    surface = td["surface"][var_idx]  # (T, lat, lon)
    surface = torch.where(surface.abs() < 1e30, surface, torch.nan)
    # CanESM5's memmap stores time as cftime.DatetimeNoLeap (non-Gregorian
    # calendar) objects, which pandas can't parse into a DatetimeIndex
    # directly -- only .year is needed downstream, so pull that straight off
    # the cftime object instead of round-tripping through pandas.
    start_year = None
    if "time" in td.keys():
        start_year = np.asarray(td["time"])[0].year
    annual = to_annual(surface.unsqueeze(0))[0]  # (n_years, lat, lon)
    return annual, start_year


def plot_band(ax, annual_gm, dates, color, label):
    mean = annual_gm.mean(dim=0)
    std = annual_gm.std(dim=0)
    ax.plot(dates, mean, color=color, lw=1.5, label=label)
    ax.fill_between(dates, mean - std, mean + std, color=color, alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-module", default="canesm5_damip_pf4_energy_score_w050")
    parser.add_argument(
        "--scenarios", default="434,585", help="Comma-separated scenario keys to include."
    )
    parser.add_argument(
        "--output", default="plots/canesm5_global_mean_multiscenario/canesm5_global_mean.pdf"
    )
    args = parser.parse_args()
    scenario_keys = args.scenarios.split(",")

    work, scratch, cfg, train_dataset = initialize_notebook(
        domain="val",
        config_module=args.config_module,
        dataloader="cmip_random_lead_times_canesm5_native",
    )
    surface_variables = list(cfg.module.surface_variables)
    var_idx = surface_variables.index("tas")
    data_mean = train_dataset.data_mean["surface"].squeeze(1)
    data_std = train_dataset.data_std["surface"].view(-1)
    dataset_path = cfg.module.dataset_path

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    for key in scenario_keys:
        s = SCENARIOS[key]
        run_path = f"{scratch}/generated_data/{s['run_dir']}"
        if not os.path.isdir(run_path):
            print(f"-- skipping {s['label']}: no rollout at {run_path}")
            continue
        members = load_run(run_path, var_idx, data_mean, data_std)  # (n_members, T, lat, lon)
        model_annual = to_annual(members)  # (n_members, n_years, lat, lon)
        model_gm = model_annual.nanmean(dim=(-1, -2))  # (n_members, n_years)

        target_annual, start_year = load_target_annual(
            scratch,
            dataset_path,
            "r1i1p1f1",
            s["target_scenario"],
            var_idx,
        )
        target_gm = target_annual.nanmean(dim=(-1, -2)).unsqueeze(0)  # (1, n_years)

        n_years = min(model_gm.shape[1], target_gm.shape[1])
        model_gm, target_gm = model_gm[:, :n_years], target_gm[:, :n_years]
        start_year = start_year if start_year is not None else 2015
        dates = pd.date_range(start=f"{start_year}-01-01", periods=n_years, freq="YS")

        plot_band(ax, model_gm, dates, s["color"], f"{s['label']} (model, n={model_gm.shape[0]})")
        ax.plot(
            dates,
            target_gm[0],
            color=s["color"],
            lw=1.5,
            linestyle=":",
            label=f"{s['label']} (CanESM5 target)",
        )

    ax.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Global-mean tas (K)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
