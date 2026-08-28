r"""Generate the "ocean RMSE across time" depth-level grid from a scenario.

Adapted from analysis/global_mean_rollouts.ipynb (cell 7, "Ocean RMSE" --
that cell referenced undefined variables (MODEL_COLOUR, REGION_STYLE), so it
never actually ran as written; this is a working port). For one scenario
(identified by --scenario, matching a `key` in the same YAML config used by
paper_figures_table.py), plots per-region, per-depth-level, per-year RMSE
(mean +/- std across ensemble members/seeds) of ocean temperature (thetao)
against the IPSL target. One line per model declared in the scenario's
`models` dict. No MESMER-M row -- MESMER-M is a land-surface-temperature-only
emulator and has no ocean-depth analogue.

Usage:
    python analysis/paper_figures_ocean_rmse_plot.py \\
        --config analysis/configs/paper_figures_table_example.yaml \\
        --scenario 534
"""

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run_lev
from analysis.paper_figures_rmse_plot import (
    REGIONS,
    compute_region_rmse,
    monthly_to_yearly,
)

MODEL_COLORS = ["cyan", "green", "blue", "darkorange", "magenta", "gold"]


def load_target_lev_yearly(scratch, realization, scenario, dataset_path="memmap_filled_in"):
    """Returns (yearly tensor (T_yrs, n_depth, 144, 144), first timestamp, last timestamp)."""
    import numpy as np
    import pandas as pd

    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    lev = td["lev"][0]  # (T, n_depth, 144, 144) -- single ocean variable (thetao)
    lev = torch.where(lev.abs() < 1e30, lev, torch.nan)
    time = pd.DatetimeIndex(np.asarray(td["time"])) if "time" in td.keys() else None
    return (
        monthly_to_yearly(lev, time_dim=0),
        time[0] if time is not None else None,
        time[-1] if time is not None else None,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    parser.add_argument("--scenario", required=True, help="Scenario key to plot (e.g. '534').")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or __import__("os").environ.get("SCRATCH", "/scratch/gclyne")
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfg = next(s for s in cfg.scenarios if str(s.key) == str(args.scenario))

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    depth_levels = [round(float(d), 1) for d in hydra_cfg.module.depth_levels]
    data_mean_lev = train_dataset.data_mean["lev"].squeeze(0)  # (n_depth, 144, 144)
    data_std_lev = train_dataset.data_std["lev"].view(-1)  # (n_depth,)

    target_yr, target_first, target_last = load_target_lev_yearly(
        scratch,
        scenario_cfg.target.get("realization", "r1i1p1f1"),
        scenario_cfg.target.scenario,
        dataset_path=hydra_cfg.module.dataset_path,
    )  # (T_yrs, n_depth, 144, 144)

    model_colors = {
        key: MODEL_COLORS[i % len(MODEL_COLORS)] for i, key in enumerate(scenario_cfg.models)
    }
    model_labels = {key: scenario_cfg.models[key].get("label", key) for key in scenario_cfg.models}
    model_yrs = {}
    for key, model_cfg in scenario_cfg.models.items():
        members = [
            load_run_lev(f"{scratch}/generated_data/{name}", data_mean_lev, data_std_lev)
            for name in model_cfg.runs
        ]
        t_min = min(m.shape[1] for m in members)
        raw = torch.cat([m[:, :t_min] for m in members], dim=0)  # (n_seeds, T, n_depth, 144, 144)
        model_yrs[key] = monthly_to_yearly(raw, time_dim=1)  # (n_seeds, T_yrs, n_depth, 144, 144)

    T_yrs = min([target_yr.shape[0]] + [v.shape[1] for v in model_yrs.values()])
    model_yrs = {k: v[:, :T_yrs] for k, v in model_yrs.items()}
    target_yr = target_yr[:T_yrs]
    start_year = target_first.year if target_first is not None else 2100 - T_yrs
    yrs = np.arange(start_year, start_year + T_yrs)

    region_list = list(REGIONS.keys())
    n_depths = len(depth_levels)
    fig, axes = plt.subplots(n_depths, len(region_list), figsize=(16, 2.2 * n_depths), sharey=False)
    fig.suptitle(
        f"Ocean Temperature RMSE: Per-Seed Mean ± Std vs. IPSL {scenario_cfg.label} Target",
        fontsize=13,
        y=1.01,
    )

    for d_idx, depth in enumerate(depth_levels):
        for r_idx, region in enumerate(region_list):
            ax = axes[d_idx, r_idx]
            region_slice = REGIONS[region]
            tgt = target_yr[:, d_idx]

            for key, pred_all in model_yrs.items():
                pred = pred_all[:, :, d_idx]
                r = compute_region_rmse(pred, tgt, region_slice)  # (n_seeds, T)
                mean, std = r.mean(0).numpy(), r.std(0).numpy()
                t = yrs[: len(mean)]
                ax.plot(t, mean, color=model_colors[key], lw=1.5)
                ax.fill_between(t, mean - std, mean + std, color=model_colors[key], alpha=0.2, lw=0)

            ax.grid(alpha=0.3, linestyle=":")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            ax.set_ylabel("RMSE (K)", fontsize=12)
            if d_idx == 0:
                ax.set_title(region, fontsize=13)
            if d_idx == n_depths - 1:
                ax.set_xlabel("Year", fontsize=12)
            if r_idx == 0:
                ax.text(
                    -0.35,
                    0.5,
                    f"{depth} m",
                    transform=ax.transAxes,
                    fontsize=13,
                    va="center",
                    rotation=90,
                )

    handles = [
        mlines.Line2D([], [], color=model_colors[key], lw=2, label=model_labels[key])
        for key in scenario_cfg.models
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=13,
        frameon=True,
        bbox_to_anchor=(0.5, 1.04),
    )
    plt.tight_layout()

    out_path = output_dir / f"rmse_ocean_regional_{scenario_cfg.key}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved ocean RMSE plot to {out_path}")


if __name__ == "__main__":
    main()
