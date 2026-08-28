r"""Heatmap of global RMSE (vs. the real CMIP target) across pressure level.

One panel per atmospheric level variable, for the baseline ("full") model only.

Unlike paper_figures_whisker_plots.py (which compares ablations' delta from
the baseline model), this compares one model directly against ground truth.
Each variable gets its own subplot and its own color scale/colorbar, since
magnitudes differ by orders of magnitude across variables (zg's RMSE in
meters dwarfs ta's in Kelvin on any shared scale). Global (full-sphere) RMSE
only, not broken out by region. Ensemble-mean-vs-target RMSE only (no
per-member std -- a heatmap cell can't show mean+std the way a
profile/whisker plot can).

Config schema: identical scenarios/models/runs structure as
paper_figures_table_example.yaml; requires a model key literally named
"full" per scenario (the model to evaluate).

Usage:
    python analysis/paper_figures_rmse_heatmap.py \\
        --config analysis/configs/paper_figures_table_example.yaml \\
        --scenario 534 --variables ta,zg,hus,ua,va
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run_level_all


def load_target_last_decade_mean(
    scratch, realization, scenario, var_idx, final_decade_months, dataset_path="memmap_filled_in"
):
    """Final-decade time-mean of one level variable, at every pressure level.

    Returns tensor of shape (n_plevels, 144, 144).
    """
    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    level = td["level"][var_idx]  # (T, n_plevels, 144, 144)
    level = torch.where(level.abs() < 1e30, level, torch.nan)
    return level[-final_decade_months:].nanmean(dim=0)


def load_model_ensemble_mean_last_decade(
    scratch, run_names, var_idx, data_mean_level, data_std_level, final_decade_months
):
    """Ensemble-mean (across every member of every run in run_names) final-decade time-mean.

    Returns tensor of shape (n_plevels, 144, 144).
    """
    means = []
    for name in run_names:
        run_dir = os.path.join(scratch, "generated_data", name)
        raw = load_run_level_all(
            run_dir, data_mean_level, data_std_level
        )  # (n_members, T, n_vars, n_plevels, lat, lon)
        var_raw = raw[:, :, var_idx]
        member_means = var_raw[:, -final_decade_months:].nanmean(
            dim=1
        )  # (n_members, n_plevels, lat, lon)
        means.extend(member_means)
    return torch.stack(means, dim=0).mean(dim=0)  # (n_plevels, lat, lon)


def region_weighted_rmse(pred_map, target_map, lat_slice):
    """pred_map/target_map: (n_plevels, 144, 144) -> (n_plevels,).

    Lat-weighted RMSE over the region.
    """
    lat_all = torch.linspace(-90, 90, 144)
    lats = lat_all[lat_slice]
    weights = torch.cos(torch.deg2rad(lats)).view(1, -1, 1)

    pred_region = pred_map[:, lat_slice, :]
    target_region = target_map[:, lat_slice, :]
    mask = ~torch.isnan(pred_region - target_region)
    effective_weights = weights * mask.float()
    sq_err = torch.nan_to_num((pred_region - target_region) ** 2, nan=0.0)
    weighted_mse = (sq_err * effective_weights).sum(dim=(-2, -1)) / effective_weights.sum(
        dim=(-2, -1)
    )
    return torch.sqrt(weighted_mse)  # (n_plevels,)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    parser.add_argument("--scenario", required=True, help="Scenario key to plot (e.g. '534').")
    parser.add_argument(
        "--variables",
        default="ta,zg,hus,ua,va",
        help="Comma-separated level variables to include as heatmap columns.",
    )
    parser.add_argument("--model-key", default="full", help="Which model dict entry to evaluate.")
    args = parser.parse_args()
    variables = args.variables.split(",")

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    final_decade_years = cfg.get("final_decade_years", 10)
    final_decade_months = final_decade_years * 12
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfg = next(s for s in cfg.scenarios if str(s.key) == str(args.scenario))
    if args.model_key not in scenario_cfg.models:
        raise ValueError(
            f"model key {args.model_key!r} not found in scenario {args.scenario!r}'s models"
        )
    run_names = list(scenario_cfg.models[args.model_key].runs)

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    level_vars = list(hydra_cfg.module.level_variables)
    pressure_levels = [float(p) for p in hydra_cfg.module.pressure_levels]
    plevel_labels = [str(int(p / 100)) for p in pressure_levels]
    data_mean_level = train_dataset.data_mean["level"]  # (n_vars, n_plevels, 144, 144)
    data_std_level = train_dataset.data_std["level"]  # (n_vars, n_plevels, 1, 1)

    # global (full-sphere) RMSE per variable, per pressure level -- one column
    # vector per variable, each on its own color scale (magnitudes differ by
    # orders of magnitude across variables -- zg dominates any shared scale).
    global_slice = slice(0, 144)
    rmse_by_var = {}
    for var in variables:
        var_idx = level_vars.index(var)
        target_mean = load_target_last_decade_mean(
            scratch,
            scenario_cfg.target.get("realization", "r1i1p1f1"),
            scenario_cfg.target.scenario,
            var_idx,
            final_decade_months,
            dataset_path=hydra_cfg.module.dataset_path,
        )
        pred_mean = load_model_ensemble_mean_last_decade(
            scratch,
            run_names,
            var_idx,
            data_mean_level,
            data_std_level,
            final_decade_months,
        )
        rmse_by_var[var] = region_weighted_rmse(
            pred_mean, target_mean, global_slice
        ).numpy()  # (n_plevels,)

    fig, axes = plt.subplots(1, len(variables), figsize=(2.2 * len(variables), 8), sharey=True)
    if len(variables) == 1:
        axes = [axes]
    for ax, var in zip(axes, variables):
        grid = rmse_by_var[var].reshape(-1, 1)  # (n_plevels, 1)
        # origin="lower": index 0 (surface/1000hPa) at the bottom, matching
        # standard meteorological convention (imshow defaults to
        # origin="upper", which would put the surface at the top instead).
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=grid.max(), origin="lower")
        ax.set_yticks(range(len(pressure_levels)))
        ax.set_yticklabels(plevel_labels)
        ax.set_xticks([0])
        ax.set_xticklabels([var], fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.15, orientation="horizontal", location="bottom")
    axes[0].set_ylabel("hPa", fontsize=18)

    model_label = scenario_cfg.models[args.model_key].get("label", args.model_key)
    fig.suptitle(
        f"{model_label} vs. {scenario_cfg.label} target -- global RMSE by pressure level",
        fontsize=15,
        y=1.02,
    )

    out_path = output_dir / f"rmse_heatmap_{scenario_cfg.key}_{args.model_key}_global.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved RMSE heatmap to {out_path}")


if __name__ == "__main__":
    main()
