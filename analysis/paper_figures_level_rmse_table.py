r"""Appendix "rest of model performance" table: RMSE, spatial std, and IAV per level.

Last-decade RMSE, spatial std, and detrended inter-annual variability (IAV)
-- all vs. the real CMIP target where applicable -- for every atmospheric
level variable at every pressure level, and for ocean temperature (thetao)
at every depth level. One baseline ("full") model, one scenario.

Companion to paper_figures_rmse_heatmap.py (same RMSE computation), but
emits a LaTeX table instead of a heatmap figure, and covers both the
atmospheric column (level/pressure) and the ocean column (lev/depth).
Spatial std / IAV definitions match analysis/compare_runs.py's
compute_metrics (same primitives: nanstd across space, get_iav's
per-pixel linear-detrend-then-std-over-time), applied to the last
final_decade_years years of the rollout, per pressure/depth level.

Usage:
    python -m analysis.paper_figures_level_rmse_table \\
        --config analysis/configs/paper_figures_pf4_emafix_step22000_whisker_val.yaml \\
        --scenario 434 --model-key full
"""

import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import (
    get_iav,
    load_run_lev,
    load_run_level_all,
    nanstd,
)


def load_target_level_last_decade_mean(
    scratch, realization, scenario, var_idx, final_decade_months, dataset_path
):
    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    level = td["level"][var_idx]  # (T, n_plevels, 144, 144)
    level = torch.where(level.abs() < 1e30, level, torch.nan)
    return level[-final_decade_months:].nanmean(dim=0)


def load_target_level_last_decade_annual(
    scratch, realization, scenario, var_idx, final_decade_months, dataset_path
):
    """Same source as load_target_level_last_decade_mean, keeps the annual series.

    Not yet time-averaged, with a fake member axis of 1, so
    spatial_std_by_level/iav_by_level (both member-axis-aware) can be reused
    directly on the IPSL target the same way they're used on the model.
    """
    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    level = td["level"][var_idx]  # (T, n_plevels, 144, 144)
    level = torch.where(level.abs() < 1e30, level, torch.nan)
    return to_annual_last_decade(
        level.unsqueeze(0), final_decade_months
    )  # (1, n_years, n_plevels, 144, 144)


def load_target_lev_last_decade_annual(
    scratch, realization, scenario, final_decade_months, dataset_path
):
    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    lev = td["lev"][0]  # (T, n_depths, 144, 144)
    lev = torch.where(lev.abs() < 1e30, lev, torch.nan)
    return to_annual_last_decade(
        lev.unsqueeze(0), final_decade_months
    )  # (1, n_years, n_depths, 144, 144)


def load_target_lev_last_decade_mean(
    scratch, realization, scenario, final_decade_months, dataset_path
):
    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    lev = td["lev"][0]  # (T, n_depths, 144, 144) -- single ocean variable (thetao)
    lev = torch.where(lev.abs() < 1e30, lev, torch.nan)
    return lev[-final_decade_months:].nanmean(dim=0)


def region_weighted_rmse(pred_map, target_map, lat_slice=slice(0, 144)):
    """pred_map/target_map: (n_levels, 144, 144) -> (n_levels,), lat-weighted global RMSE."""
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
    return torch.sqrt(weighted_mse)  # (n_levels,)


def to_annual_last_decade(monthly, final_decade_months):
    """(n_members, T, n_levels, 144, 144) -> (n_members, n_years, n_levels, 144, 144)."""
    tail = monthly[:, -final_decade_months:]
    n_years = tail.shape[1] // 12
    tail = tail[:, : n_years * 12]
    shape = tail.shape
    return tail.view(shape[0], n_years, 12, *shape[2:]).mean(dim=2)


def load_model_level_last_decade_annual(
    scratch, run_names, var_idx, data_mean_level, data_std_level, final_decade_months
):
    """Returns (n_members_total, n_years, n_plevels, 144, 144) annual-mean tensor."""
    annuals = []
    for name in run_names:
        run_dir = os.path.join(scratch, "generated_data", name)
        raw = load_run_level_all(run_dir, data_mean_level, data_std_level)
        var_raw = raw[:, :, var_idx]  # (n_members, T, n_plevels, 144, 144)
        annuals.append(to_annual_last_decade(var_raw, final_decade_months))
    return torch.cat(annuals, dim=0)


def load_model_lev_last_decade_annual(
    scratch, run_names, data_mean_lev, data_std_lev, final_decade_months
):
    """Returns (n_members_total, n_years, n_depths, 144, 144) annual-mean tensor."""
    annuals = []
    for name in run_names:
        run_dir = os.path.join(scratch, "generated_data", name)
        raw = load_run_lev(
            run_dir, data_mean_lev, data_std_lev
        )  # (n_members, T, n_depths, 144, 144)
        annuals.append(to_annual_last_decade(raw, final_decade_months))
    return torch.cat(annuals, dim=0)


def spatial_std_by_level(annual):
    """annual: (n_members, n_years, n_levels, 144, 144) -> (n_levels,)."""
    n_levels = annual.shape[2]
    return torch.stack([nanstd(annual[:, :, i], dim=(-1, -2)).mean() for i in range(n_levels)])


def iav_by_level(annual):
    """annual: (n_members, n_years, n_levels, 144, 144) -> (n_levels,)."""
    n_levels = annual.shape[2]
    return torch.stack([get_iav(annual[:, :, i]).nanmean() for i in range(n_levels)])


def format_latex_table(
    header_label, level_labels, level_unit, metrics_by_var, var_units, metric_names
):
    """metrics_by_var: {var: {metric_name: (n_levels,) array}}."""
    variables = list(metrics_by_var.keys())
    cols = [(v, m) for v in variables for m in metric_names]
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    col_spec = "l" + "c" * len(cols)
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    top_header = (
        " & "
        + " & ".join(
            rf"\multicolumn{{{len(metric_names)}}}{{c}}{{{v} [{var_units.get(v, '')}]}}"
            for v in variables
        )
        + r" \\"
    )
    lines.append(top_header)
    sub_header = (
        header_label
        + " ("
        + level_unit
        + ")"
        + " & "
        + " & ".join(m for _ in variables for m in metric_names)
        + r" \\"
    )
    lines.append(sub_header)
    lines.append(r"\midrule")
    for i, lvl_label in enumerate(level_labels):
        row = [lvl_label]
        for v in variables:
            for m in metric_names:
                cell = f"{metrics_by_var[v][m][i]:.3g}"
                target_key = f"{m}_target"
                if target_key in metrics_by_var[v]:
                    cell += f" ({metrics_by_var[v][target_key][i]:.3g})"
                row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    parser.add_argument("--scenario", required=True, help="Scenario key to evaluate (e.g. '434').")
    parser.add_argument(
        "--variables",
        default="ta,zg,hus,ua,va",
        help="Comma-separated atmospheric level variables to include.",
    )
    parser.add_argument("--model-key", default="full", help="Which model dict entry to evaluate.")
    args = parser.parse_args()
    variables = args.variables.split(",")
    metric_names = ["RMSE", "SpStd", "IAV"]

    var_units = {"ta": "K", "zg": "m", "hus": "kg/kg", "ua": "m/s", "va": "m/s", "thetao": "K"}

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
    realization = scenario_cfg.target.get("realization", "r1i1p1f1")
    experiment = scenario_cfg.target.scenario

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    level_vars = list(hydra_cfg.module.level_variables)
    pressure_levels = [float(p) for p in hydra_cfg.module.pressure_levels]
    depth_levels = [float(d) for d in hydra_cfg.module.depth_levels]
    plevel_labels = [str(int(p / 100)) for p in pressure_levels]
    depth_labels = [f"{d:.1f}" for d in depth_levels]

    data_mean_level = train_dataset.data_mean["level"]
    data_std_level = train_dataset.data_std["level"]
    data_mean_lev = train_dataset.data_mean["lev"].squeeze(0)
    data_std_lev = train_dataset.data_std["lev"].squeeze(0)

    # --- Atmosphere: RMSE / spatial std / IAV per variable, per pressure level ---
    metrics_by_var = {}
    for var in variables:
        var_idx = level_vars.index(var)
        target_mean = load_target_level_last_decade_mean(
            scratch,
            realization,
            experiment,
            var_idx,
            final_decade_months,
            dataset_path=hydra_cfg.module.dataset_path,
        )
        annual = load_model_level_last_decade_annual(
            scratch,
            run_names,
            var_idx,
            data_mean_level,
            data_std_level,
            final_decade_months,
        )
        target_annual = load_target_level_last_decade_annual(
            scratch,
            realization,
            experiment,
            var_idx,
            final_decade_months,
            dataset_path=hydra_cfg.module.dataset_path,
        )
        pred_mean = annual.mean(dim=(0, 1))  # (n_plevels, 144, 144)
        metrics_by_var[var] = {
            "RMSE": region_weighted_rmse(pred_mean, target_mean).numpy(),
            "SpStd": spatial_std_by_level(annual).numpy(),
            "IAV": iav_by_level(annual).numpy(),
            "SpStd_target": spatial_std_by_level(target_annual).numpy(),
            "IAV_target": iav_by_level(target_annual).numpy(),
        }

    atm_table = format_latex_table(
        "Pressure level",
        plevel_labels,
        "hPa",
        metrics_by_var,
        var_units,
        metric_names,
    )

    # --- Ocean: RMSE / spatial std / IAV for thetao, per depth level ---
    target_lev_mean = load_target_lev_last_decade_mean(
        scratch,
        realization,
        experiment,
        final_decade_months,
        dataset_path=hydra_cfg.module.dataset_path,
    )
    lev_annual = load_model_lev_last_decade_annual(
        scratch,
        run_names,
        data_mean_lev,
        data_std_lev,
        final_decade_months,
    )
    target_lev_annual = load_target_lev_last_decade_annual(
        scratch,
        realization,
        experiment,
        final_decade_months,
        dataset_path=hydra_cfg.module.dataset_path,
    )
    pred_lev_mean = lev_annual.mean(dim=(0, 1))
    ocean_metrics = {
        "thetao": {
            "RMSE": region_weighted_rmse(pred_lev_mean, target_lev_mean).numpy(),
            "SpStd": spatial_std_by_level(lev_annual).numpy(),
            "IAV": iav_by_level(lev_annual).numpy(),
            "SpStd_target": spatial_std_by_level(target_lev_annual).numpy(),
            "IAV_target": iav_by_level(target_lev_annual).numpy(),
        }
    }
    ocean_table = format_latex_table(
        "Depth level",
        depth_labels,
        "m",
        ocean_metrics,
        var_units,
        metric_names,
    )

    model_label = scenario_cfg.models[args.model_key].get("label", args.model_key)
    out_path = output_dir / f"level_rmse_table_{scenario_cfg.key}_{args.model_key}.tex"
    with open(out_path, "w") as f:
        f.write(
            f"% Last-{final_decade_years}-year RMSE / spatial std / detrended IAV vs. target, "
            f"model={model_label}, scenario={scenario_cfg.label}\n"
            f"% SpStd/IAV cells are 'model (IPSL)'\n"
        )
        f.write("% Atmosphere (by pressure level)\n")
        f.write(atm_table + "\n\n")
        f.write("% Ocean (by depth level)\n")
        f.write(ocean_table + "\n")

    print(f"Saved LaTeX table to {out_path}")
    print()
    print(atm_table)
    print()
    print(ocean_table)


if __name__ == "__main__":
    main()
