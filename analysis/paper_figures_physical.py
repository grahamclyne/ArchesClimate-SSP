r"""Hydrostatic-consistency and energy-balance figures for one scenario.

Reuses the diagnostic math from analysis/energy_hydrostatic_balance.py
(hydrostatic residual, atmospheric column energy decomposition, ocean heat
content, global energy-conservation residual, supersaturation, Clausius-
Clapeyron scaling) but loads rollouts the same way as the other
paper_figures_*.py scripts: via a YAML config's `models` dict (matching
paper_figures_table.py's schema; runs are stacked as pseudo-ensemble-
members), rather than energy_hydrostatic_balance.py's own
--experiment_name/--member/--seed CLI (which also assumes a different
rollout filename convention).

Ensemble members are averaged into one field *before* running diagnostics --
exact for every energy term except atmospheric kinetic energy (nonlinear in
ua/va, so this is mean-of-fields, not mean-of-energies), which is an
acceptable approximation for a small ensemble spread and keeps this
consistent with how the other paper_figures_*.py scripts collapse the member
axis.

For one scenario (--scenario, matching a `key` in the config), for each
model in that scenario's `models` dict, saves:
  - hydrostatic_{scenario}_{model}.pdf: zonal-mean hydrostatic residual
  - energy_budget_{scenario}_{model}.pdf: E_atm/E_oce trajectories + dE/dt -
    F_net conservation residual, model vs. IPSL target
  - supersaturation_{scenario}_{model}.pdf: hus > q_sat frequency by level
  - cc_scaling_{scenario}_{model}.pdf: Clausius-Clapeyron ln(hus) vs. ta fit
  - spatial_bias_{scenario}_{model}.pdf: last-decade-mean E_atm/E_oce maps,
    model vs. target vs. bias

Usage:
    python analysis/paper_figures_physical.py \\
        --config analysis/configs/paper_figures_table_example.yaml \\
        --scenario 434
"""

import argparse
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import (
    load_run_all_vars,
    load_run_lev,
    load_run_level_all,
)
from analysis.energy_hydrostatic_balance import (
    energy_component_mae,
    grid_cell_area,
    load_target,
    plot_cc_scaling_comparison,
    plot_energy_budget_comparison,
    plot_hydrostatic_residual,
    plot_hydrostatic_residual_decades,
    plot_spatial_bias,
    plot_supersaturation_comparison,
    run_diagnostics,
)


def _cat_members(tensors):
    t_min = min(t.shape[1] for t in tensors)
    return torch.cat([t[:, :t_min] for t in tensors], dim=0)


def load_model_fields(
    scratch,
    run_names,
    data_mean_surface,
    data_std_surface,
    data_mean_level,
    data_std_level,
    data_mean_lev,
    data_std_lev,
):
    """Ensemble-mean surface/level/lev fields for one model's declared runs.

    Returns {"surface": (T, n_vars, 144, 144), "level": (T, n_vars, n_plevels,
    144, 144), "lev": (T, n_depth, 144, 144)}, matching the shapes
    energy_hydrostatic_balance.run_diagnostics expects.
    """
    run_dirs = [os.path.join(scratch, "generated_data", name) for name in run_names]
    surface = _cat_members(
        [load_run_all_vars(d, data_mean_surface, data_std_surface) for d in run_dirs]
    )
    level = _cat_members([load_run_level_all(d, data_mean_level, data_std_level) for d in run_dirs])
    lev = _cat_members([load_run_lev(d, data_mean_lev, data_std_lev) for d in run_dirs])
    return dict(surface=surface.nanmean(dim=0), level=level.nanmean(dim=0), lev=lev.nanmean(dim=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    parser.add_argument("--scenario", required=True, help="Scenario key to plot (e.g. '434').")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    output_dir = Path("plots") / cfg.output_name / "physical"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfg = next(s for s in cfg.scenarios if str(s.key) == str(args.scenario))

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    pressure_levels = [float(p) for p in hydra_cfg.module.pressure_levels]
    depth_levels = [float(d) for d in hydra_cfg.module.depth_levels]
    level_vars = list(hydra_cfg.module.level_variables)
    surface_vars = list(hydra_cfg.module.surface_variables)
    area = grid_cell_area()

    data_mean_surface = train_dataset.data_mean["surface"].squeeze(1)
    data_std_surface = train_dataset.data_std["surface"].view(-1)
    data_mean_level = train_dataset.data_mean["level"]  # (n_vars, n_plevels, 144, 144)
    data_std_level = train_dataset.data_std["level"]  # (n_vars, n_plevels, 1, 1)
    data_mean_lev = train_dataset.data_mean["lev"].squeeze(0)
    data_std_lev = train_dataset.data_std["lev"].view(-1)

    realization = scenario_cfg.target.get("realization", "r1i1p1f1")
    scenario_name = scenario_cfg.target.scenario

    for model_key, model_cfg in scenario_cfg.models.items():
        label = model_cfg.get("label", model_key)
        tag = f"{scenario_cfg.key}_{model_key}"
        print(f"[{tag}] loading rollouts...")

        pred = load_model_fields(
            scratch,
            list(model_cfg.runs),
            data_mean_surface,
            data_std_surface,
            data_mean_level,
            data_std_level,
            data_mean_lev,
            data_std_lev,
        )
        n_t = pred["surface"].shape[0]
        # predicted step i corresponds to ground truth month i+1 (lead_time_months=1),
        # matching energy_hydrostatic_balance.py's own load_target usage.
        target = load_target(
            scenario_name,
            start=1,
            stop=1 + n_t,
            realization=realization,
            dataset_path=hydra_cfg.module.dataset_path,
        )
        n_t = min(n_t, target["surface"].shape[0])
        pred = {k: v[:n_t] for k, v in pred.items()}
        target = {k: v[:n_t] for k, v in target.items()}

        pred_diag = run_diagnostics(
            pred, pressure_levels, depth_levels, level_vars, surface_vars, area
        )
        target_diag = run_diagnostics(
            target, pressure_levels, depth_levels, level_vars, surface_vars, area
        )

        print(
            f"[{tag}] hydrostatic residual MAE: "
            f"pred={pred_diag['hydrostatic_residual'].abs().nanmean():.2f} m, "
            f"target={target_diag['hydrostatic_residual'].abs().nanmean():.2f} m"
        )
        print(
            f"[{tag}] energy budget residual mean: "
            f"pred={pred_diag['budget_residual'].mean():.3e} W, "
            f"target={target_diag['budget_residual'].mean():.3e} W"
        )

        mae = {
            "e_atm": energy_component_mae(pred_diag["e_atm"], target_diag["e_atm"], area)
            .mean()
            .item(),
            "e_oce": energy_component_mae(pred_diag["e_oce"], target_diag["e_oce"], area)
            .mean()
            .item(),
        }
        for component in ("sensible", "potential", "kinetic", "latent"):
            mae[f"e_atm_{component}"] = (
                energy_component_mae(
                    pred_diag["e_atm_components"][component],
                    target_diag["e_atm_components"][component],
                    area,
                )
                .mean()
                .item()
            )
        for name, val in mae.items():
            print(f"[{tag}] {name} MAE (pred vs target): {val:.3e} J/m^2")

        plot_hydrostatic_residual(
            pred_diag["hydrostatic_residual"],
            pressure_levels,
            title=f"{scenario_cfg.label} -- {label} hydrostatic residual",
            out_path=output_dir / f"hydrostatic_{tag}.pdf",
            dz_hydro=pred_diag["hydrostatic_dz_hydro"],
        )
        plot_hydrostatic_residual_decades(
            pred_diag["hydrostatic_residual"],
            pressure_levels,
            out_path=output_dir / f"hydrostatic_decades_{tag}.pdf",
            dz_hydro=pred_diag["hydrostatic_dz_hydro"],
        )
        plot_energy_budget_comparison(
            pred_diag,
            target_diag,
            title=f"{scenario_cfg.label} -- {label}",
            out_path=output_dir / f"energy_budget_{tag}.pdf",
        )
        plot_supersaturation_comparison(
            pred_diag["supersaturation"],
            target_diag["supersaturation"],
            pressure_levels,
            title=f"{scenario_cfg.label} -- {label}: supersaturation frequency",
            out_path=output_dir / f"supersaturation_{tag}.pdf",
        )
        plot_cc_scaling_comparison(
            pred_diag["cc_scaling"],
            target_diag["cc_scaling"],
            title=f"{scenario_cfg.label} -- {label}: Clausius-Clapeyron scaling",
            out_path=output_dir / f"cc_scaling_{tag}.pdf",
        )
        n_decade = min(120, pred_diag["e_atm"].shape[0])
        plot_spatial_bias(
            pred_diag["e_atm"][-n_decade:].mean(dim=0).numpy(),
            target_diag["e_atm"][-n_decade:].mean(dim=0).numpy(),
            pred_diag["e_oce"][-n_decade:].mean(dim=0).numpy(),
            target_diag["e_oce"][-n_decade:].mean(dim=0).numpy(),
            title=f"{scenario_cfg.label} -- {label}: last {n_decade} months mean",
            out_path=output_dir / f"spatial_bias_{tag}.pdf",
        )

    print(f"\nSaved physical-consistency figures under {output_dir}/")


if __name__ == "__main__":
    main()
