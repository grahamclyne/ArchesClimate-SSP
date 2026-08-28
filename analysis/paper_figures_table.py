r"""Generate the paper's model-performance LaTeX table (IPSL / MESMER-M / models).

Given a YAML config listing, per scenario, a target CMIP scenario, an
optional MESMER-M emulation, and one or more named models (each a list of
rollout folders; multiple checkpoints of the same model are stacked together
as pseudo-ensemble-members, matching the practice in
analysis/global_mean_stats.ipynb), computes for the final
`final_decade_years` of each scenario:

  - RMSE (K) of each model's/MESMER-M's annual-mean maps against the IPSL
    target
  - spatial standard deviation
  - ensemble spread (nan for MESMER-M / IPSL, which have no member axis here)
  - detrended inter-annual variability (IAV) of the anomaly from a historical
    climatology

and renders a \\multirow LaTeX table, one scenario block per group, row order
= models (in config-declared order), then MESMER-M, then IPSL.

Usage:
    python analysis/paper_figures_table.py \\
        --config analysis/configs/paper_figures_table_example.yaml
"""

import argparse
import os
import warnings
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run, nanstd

# --------------------------------------------------------------------------
# Metrics (final-decade, climatology-relative -- mirrors
# analysis/global_mean_stats.ipynb cell 6, not compare_runs.py's more
# general monthly/decade-chunked compute_metrics)
# --------------------------------------------------------------------------


def compute_lat_weighted_rmse(pred, target, lats):
    """pred/target: (member, n_years, lat, lon) -> (member,)."""
    weights = torch.cos(torch.deg2rad(lats)).view(1, 1, -1, 1)
    mask = ~torch.isnan(pred - target)
    effective_weights = weights * mask.float()
    sq_err = torch.nan_to_num((pred - target) ** 2, nan=0.0)
    weighted_mse = (sq_err * effective_weights).sum(dim=(1, 2, 3)) / effective_weights.sum(
        dim=(1, 2, 3)
    )
    return torch.sqrt(weighted_mse)


def compute_lat_weighted_crps(preds, target, lats):
    """Fair (unbiased) ensemble CRPS, lat-weighted, reduced to a scalar.

    preds: (member, n_years, lat, lon), target: (1, n_years, lat, lon).
    Uses the standard PWM/fair estimator (matches metrics/ensemble_metrics.py's
    `fcrps`): fcrps = mae - 0.5 * dispersion * n/(n-1), where mae is the
    mean absolute error of each member against the target and dispersion is
    the mean pairwise absolute difference between members (an unbiased
    estimator of the ensemble's own spread). Undefined for a single-member
    ensemble (returns None) since dispersion requires at least 2 members.
    """
    n = preds.shape[0]
    if n < 2:
        return None
    mae = (preds - target).abs().mean(0)  # (n_years, lat, lon)
    dispersion = (
        (preds.unsqueeze(0) - preds.unsqueeze(1)).abs().mean(dim=(0, 1))
    )  # (n_years, lat, lon)
    fcrps_map = mae - 0.5 * dispersion * n / (n - 1)
    mask = ~torch.isnan(fcrps_map)
    weights = torch.cos(torch.deg2rad(lats)).view(1, -1, 1).expand_as(fcrps_map)
    effective_weights = weights * mask.float()
    val = torch.nan_to_num(fcrps_map, nan=0.0)
    return (val * effective_weights).sum().item() / effective_weights.sum().item()


def get_iav(tensor):
    """Detrended inter-annual variability for (member, n_years, lat, lon)."""
    _, t, _, _ = tensor.shape
    x = torch.linspace(0, t - 1, t, dtype=tensor.dtype).view(1, t, 1, 1)
    x_mean = x.nanmean()
    y_mean = tensor.nanmean(dim=1, keepdim=True)
    slope = ((x - x_mean) * (tensor - y_mean)).sum(dim=1, keepdim=True) / ((x - x_mean) ** 2).sum(
        dim=1, keepdim=True
    )
    intercept = y_mean - slope * x_mean
    detrended = tensor - (slope * x + intercept)
    return torch.std(detrended, dim=1)


def to_annual(monthly):
    """(..., T, lat, lon) -> (..., T // 12, lat, lon), trimming the remainder."""
    t = monthly.shape[-3]
    n_years = t // 12
    trimmed = monthly[..., : n_years * 12, :, :]
    shape = trimmed.shape
    return trimmed.view(*shape[:-3], n_years, 12, *shape[-2:]).mean(dim=-3)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_target_annual(scratch, realization, scenario, var_idx, dataset_path="memmap_filled_in"):
    """Returns (annual tensor (1, n_years, lat, lon), first timestamp, last timestamp).

    The timestamps are the target's own calendar range (e.g. ssp534-over runs
    2040-2100, not 2015-2100) -- needed to align MESMER-M, whose source file
    covers a much wider/differently-bounded window, by calendar date rather
    than by array position.
    """
    import numpy as np
    import pandas as pd

    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    surface = td["surface"][var_idx][None]  # (1, T, lat, lon), fake member dim
    surface = torch.where(surface.abs() < 1e30, surface, torch.nan)
    time = pd.DatetimeIndex(np.asarray(td["time"])) if "time" in td.keys() else None
    return (
        to_annual(surface),
        time[0] if time is not None else None,
        time[-1] if time is not None else None,
    )


def load_model_ensemble_annual(scratch, run_names, var_idx, data_mean, data_std):
    """Load and stack one or more rollout folders as pseudo-ensemble-members."""
    members = []
    for name in run_names:
        run_dir = os.path.join(scratch, "generated_data", name)
        members.append(load_run(run_dir, var_idx, data_mean, data_std))
    t_min = min(m.shape[1] for m in members)
    stacked = torch.cat([m[:, :t_min] for m in members], dim=0)  # (n_members, T, lat, lon)
    return to_annual(stacked)


def compute_climatology_annual(scratch, clim_cfg, n_years, dataset_path="memmap_filled_in"):
    """Historical climatology, tiled to n_years.

    Mirrors the notebook's historical[:, -(skip+n_clim):-skip] average,
    tiled across the scenario's projection window.

    `field`/`var_idx` here mean what they mean in the notebook: `field`
    selects the TensorDict key (default 'lev', i.e. ocean thetao), and
    `var_idx` indexes the field's *depth* axis (0 = shallowest, ~0.5m) --
    used as a stand-in proxy when detrending land-surface-temperature
    anomalies, not a surface-variable index. Preserved as-is for fidelity
    with the reference table's methodology.
    """
    realization = clim_cfg.get("realization", "r1i1p1f1")
    path = f"{scratch}/{dataset_path}/{realization}_historical_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    field = clim_cfg.get("field", "lev")
    depth_idx = clim_cfg.get("var_idx", 0)
    skip_last = clim_cfg.get("skip_last_years", 10)
    n_clim_years = clim_cfg.get("n_clim_years", 20)

    data = td[field]  # (n_vars, T, n_depth, 144, 144)
    n_months = n_clim_years * 12
    window = data[
        :, -(skip_last * 12 + n_months) : -(skip_last * 12)
    ]  # (n_vars, T_window, n_depth, 144, 144)
    climatology = window.view(n_clim_years, 12, window.shape[0], window.shape[2], 144, 144).mean(
        0
    )  # (12, n_vars, n_depth, 144, 144)

    n_tiles = n_years  # each tile of the 12-month cycle collapses to one output year
    tiled = climatology.tile(n_tiles, 1, 1, 1, 1)  # (12*n_tiles, n_vars, n_depth, 144, 144)
    clima_yearly = tiled.view(-1, 12, climatology.shape[1], climatology.shape[2], 144, 144).mean(
        1
    )  # (n_tiles, n_vars, n_depth, 144, 144)
    return clima_yearly[:n_years, 0, depth_idx][None]  # (1, n_years, 144, 144)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def compute_scenario_metrics(
    scenario_cfg,
    scratch,
    var_idx,
    data_mean,
    data_std,
    lats,
    final_decade_years,
    climatology_cfg,
    dataset_path="memmap_filled_in",
):
    ipsl_annual, target_first, target_last = load_target_annual(
        scratch,
        scenario_cfg.target.get("realization", "r1i1p1f1"),
        scenario_cfg.target.scenario,
        var_idx,
        dataset_path=dataset_path,
    )

    model_annuals = {
        model_key: load_model_ensemble_annual(
            scratch, list(model_cfg.runs), var_idx, data_mean, data_std
        )
        for model_key, model_cfg in scenario_cfg.models.items()
    }

    mesmer_annual = None
    if "mesmer" in scenario_cfg:
        import xarray as xr

        ds = xr.open_dataset(scenario_cfg.mesmer.path)
        da = ds["tas"]
        if "variable" in da.dims:
            da = da.isel(variable=0)
        if "member" in da.dims:
            da = da.isel(member=0)
        # Align by *calendar date*, not array position: MESMER-M's source file
        # spans a much wider (and differently-bounded) window than the target
        # scenario -- e.g. ssp534-over runs 2040-2100, but the MESMER file
        # itself starts in 1851. Slicing by the target's own first/last
        # timestamp is required; a positional slice silently compares the
        # wrong decades (this previously inflated MESMER's RMSE hugely: it
        # was being compared against a ~20-25 year offset).
        if target_first is not None:
            da = da.sel(time=slice(target_first, target_last))
            if da.sizes.get("time", 0) == 0:
                warnings.warn(
                    f"MESMER-M has no overlap with target window "
                    f"{target_first}..{target_last}; skipping MESMER-M row."
                )
                da = None
        if da is not None:
            arr = torch.tensor(
                da.transpose("realisation", "time", "lat", "lon").values, dtype=torch.float32
            )
            mesmer_annual = to_annual(arr)
            if mesmer_annual.shape[1] < final_decade_years:
                warnings.warn(
                    f"MESMER-M only covers {mesmer_annual.shape[1]} overlapping years "
                    f"with the target (< final_decade_years={final_decade_years}); "
                    "its 'final decade' will actually be its last available years, "
                    "not the same calendar decade as AC/IPSL."
                )
            land_mask = torch.isnan(mesmer_annual[0, 0])  # True over ocean
            ipsl_annual = torch.where(land_mask, torch.nan, ipsl_annual)
            model_annuals = {
                k: torch.where(land_mask, torch.nan, v) for k, v in model_annuals.items()
            }

    # Each model's overlap with IPSL is independent of MESMER-M's (possibly
    # shorter) calendar coverage -- MESMER-M gets its own comparison window
    # below, so a short MESMER-M file doesn't truncate any model-vs-IPSL
    # final decade.
    n_years = min([ipsl_annual.shape[1]] + [v.shape[1] for v in model_annuals.values()])
    clima_annual = compute_climatology_annual(
        scratch, climatology_cfg, n_years, dataset_path=dataset_path
    )

    final = slice(-final_decade_years, None)
    ipsl_d = ipsl_annual[:, :n_years][:, final]
    clima_d = clima_annual[:, final]
    model_ds = {k: v[:, :n_years][:, final] for k, v in model_annuals.items()}

    mesmer_d = None
    ipsl_d_for_mesmer = None
    if mesmer_annual is not None:
        mesmer_d = mesmer_annual[:, -final_decade_years:]
        # Same calendar years MESMER-M actually covers (its own last decade),
        # not the models'/IPSL's final decade if MESMER-M falls short of it.
        # The climatology is a repeating 12-month cycle (identical every
        # year), so clima_d is valid for this window too without recomputing.
        ipsl_d_for_mesmer = ipsl_annual[:, : mesmer_annual.shape[1]][:, -final_decade_years:]

    metrics = {}
    for model_key, model_d in model_ds.items():
        metrics[model_key] = {
            "rmse": compute_lat_weighted_rmse(model_d, ipsl_d, lats).mean().item(),
            "crps": compute_lat_weighted_crps(model_d, ipsl_d, lats),
            "sp_std": nanstd(model_d, dim=(-1, -2)).mean().item(),
            "spread": model_d.nanmean(dim=(-1, -2)).std(0).nanmean().item(),
            "iav": get_iav(model_d - clima_d).nanmean().item(),
        }
    if mesmer_d is not None:
        metrics["MESMER"] = {
            "rmse": compute_lat_weighted_rmse(mesmer_d, ipsl_d_for_mesmer, lats).mean().item(),
            "crps": compute_lat_weighted_crps(mesmer_d, ipsl_d_for_mesmer, lats),
            "sp_std": nanstd(mesmer_d, dim=(-1, -2)).mean().item(),
            "spread": mesmer_d.nanmean(dim=(-1, -2)).std(0).nanmean().item(),
            "iav": get_iav(mesmer_d - clima_d).nanmean().item(),
        }
    else:
        warnings.warn(f"No MESMER config for scenario {scenario_cfg.key}; row omitted.")
    metrics["IPSL"] = {
        "rmse": None,
        "crps": None,
        "sp_std": nanstd(ipsl_d, dim=(-1, -2)).mean().item(),
        "spread": None,
        "iav": get_iav(ipsl_d - clima_d).nanmean().item(),
    }

    # dict insertion order above is exactly the desired row order: models (in
    # config-declared order), then MESMER-M (if present), then IPSL.
    labels = {k: scenario_cfg.models[k].get("label", k) for k in scenario_cfg.models}
    labels["MESMER"] = r"$MESMER\text{-}M$"
    labels["IPSL"] = "IPSL"

    return metrics, labels


def fmt(val, digits=4):
    return f"{val:.{digits}f}" if val is not None else "n/a"


def build_latex_table(all_metrics, all_labels, scenario_labels):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Model} & \textbf{RMSE (K)} & "
        r"\textbf{\shortstack{Fair\\\\ CRPS (K)}} & "
        r"\textbf{\shortstack{Spatial Std.\\\\ Deviation}} & "
        r"\textbf{\shortstack{Ensemble\\\\ Spread}} & "
        r"\textbf{\shortstack{Inter-annual\\\\ Variability}} \\\\",
        r"\midrule",
    ]
    for i, (scen_key, scen_label) in enumerate(scenario_labels.items()):
        if i > 0:
            lines.append(r"\midrule")
        rows = list(all_metrics[scen_key].keys())  # insertion order = models, MESMER, IPSL
        for j, model in enumerate(rows):
            m = all_metrics[scen_key][model]
            label = all_labels[scen_key][model]
            row = (
                f"    & {label} & {fmt(m['rmse'])} & {fmt(m['crps'])} & {fmt(m['sp_std'])} & "
                f"{fmt(m['spread'])} & {fmt(m['iav'])} \\\\"
            )
            if j == 0:
                lines.append(rf"\multirow{{{len(rows)}}}{{*}}{{\textbf{{{scen_label}}}}}")
            lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-table YAML config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    final_decade_years = cfg.get("final_decade_years", 10)
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    surface_variables = list(hydra_cfg.module.surface_variables)
    variable = cfg.get("variable", "tas")
    var_idx = surface_variables.index(variable)
    data_mean = train_dataset.data_mean["surface"].squeeze(1)
    data_std = train_dataset.data_std["surface"].view(-1)
    lats = torch.linspace(-90, 90, 144)

    all_metrics = {}
    all_labels = {}
    scenario_labels = {}
    for scenario_cfg in cfg.scenarios:
        scenario_labels[scenario_cfg.key] = scenario_cfg.label
        all_metrics[scenario_cfg.key], all_labels[scenario_cfg.key] = compute_scenario_metrics(
            scenario_cfg,
            scratch,
            var_idx,
            data_mean,
            data_std,
            lats,
            final_decade_years,
            cfg.get("climatology", {}),
            dataset_path=hydra_cfg.module.dataset_path,
        )

    latex = build_latex_table(all_metrics, all_labels, scenario_labels)
    print(latex)

    out_path = output_dir / "table_model_performance.tex"
    out_path.write_text(latex + "\n")
    print(f"\nSaved LaTeX table to {out_path}")


if __name__ == "__main__":
    main()
