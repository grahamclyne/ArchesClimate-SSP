r"""Generate the "RMSE across time" surface-variable grid from a scenario.

Adapted from analysis/global_mean_rollouts.ipynb (cell 6, "Surface RMSE").
For one scenario (identified by --scenario, matching a `key` in the same
YAML config used by paper_figures_table.py), plots per-region, per-year RMSE
(mean +/- std across ensemble members/seeds) against the IPSL target for
four surface variables (tas, ps, pr, net_flux) plus a land-masked tas row,
with MESMER-M shown on that land-tas row. One line per model declared in the
scenario's `models` dict (see paper_figures_table.py for the same config
schema).

Usage:
    python analysis/paper_figures_rmse_plot.py \\
        --config analysis/configs/paper_figures_table_example.yaml \\
        --scenario 534
"""

import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from matplotlib.ticker import MaxNLocator
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run_all_vars

REGIONS = {
    "North (86°-144°)": slice(86, 144),
    "Tropics (56°-86°)": slice(56, 86),
    "South (0°-56°)": slice(0, 56),
}
PLOT_VARS = [
    # (var_idx, display name, unit, "surf" or "land")
    (0, "tas", "K", "surf"),
    (5, "ps", "Pa", "surf"),
    (2, "pr", "kg m⁻² s⁻¹", "surf"),
    (6, "net\\_surface\\_flux", "W m⁻²", "surf"),
    (0, "tas (land)", "K", "land"),
]
MODEL_COLORS = ["cyan", "green", "blue", "darkorange", "magenta", "gold"]
MESMER_COLOR = "firebrick"


def monthly_to_yearly(t, time_dim):
    """(..., T, ...) -> (..., T // 12, ...) along time_dim, trimming the remainder."""
    T = t.shape[time_dim]
    T_yrs = T // 12
    idx = [slice(None)] * t.ndim
    idx[time_dim] = slice(None, T_yrs * 12)
    t = t[tuple(idx)]
    shape = list(t.shape)
    shape[time_dim : time_dim + 1] = [T_yrs, 12]
    return t.reshape(shape).mean(time_dim + 1)


def load_target_all_vars_yearly(scratch, realization, scenario, dataset_path="memmap_filled_in"):
    """Returns (yearly tensor (T_yrs, n_vars, 144, 144), first timestamp, last timestamp)."""
    import numpy as np
    import pandas as pd

    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    surface = td["surface"]  # (n_vars, T, 144, 144)
    surface = torch.where(surface.abs() < 1e30, surface, torch.nan)
    surface = surface.permute(1, 0, 2, 3)  # (T, n_vars, 144, 144)
    time = pd.DatetimeIndex(np.asarray(td["time"])) if "time" in td.keys() else None
    return (
        monthly_to_yearly(surface, time_dim=0),
        time[0] if time is not None else None,
        time[-1] if time is not None else None,
    )


def load_mesmer_yearly(path, target_first, target_last):
    """Align by the target's own calendar range, not a guessed start year.

    MESMER-M's source file can span a much wider (and differently-bounded)
    window than the target scenario (see paper_figures_table.py's note on
    this same alignment issue).
    """
    ds = xr.open_dataset(path)
    da = ds["tas"]
    if "variable" in da.dims:
        da = da.isel(variable=0)
    if "member" in da.dims:
        da = da.isel(member=0)
    if target_first is not None:
        da = da.sel(time=slice(target_first, target_last))
    da = da.transpose("time", "realisation", "lat", "lon")
    raw = torch.tensor(da.to_numpy(), dtype=torch.float32)  # (T_monthly, n_realisations, 144, 144)
    land_mask = raw[0, 0].isnan()  # True = ocean
    yearly = monthly_to_yearly(raw, time_dim=0)  # (T_yrs, n_realisations, 144, 144)
    return yearly, land_mask


def compute_region_rmse(pred_yr, tgt_yr, region_slice, var_idx=None):
    """pred_yr: (n_members, T, [n_vars,] lat, lon); tgt_yr: (T, [n_vars,] lat, lon).

    Returns (n_members, T[, n_vars]) RMSE over the region's lat slice.
    """
    p = pred_yr[..., region_slice, :]
    t = tgt_yr[..., region_slice, :]
    return ((p - t) ** 2).nanmean(dim=(-1, -2)).sqrt()


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
    surface_variables = list(hydra_cfg.module.surface_variables)
    data_mean = train_dataset.data_mean["surface"].squeeze(1)  # (n_vars, 144, 144)
    data_std = train_dataset.data_std["surface"].view(-1)

    target_yr, target_first, target_last = load_target_all_vars_yearly(
        scratch,
        scenario_cfg.target.get("realization", "r1i1p1f1"),
        scenario_cfg.target.scenario,
        dataset_path=hydra_cfg.module.dataset_path,
    )  # (T_yrs, n_vars, 144, 144)

    model_colors = {
        key: MODEL_COLORS[i % len(MODEL_COLORS)] for i, key in enumerate(scenario_cfg.models)
    }
    model_labels = {key: scenario_cfg.models[key].get("label", key) for key in scenario_cfg.models}
    model_yrs = {}
    for key, model_cfg in scenario_cfg.models.items():
        members = [
            load_run_all_vars(f"{scratch}/generated_data/{name}", data_mean, data_std)
            for name in model_cfg.runs
        ]
        t_min = min(m.shape[1] for m in members)
        raw = torch.cat([m[:, :t_min] for m in members], dim=0)  # (n_seeds, T, n_vars, 144, 144)
        model_yrs[key] = monthly_to_yearly(raw, time_dim=1)  # (n_seeds, T_yrs, n_vars, 144, 144)

    T_yrs = min([target_yr.shape[0]] + [v.shape[1] for v in model_yrs.values()])
    model_yrs = {k: v[:, :T_yrs] for k, v in model_yrs.items()}
    target_yr = target_yr[:T_yrs]
    start_year = target_first.year if target_first is not None else 2100 - T_yrs
    yrs = np.arange(start_year, start_year + T_yrs)

    mesmer_yr, land_mask = None, None
    if "mesmer" in scenario_cfg:
        mesmer_yr, land_mask = load_mesmer_yearly(
            scenario_cfg.mesmer.path, target_first, target_last
        )
        n = min(mesmer_yr.shape[0], T_yrs)
        mesmer_yr = mesmer_yr[:n]

    tas_idx = surface_variables.index("tas")
    tas_target_land = target_yr[:, tas_idx].clone()
    tas_model_land = {k: v[:, :, tas_idx].clone() for k, v in model_yrs.items()}
    if land_mask is not None:
        # Per-timestep 2-D slice x 2-D mask assignment -- avoids unreliable
        # 3-D boolean broadcasting (matches global_mean_rollouts.ipynb's own
        # workaround for the same issue).
        for t in range(tas_target_land.shape[0]):
            tas_target_land[t][land_mask] = float("nan")
        for k in tas_model_land:
            for s in range(tas_model_land[k].shape[0]):
                for t in range(tas_model_land[k].shape[1]):
                    tas_model_land[k][s, t][land_mask] = float("nan")

    region_list = list(REGIONS.keys())
    fig, axes = plt.subplots(
        len(PLOT_VARS), len(region_list), figsize=(16, 3.4 * len(PLOT_VARS)), sharey="row"
    )

    for row, (v_idx, vname, vunit, vtype) in enumerate(PLOT_VARS):
        for col, region in enumerate(region_list):
            ax = axes[row, col]
            region_slice = REGIONS[region]

            if vtype == "land":
                for key, tas_land in tas_model_land.items():
                    r = compute_region_rmse(tas_land, tas_target_land, region_slice)  # (n_seeds, T)
                    mean, std = r.mean(0).numpy(), r.std(0).numpy()
                    ax.plot(yrs[: len(mean)], mean, color=model_colors[key], lw=2)
                    ax.fill_between(
                        yrs[: len(mean)],
                        mean - std,
                        mean + std,
                        color=model_colors[key],
                        alpha=0.2,
                        lw=0,
                    )
                if mesmer_yr is not None:
                    t_mes = tas_target_land[: mesmer_yr.shape[0]]
                    r_mes = compute_region_rmse(
                        mesmer_yr.permute(1, 0, 2, 3), t_mes, region_slice
                    )  # (n_realisations, T)
                    mean_m, std_m = r_mes.mean(0).numpy(), r_mes.std(0).numpy()
                    t_m = yrs[: len(mean_m)]
                    ax.plot(t_m, mean_m, color=MESMER_COLOR, lw=2)
                    ax.fill_between(
                        t_m, mean_m - std_m, mean_m + std_m, color=MESMER_COLOR, alpha=0.2, lw=0
                    )
            else:
                tgt = target_yr[:, v_idx]
                for key, pred_all in model_yrs.items():
                    pred = pred_all[:, :, v_idx]
                    r = compute_region_rmse(pred, tgt, region_slice)  # (n_seeds, T)
                    mean, std = r.mean(0).numpy(), r.std(0).numpy()
                    ax.plot(yrs[: len(mean)], mean, color=model_colors[key], lw=2)
                    ax.fill_between(
                        yrs[: len(mean)],
                        mean - std,
                        mean + std,
                        color=model_colors[key],
                        alpha=0.2,
                        lw=0,
                    )

            ax.grid(alpha=0.3, linestyle=":")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            if row == 0:
                ax.set_title(region, fontsize=14)
            if col == 0:
                ax.set_ylabel(f"${vname}$\nRMSE ({vunit})", fontsize=13)
            if row == len(PLOT_VARS) - 1:
                ax.set_xlabel("Year", fontsize=13)

    handles = [
        mlines.Line2D([], [], color=model_colors[key], lw=2, label=model_labels[key])
        for key in scenario_cfg.models
    ]
    if mesmer_yr is not None:
        handles.append(mlines.Line2D([], [], color=MESMER_COLOR, lw=2, label="MESMER-M"))
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=13,
        frameon=True,
        bbox_to_anchor=(0.5, 1.02),
    )
    plt.tight_layout()

    out_path = output_dir / f"rmse_surface_regional_{scenario_cfg.key}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    print(f"Saved RMSE plot to {out_path}")


if __name__ == "__main__":
    main()
