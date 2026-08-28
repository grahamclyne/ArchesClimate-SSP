r"""Generate the regional ocean-temperature Hovmoller grid (time x latitude).

Adapted from analysis/regional_hovmoller.ipynb (final cell, saved as
"hovmoller_figure.pdf"). For each of three latitude regions (North/Tropics/
South, matching paper_figures_rmse_plot.py's REGIONS) and each of SSP4-3.4/
SSP5-3.4, plots the ensemble-mean, longitude-collapsed, climatology-relative
anomaly of shallow ocean temperature (thetao, ~0.5 m depth) over time and
latitude. One column per model declared in the config (same `models` dict
schema as paper_figures_table.py) plus a final IPSL column.

Requires both scenario keys "434" and "534" in the config -- like
paper_figures_global_mean.py, this is inherently a side-by-side comparison,
not a per-scenario figure.

Usage:
    python analysis/paper_figures_hovmoller.py \\
        --config analysis/configs/paper_figures_table_example.yaml
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run_lev
from analysis.paper_figures_rmse_plot import REGIONS, monthly_to_yearly

DEPTH_IDX = 0  # shallowest ocean level (~0.5 m) -- Hovmoller is 2-D (time x lat), one depth only


def compute_shallow_ocean_climatology(
    scratch,
    realization="r1i1p1f1",
    skip_last_months=10,
    n_clim_years=20,
    dataset_path="memmap_filled_in",
):
    """Single (144, 144) de-seasonalized baseline map at DEPTH_IDX.

    From the trailing `n_clim_years` of historical data before the last `skip_last_months`.

    Mirrors regional_hovmoller.ipynb cell 2 exactly (historical[:, -250:-10],
    i.e. skip 10 *months*, not years -- a different convention from
    paper_figures_table.py's climatology config, which skips
    `skip_last_years` years; preserved as-is for fidelity with this notebook).
    """
    path = f"{scratch}/{dataset_path}/{realization}_historical_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    lev = td["lev"][0]  # (T, n_depth, 144, 144) -- single ocean variable (thetao)
    n_months = n_clim_years * 12
    window = lev[
        -(skip_last_months + n_months) : -skip_last_months
    ]  # (T_window, n_depth, 144, 144)
    climatology = window.view(n_clim_years, 12, window.shape[1], 144, 144).mean(
        0
    )  # (12, n_depth, 144, 144)
    return climatology.mean(0)[DEPTH_IDX]  # (144, 144)


def load_target_anomaly_annual(
    scratch, realization, scenario, climatology_map, dataset_path="memmap_filled_in"
):
    import numpy as np
    import pandas as pd

    path = f"{scratch}/{dataset_path}/{realization}_{scenario}_interpolation.memmap"
    td = TensorDict.load_memmap(path)
    lev = td["lev"][0, :, DEPTH_IDX]  # (T, 144, 144)
    lev = torch.where(lev.abs() < 1e30, lev, torch.nan)
    time = pd.DatetimeIndex(np.asarray(td["time"])) if "time" in td.keys() else None
    annual = monthly_to_yearly(lev[None], time_dim=1)[0] - climatology_map  # (n_years, 144, 144)
    return annual, time[0] if time is not None else None


def load_model_anomaly_annual(scratch, run_names, data_mean_lev, data_std_lev, climatology_map):
    """Ensemble-mean, climatology-relative annual anomaly at DEPTH_IDX.

    Runs listed for one model are stacked as pseudo-members (matches other
    paper_figures_*.py scripts), then averaged over that member axis --
    unlike the RMSE plots, the Hovmoller only ever shows the ensemble mean.
    """
    members = [
        load_run_lev(f"{scratch}/generated_data/{name}", data_mean_lev, data_std_lev)[
            :, :, DEPTH_IDX
        ]
        for name in run_names
    ]  # each (n_members, T, 144, 144)
    t_min = min(m.shape[1] for m in members)
    raw = torch.cat([m[:, :t_min] for m in members], dim=0)  # (n_members, T, 144, 144)
    annual = (
        monthly_to_yearly(raw, time_dim=1) - climatology_map[None, None]
    )  # (n_members, n_years, 144, 144)
    return annual.nanmean(dim=0)  # (n_years, 144, 144)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap.name}_trunc", cmap(np.linspace(minval, maxval, n))
    )


def plot_hovmoller(ax, region_anomaly, vmin=-5, vmax=5):
    """region_anomaly: (n_years, lat_sub, lon). Collapses longitude, plots (lat_sub, n_years)."""
    data_plot = region_anomaly.nanmean(dim=-1)  # (n_years, lat_sub)
    levels = np.linspace(vmin, vmax, 20)
    norm = BoundaryNorm(levels, ncolors=256, extend="both")
    cmap = truncate_colormap(plt.get_cmap("RdBu_r"), 0, 1)
    return ax.pcolormesh(data_plot.T.numpy(), cmap=cmap, norm=norm, shading="auto")


def _pad_front(t, n_pad):
    if n_pad <= 0:
        return t
    pad_shape = (n_pad, *t.shape[1:])
    return torch.cat([torch.full(pad_shape, float("nan"), dtype=t.dtype), t], dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfgs = {str(s.key): s for s in cfg.scenarios if str(s.key) in ("434", "534")}
    missing = [k for k in ("434", "534") if k not in scenario_cfgs]
    if missing:
        raise ValueError(
            f"paper_figures_hovmoller.py needs both '434' and '534' scenarios in "
            f"the config (it plots them side by side); missing {missing}."
        )

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    data_mean_lev = train_dataset.data_mean["lev"].squeeze(0)  # (n_depth, 144, 144)
    data_std_lev = train_dataset.data_std["lev"].view(-1)  # (n_depth,)

    climatology_map = compute_shallow_ocean_climatology(
        scratch, dataset_path=hydra_cfg.module.dataset_path
    )

    # Every scenario is expected to declare the same model keys, in the same
    # order (the model column layout is shared across both scenario rows).
    model_keys = list(scenario_cfgs["434"].models.keys())
    model_labels = {k: scenario_cfgs["434"].models[k].get("label", k) for k in model_keys}

    per_scenario = {}
    scenario_start_years = {}
    for key, scen in scenario_cfgs.items():
        target_annual, target_first = load_target_anomaly_annual(
            scratch,
            scen.target.get("realization", "r1i1p1f1"),
            scen.target.scenario,
            climatology_map,
            dataset_path=hydra_cfg.module.dataset_path,
        )
        model_annuals = {
            mk: load_model_anomaly_annual(
                scratch,
                list(scen.models[mk].runs),
                data_mean_lev,
                data_std_lev,
                climatology_map,
            )
            for mk in model_keys
            if mk in scen.models
        }
        per_scenario[key] = {"target": target_annual, "models": model_annuals, "label": scen.label}
        scenario_start_years[key] = target_first.year if target_first is not None else 2015

    # Scenarios can start at different calendar years (e.g. ssp534-over
    # branches off well after 2015) -- left-pad the shorter-history scenario
    # with nan so both share one absolute-year x-axis, rather than the
    # notebook's hardcoded PAD_534=25.
    global_start_year = min(scenario_start_years.values())
    for key in per_scenario:
        pad = scenario_start_years[key] - global_start_year
        per_scenario[key]["target"] = _pad_front(per_scenario[key]["target"], pad)
        per_scenario[key]["models"] = {
            mk: _pad_front(v, pad) for mk, v in per_scenario[key]["models"].items()
        }

    n_years = min(
        [per_scenario[k]["target"].shape[0] for k in per_scenario]
        + [v.shape[0] for scen in per_scenario.values() for v in scen["models"].values()]
    )
    for key in per_scenario:
        per_scenario[key]["target"] = per_scenario[key]["target"][:n_years]
        per_scenario[key]["models"] = {
            mk: v[:n_years] for mk, v in per_scenario[key]["models"].items()
        }

    lat = np.linspace(-90, 90, 144)
    region_names = list(REGIONS.keys())
    scenario_keys = ["434", "534"]
    col_models = model_keys + ["IPSL"]
    col_titles = [model_labels[mk] for mk in model_keys] + ["IPSL"]

    n_rows = len(region_names) * len(scenario_keys)
    n_cols = len(col_models)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), sharex=True)

    im = None
    for row_idx in range(n_rows):
        region_idx = row_idx // len(scenario_keys)
        scen_idx = row_idx % len(scenario_keys)
        region_name = region_names[region_idx]
        scen_key = scenario_keys[scen_idx]
        region_slice = REGIONS[region_name]
        lat_sub = lat[region_slice]

        y_ticks = np.linspace(0, len(lat_sub) - 1, 5)
        y_labels = np.linspace(lat_sub[0], lat_sub[-1], 5, dtype=int)

        for col_idx, model_key in enumerate(col_models):
            ax = axes[row_idx, col_idx]
            if model_key == "IPSL":
                data = per_scenario[scen_key]["target"][:, region_slice, :]
            else:
                data = per_scenario[scen_key]["models"].get(model_key)
                if data is None:
                    ax.set_visible(False)
                    continue
                data = data[:, region_slice, :]
            im = plot_hovmoller(ax, data)

            ax.set_yticks(y_ticks)
            if col_idx == 0:
                ax.set_yticklabels(y_labels, fontsize=16)
                ax.set_ylabel(
                    f"{per_scenario[scen_key]['label']}\n{region_name}\nLat (deg)", fontsize=14
                )
            else:
                ax.set_yticklabels([])

            n_time = data.shape[0]
            tick_pos = np.linspace(0, n_time - 1, 4)
            tick_labels = np.linspace(
                global_start_year, global_start_year + n_time - 1, 4, dtype=int
            )
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, rotation=60, ha="center", fontsize=13)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=16)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, extend="both")
    cbar.set_label("Temperature Anomaly (K)", fontsize=16)
    cbar.locator = ticker.MaxNLocator(nbins=6)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=14)

    plt.subplots_adjust(right=0.91, hspace=0.25, wspace=0.06)
    out_path = output_dir / "hovmoller_regional.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Hovmoller figure to {out_path}")


if __name__ == "__main__":
    main()
