"""Generate the README's illustrative tas rollout GIFs: AC-SSP vs. IPSL-CM6A-LR.

Two animations, from the single deterministic_damip_pf4_energy_score_w050_80_10
run at step-040000 EMA -- the model the paper's main figures are built on:

  - a Robinson-projection map GIF: annual-mean tas, AC-SSP vs. IPSL-CM6A-LR,
    SSP4-3.4 only.
  - a global-mean timeseries GIF: all three scenarios' (SSP4-3.4, SSP5-3.4-over,
    SSP5-8.5) lat-weighted annual-mean tas rolling out year by year, predicted
    vs. target, showing how/where they diverge.

Not part of the paper figure pipeline -- this is purely a README illustration.

Usage:
    python -m analysis.make_readme_gif --output-dir assets
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import animation

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import lat_weighted_mean, load_run, load_target, to_annual

CONFIG_MODULE = "deterministic_damip_pf4_energy_score_w050_80_10"
REALIZATION = "r1i1p1f1"
RUN_TEMPLATE = (
    "deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_1020_1_{domain}_0_"
    "step-step=040000.ckpt_ema"
)
SCENARIOS = [
    # (label, rollout domain, target CMIP scenario name)
    ("SSP4-3.4", "val", "ssp434"),
    ("SSP5-3.4-over", "val3", "ssp534-over"),
    ("SSP5-8.5", "test", "ssp585"),
]

BG = "#0d1117"  # GitHub dark-mode background
FG = "#c9d1d9"  # GitHub dark-mode body text
GRID = "#30363d"  # GitHub dark-mode border/muted color

LAT = np.linspace(-90, 90, 144)
LON = np.linspace(-2, 358, 144)


def _style():
    sns.set_theme(style="darkgrid", rc={"axes.facecolor": BG, "figure.facecolor": BG})
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": FG,
            "axes.labelcolor": FG,
            "axes.edgecolor": GRID,
            "xtick.color": FG,
            "ytick.color": FG,
            "grid.color": GRID,
            "font.size": 11,
        }
    )


def load_scenario_data(hydra_cfg, train_dataset, scratch, var_idx):
    """Returns {label: (pred_annual [n_years,144,144] degC, target_annual [...], global_pred [n_years], global_target [n_years])}."""
    data_mean = train_dataset.data_mean["surface"].squeeze(1)
    data_std = train_dataset.data_std["surface"].view(-1)

    out = {}
    for label, domain, scenario in SCENARIOS:
        run_dir = os.path.join(scratch, "generated_data", RUN_TEMPLATE.format(domain=domain))
        pred_monthly = load_run(run_dir, var_idx, data_mean, data_std)[0]  # (T, 144, 144)

        target_monthly, target_time = load_target(
            scratch, REALIZATION, scenario, var_idx, dataset_path=hydra_cfg.module.dataset_path
        )
        t_common = min(pred_monthly.shape[0], target_monthly.shape[0] - 1)
        pred_monthly = pred_monthly[:t_common]
        target_aligned = target_monthly[1 : 1 + t_common]

        pred_annual = to_annual(pred_monthly[None])[0] - 273.15
        target_annual = to_annual(target_aligned[None])[0] - 273.15
        # Year 0 is contaminated by load_run's 2-month NaN lead-in (plain,
        # non-NaN-aware annual mean) -- drop it, same fix as before.
        pred_annual = pred_annual[1:]
        target_annual = target_annual[1:]

        global_pred = lat_weighted_mean(pred_annual)
        global_target = lat_weighted_mean(target_annual)

        start_year = (target_time[0].year + 2) if target_time is not None else 0
        years = np.arange(start_year, start_year + pred_annual.shape[0])

        out[label] = {
            "pred_annual": pred_annual,
            "target_annual": target_annual,
            "global_pred": global_pred.numpy(),
            "global_target": global_target.numpy(),
            "years": years,
        }
    return out


def _year_axis(data):
    """Common calendar-year frame axis spanning every scenario's actual coverage.

    SSP5-3.4-over only has target/rollout data from its branch year (~2042)
    onward -- indexing frames by raw array position (instead of calendar
    year) would show its first available year under every other scenario's
    *2017* label. Index by year instead so each row/line only appears once
    its own data actually starts.
    """
    lo = min(d["years"][0] for d in data.values())
    hi = max(d["years"][-1] for d in data.values())
    return np.arange(lo, hi + 1)


def make_map_gif(data, scenarios, output_path, fps):
    import cartopy.crs as ccrs

    year_axis = _year_axis({label: data[label] for label, _, _ in scenarios})
    n_frames = len(year_axis)
    blank = np.full((144, 144), np.nan)
    all_target = np.concatenate([data[label]["target_annual"].numpy().ravel() for label, _, _ in scenarios])
    vmin, vmax = np.nanpercentile(all_target, [1, 99])

    fig, axes = plt.subplots(
        len(scenarios),
        2,
        figsize=(9, 3.1 * len(scenarios)),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
        squeeze=False,
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.06, hspace=0.15, wspace=0.02)

    meshes = {}
    for row, (label, _, _) in enumerate(scenarios):
        for col, title in enumerate(["AC-SSP", "IPSL-CM6A-LR"]):
            ax = axes[row, col]
            ax.set_global()
            ax.spines["geo"].set_visible(False)
            ax.patch.set_alpha(0.0)
            mesh = ax.pcolormesh(
                LON,
                LAT,
                np.zeros((144, 144)),
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
            ax.coastlines(color=FG, linewidth=0.5)
            if row == 0:
                ax.set_title(title, color=FG, fontsize=13, pad=8)
            if len(scenarios) > 1:
                ax.text(
                    -0.05,
                    0.5,
                    label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    color=FG,
                    fontsize=11,
                    visible=(col == 0),
                )
            meshes[(label, col)] = mesh

    cbar = fig.colorbar(
        meshes[(scenarios[0][0], 0)], ax=axes, shrink=0.6, pad=0.02, aspect=30, label="tas (°C)"
    )
    cbar.ax.yaxis.label.set_color(FG)
    cbar.ax.tick_params(colors=FG)
    cbar.outline.set_edgecolor(GRID)

    title_prefix = scenarios[0][0] + " -- " if len(scenarios) == 1 else ""
    year_text = fig.suptitle("", color=FG, fontsize=14)

    def update(frame):
        year = year_axis[frame]
        for label, _, _ in scenarios:
            d = data[label]
            hit = np.nonzero(d["years"] == year)[0]
            if len(hit):
                idx = int(hit[0])
                pred_map = d["pred_annual"][idx].numpy()
                target_map = d["target_annual"][idx].numpy()
            elif year < d["years"][0]:
                pred_map = target_map = blank  # scenario hasn't started yet
            else:
                pred_map = d["pred_annual"][-1].numpy()  # hold last frame past scenario end
                target_map = d["target_annual"][-1].numpy()
            meshes[(label, 0)].set_array(pred_map.ravel())
            meshes[(label, 1)].set_array(target_map.ravel())
        year_text.set_text(f"{title_prefix}{year}")
        return [*meshes.values(), year_text]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wrote {output_path} ({n_frames} frames)")


def make_timeseries_gif(data, output_path, fps):
    year_axis = _year_axis(data)
    n_frames = len(year_axis)
    colors = sns.color_palette("flare", n_colors=len(SCENARIOS))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlabel("Year")
    ax.set_ylabel("Global-mean tas (°C)")
    ax.spines[["top", "right"]].set_visible(False)

    lines = {}
    for (label, _, _), color in zip(SCENARIOS, colors):
        (pred_line,) = ax.plot([], [], color=color, lw=2, label=f"{label} (AC-SSP)")
        (target_line,) = ax.plot([], [], color=color, lw=1.3, ls="--", alpha=0.75, label=f"{label} (IPSL)")
        lines[label] = (pred_line, target_line)

    all_years = np.concatenate([d["years"] for d in data.values()])
    all_vals = np.concatenate(
        [d["global_pred"] for d in data.values()] + [d["global_target"] for d in data.values()]
    )
    ax.set_xlim(all_years.min(), all_years.max())
    pad = 0.05 * (np.nanmax(all_vals) - np.nanmin(all_vals))
    ax.set_ylim(np.nanmin(all_vals) - pad, np.nanmax(all_vals) + pad)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=FG)

    def update(frame):
        year = year_axis[frame]
        for label, _, _ in SCENARIOS:
            d = data[label]
            # number of this scenario's own years at or before the current
            # calendar year (0 if it hasn't started yet)
            idx = int(np.searchsorted(d["years"], year, side="right"))
            pred_line, target_line = lines[label]
            pred_line.set_data(d["years"][:idx], d["global_pred"][:idx])
            target_line.set_data(d["years"][:idx], d["global_target"][:idx])
        return [ln for pair in lines.values() for ln in pair]

    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(output_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Wrote {output_path} ({n_frames} frames)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="assets")
    parser.add_argument("--fps", type=int, default=4)
    args = parser.parse_args()

    _style()

    _, scratch, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=CONFIG_MODULE, cluster="cleps"
    )
    surface_variables = list(hydra_cfg.module.surface_variables)
    var_idx = surface_variables.index("tas")

    data = load_scenario_data(hydra_cfg, train_dataset, scratch, var_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    make_map_gif(data, SCENARIOS[:1], os.path.join(args.output_dir, "tas_rollout_map.gif"), args.fps)
    make_timeseries_gif(data, os.path.join(args.output_dir, "tas_rollout_timeseries.gif"), args.fps)


if __name__ == "__main__":
    main()
