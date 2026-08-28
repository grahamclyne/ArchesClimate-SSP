"""Combine the aerosol- and stratospheric-ozone-forcing comparisons into one grid figure.

Grid layout: rows are the aerosol (compare_aerosol_forcing.py) and
stratospheric-ozone (compare_stratO3_forcing.py) forcing comparisons,
columns are the variables in VARIABLES (tas, pr) -- sharing an x-axis,
consistent line coloring across every panel, and a single legend. No
per-panel title (for figure-in-paper use); AC-SSP is used as the model
series label instead of "model". See compare_aerosol_forcing.py and
compare_stratO3_forcing.py's docstrings for the underlying quantities each
row plots and their caveats (in particular hist-stratO3's additive, not
subtractive, single-forcing construction).

The real IPSL target series is inherently noisy year to year at this
(single-realization, global-mean) resolution, so each panel also plots a
centered rolling-mean smoothed trend of the IPSL series (solid, bold) next
to the raw one (dashed, faint), to make the comparison against AC-SSP's
smoother generated trajectory easier to read.

Locates the four generated rollouts (historical, hist_piAer, hist_stratO3,
plus historical again) the same way the two source scripts do -- by
recomputing out_dir from cfg -- so this takes the same overrides:

Usage:
    python -m analysis.compare_aerosol_stratO3_forcing \\
        module=deterministic_damip_pf4_energy_score_w050_no_damip \\
        name=deterministic_damip_pf4_energy_score_w050_no_damip \\
        cluster=cleps inference.ckpt_fname='step-step=022000.ckpt' inference.use_ema=True \\
        inference.num_rollout_steps=1978

Writes analysis/outputs/compare_aerosol_stratO3_forcing/aerosol_stratO3_forcing_tas.png.
"""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf

from analysis.compare_runs import lat_weighted_mean, load_run, load_target, to_annual
from analysis.long_rollout import compute_rollout_out_dir

REALIZATION = "r1i1p1f1"
HISTORICAL_START_YEAR = 1850

MODEL_COLOR = "tab:blue"
ACTUAL_COLOR = "tab:orange"
MODEL_LABEL = "AC-SSP"

AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11
LEGEND_FONTSIZE = 11

SMOOTH_WINDOW_YEARS = 10

# (surface_variable, y-axis label, K-offset applied before differencing).
# tas is converted from Kelvin to Celsius (the offset cancels in the
# difference, but keeps intermediate values physically readable); pr is left
# in its native kg/m^2/s units, matching this codebase's other pr panels
# (see e.g. analysis/paper_figures_global_mean.py) -- no mm/day conversion.
VARIABLES = [
    ("tas", "forcing signal (K)", 273.15),
    ("pr", "forcing signal (kg m$^{-2}$ s$^{-1}$)", 0.0),
]
ROWS = [
    ("Aerosol", "hist_piAer", "hist-piAer"),
    ("Strat. ozone", "hist_stratO3", "hist-stratO3"),
]


def _centered_rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling mean, NaN-padded at the edges (no partial windows)."""
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="valid")
    pad = (len(y) - len(smoothed)) // 2
    out = np.full_like(y, np.nan, dtype=float)
    out[pad : pad + len(smoothed)] = smoothed
    return out


def _run_dir_for_target(cfg: DictConfig, target: str) -> str:
    cfg = OmegaConf.merge(cfg, {"inference": {"target": target}})
    return compute_rollout_out_dir(cfg)


def _forcing_diff(cfg, var_idx, offset, data_mean, data_std, other_target, other_target_actual):
    hist_dir = _run_dir_for_target(cfg, "historical")
    other_dir = _run_dir_for_target(cfg, other_target)
    pred_hist = load_run(hist_dir, var_idx, data_mean, data_std)[0]
    pred_other = load_run(other_dir, var_idx, data_mean, data_std)[0]
    t_common = min(pred_hist.shape[0], pred_other.shape[0])
    # Row 0 of both series is a NaN lead-in (see load_run's docstring) --
    # drop it before annualizing so it doesn't contaminate year 0.
    model_hist_annual = to_annual(pred_hist[2:t_common][None])[0] - offset
    model_other_annual = to_annual(pred_other[2:t_common][None])[0] - offset
    model_diff = lat_weighted_mean(model_hist_annual) - lat_weighted_mean(model_other_annual)

    actual_diff = None
    if cfg.get("include_actual", True):
        actual_hist, _ = load_target(cfg.cluster.data_path, REALIZATION, "historical", var_idx)
        actual_other, _ = load_target(cfg.cluster.data_path, REALIZATION, other_target_actual, var_idx)
        t_common_actual = min(actual_hist.shape[0], actual_other.shape[0])
        actual_hist_annual = to_annual(actual_hist[:t_common_actual][None])[0] - offset
        actual_other_annual = to_annual(actual_other[:t_common_actual][None])[0] - offset
        actual_diff = lat_weighted_mean(actual_hist_annual) - lat_weighted_mean(actual_other_annual)

    return model_diff, actual_diff


def _plot_panel(ax, model_diff, actual_diff, show_legend):
    model_years = HISTORICAL_START_YEAR + np.arange(len(model_diff))
    model_np = model_diff.numpy()
    ax.plot(model_years, model_np, color=MODEL_COLOR, linestyle="--", lw=0.8, alpha=0.5, label=MODEL_LABEL)
    ax.plot(
        model_years,
        _centered_rolling_mean(model_np, SMOOTH_WINDOW_YEARS),
        color=MODEL_COLOR,
        linestyle="-",
        lw=2,
        label=f"{MODEL_LABEL} ({SMOOTH_WINDOW_YEARS}-yr mean)",
    )
    if actual_diff is not None:
        actual_years = HISTORICAL_START_YEAR + np.arange(len(actual_diff))
        actual_np = actual_diff.numpy()
        ax.plot(
            actual_years, actual_np, color=ACTUAL_COLOR, linestyle="--", lw=0.8, alpha=0.5, label="IPSL"
        )
        ax.plot(
            actual_years,
            _centered_rolling_mean(actual_np, SMOOTH_WINDOW_YEARS),
            color=ACTUAL_COLOR,
            linestyle="-",
            lw=2,
            label=f"IPSL ({SMOOTH_WINDOW_YEARS}-yr mean)",
        )
    ax.axhline(0, color="black", lw=0.5)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    if show_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    surface_variables = list(cfg.module.surface_variables)

    dataset = hydra.utils.instantiate(cfg.dataloader.dataset, domain="historical")
    data_mean = dataset.data_mean["surface"].squeeze(1)
    data_std = dataset.data_std["surface"].view(-1)

    n_rows, n_cols = len(ROWS), len(VARIABLES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)

    for row_idx, (row_label, target, target_actual) in enumerate(ROWS):
        for col_idx, (var_name, ylabel_suffix, offset) in enumerate(VARIABLES):
            var_idx = surface_variables.index(var_name)
            model_diff, actual_diff = _forcing_diff(
                cfg, var_idx, offset, data_mean, data_std, target, target_actual
            )
            ax = axes[row_idx, col_idx]
            _plot_panel(ax, model_diff, actual_diff, show_legend=(row_idx == 0 and col_idx == 0))
            ax.set_ylabel(f"{row_label} {ylabel_suffix}", fontsize=AXIS_LABEL_FONTSIZE)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Year", fontsize=AXIS_LABEL_FONTSIZE)

    fig.tight_layout()

    out_dir = Path("analysis/outputs/compare_aerosol_stratO3_forcing")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aerosol_stratO3_forcing_tas.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
