r"""SSP5-8.5 extrapolation figure: yearly trend + end-of-century distribution.

Adapted from analysis/analysis_585.ipynb (final cell, "extrapolation.pdf").
For one scenario (--scenario, typically "585"/ssp5-8.5 -- the notebook's
whole point was checking extrapolation onto an *unseen* forcing pathway, not
one used at training/val time), plots:
  - top: yearly global-mean tas trend (mean +/- std across ensemble
    members/seeds) for each model declared in the scenario's `models` dict,
    plus the IPSL target;
  - bottom: end-of-century (last `final_decade_years` years) KDE of the raw
    (un-spatially-averaged) tas distribution across all grid cells, members,
    and years -- same models + IPSL.

Named "585" for historical reasons (this is where the reference notebook
lived); nothing here is hardcoded to ssp585 specifically -- it just needs a
scenario key in the config whose target.scenario names an SSP the model
never saw as val/test forcing during training, and correspondingly whose
rollouts were generated with inference.target=test (SCENARIO_DOMAIN
convention used elsewhere: ssp585 <-> test, ssp434 <-> val).

Unlike paper_figures_table.py, the IPSL target here can average over
*multiple* realizations for its own ensemble-spread band (matching the
notebook's r1i1p1f1 + r2i1p1f1 pair) -- set `target.realizations: [...]` in
the scenario config; falls back to the single `target.realization` (band
width 0) if omitted.

Usage:
    python analysis/paper_figures_585_analysis.py \\
        --config analysis/configs/paper_figures_table_example.yaml \\
        --scenario 585
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from omegaconf import OmegaConf
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run, to_annual
from analysis.paper_figures_table import load_target_annual
from analysis.plot_style import MODEL_COLORS

_FALLBACK_COLORS = ["tab:purple", "tab:brown", "tab:pink", "tab:cyan", "tab:olive"]

# Standard climatological latitude bands (Tropic of Cancer/Capricorn at
# +-23.5 deg, polar circles at +-66.5 deg) used to derive each KDE mode's
# label from the ACTUAL grid cells that populate it, rather than assuming
# modes fall in ascending-temperature = ascending-|latitude| order. That
# assumption is wrong here: Antarctica's interior is cold enough (elevation,
# not just latitude) to form its own separate mode well below Antarctica's
# own coastal/peripheral mode, while the Arctic (ocean/ice-moderated, no
# elevation effect) lands in between at temperatures that look "mid-latitude"
# by value alone -- verified by checking which grid cells actually populate
# each mode's temperature window (see conversation/analysis, not re-derived
# on every run for speed).
_LAT_BANDS = [
    ("Antarctic", -90.0, -66.5),
    ("Southern mid-latitude", -66.5, -23.5),
    ("Tropical", -23.5, 23.5),
    ("Northern mid-latitude", 23.5, 66.5),
    ("Arctic", 66.5, 90.0),
]


def _kde_local_maxima(data, n_grid=2000, prominence_frac=0.03):
    """(x, density) of each local maximum of a 1-D KDE of `data`, ascending x.

    prominence_frac: a maximum is kept only if its density is at least this
    fraction of the global peak density, to drop spurious small wiggles.
    """
    data = np.asarray(data)
    data = data[np.isfinite(data)]
    kde = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), n_grid)
    density = kde(xs)
    maxima_idx = argrelextrema(density, np.greater)[0]
    threshold = density.max() * prominence_frac
    return [(xs[i], density[i]) for i in maxima_idx if density[i] >= threshold]


def _mode_label_from_geography(mode_x, pooled, lats, window=1.5, majority_frac=0.5):
    """Label a KDE mode by which latitude band actually populates it.

    pooled: (..., lat, lon) tensor of the same pooled values the KDE for
    this mode was built from (e.g. final-decade yearly-mean tas, members and
    years flattened into leading dims). lats: (lat,) degrees, same
    convention as pooled's second-to-last axis (index 0 = -90).

    Finds every grid cell whose value falls within `window` K of the mode's
    peak location, bins their latitudes into _LAT_BANDS, and returns that
    band's name if it holds >= majority_frac of the mass; otherwise returns
    a "Mixed (band1/band2)" label naming the top two bands, so a genuinely
    ambiguous mode doesn't get silently mislabeled.
    """
    mask = (pooled > mode_x - window) & (pooled < mode_x + window)
    lat_idx = mask.nonzero(as_tuple=True)[-2]
    if lat_idx.numel() == 0:
        return "Mode"
    cell_lats = lats[lat_idx]
    counts = []
    for name, lo, hi in _LAT_BANDS:
        counts.append(((cell_lats >= lo) & (cell_lats < hi)).sum().item())
    counts = np.array(counts)
    frac = counts / counts.sum()
    order = np.argsort(frac)[::-1]
    top_name = _LAT_BANDS[order[0]][0]
    if frac[order[0]] >= majority_frac:
        return top_name
    second_name = _LAT_BANDS[order[1]][0]
    return f"Mixed ({top_name}/{second_name})"


def annotate_kde_modes(ax, data, pooled=None, lats=None, color="black"):
    """Label each detected mode of the pooled grid-cell distribution on `ax`.

    Modes are detected from `data` (typically the IPSL target's own pooled
    1-D distribution, as the physical ground truth defining each regime).
    If `pooled` (the same data, un-flattened, with a trailing (lat, lon)
    shape) and `lats` are given, each mode is labeled by the actual
    latitude-band composition of the grid cells populating it (see
    _mode_label_from_geography); otherwise falls back to generic numbering.
    A dotted guide line connects each label to its peak.
    """
    modes = _kde_local_maxima(data)
    if pooled is not None and lats is not None:
        labels = [_mode_label_from_geography(x, pooled, lats) for x, _y in modes]
    else:
        labels = [f"Mode {i + 1}" for i in range(len(modes))]
    ymax = ax.get_ylim()[1]
    for (x, y), label in zip(modes, labels):
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x, y + 0.10 * ymax),
            ha="center",
            fontsize=14,
            color=color,
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, ls=":"),
        )


def lat_weighted_mean(data, lats):
    """data: (..., n_years, lat, lon), lats: (lat,) in degrees -> (..., n_years)."""
    weights = torch.cos(torch.deg2rad(lats)).view(*([1] * (data.dim() - 2)), -1, 1)
    mask = ~torch.isnan(data)
    effective_weights = weights * mask.float()
    return torch.nan_to_num(data, nan=0.0).mul(effective_weights).sum(
        dim=(-1, -2)
    ) / effective_weights.sum(dim=(-1, -2))


def plot_ranges(ax, data, dates, color, label):
    """data: (n_members, n_years) -> mean line +/- std band."""
    means = data.mean(axis=0)
    std = data.std(axis=0)
    ax.plot(dates, means, color=color, lw=0.7)
    ax.fill_between(dates, means - std, means + std, color=color, alpha=0.3, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    parser.add_argument("--scenario", required=True, help="Scenario key to plot (e.g. '585').")
    parser.add_argument(
        "--branch-year",
        type=int,
        default=None,
        help="If set, draws a vertical dashed marker at Jan 1 of this year "
        "(matches the notebook's 2081 marker); overrides the config's "
        "branch_year if both are given.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    final_decade_years = cfg.get("final_decade_years", 10)
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfg = next(s for s in cfg.scenarios if str(s.key) == str(args.scenario))
    branch_year = args.branch_year if args.branch_year is not None else cfg.get("branch_year", None)

    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train", config_module=cfg.config_module
    )
    surface_variables = list(hydra_cfg.module.surface_variables)
    variable = cfg.get("variable", "tas")
    var_idx = surface_variables.index(variable)
    data_mean = train_dataset.data_mean["surface"].squeeze(1)
    data_std = train_dataset.data_std["surface"].view(-1)

    realizations = list(
        scenario_cfg.target.get("realizations")
        or [scenario_cfg.target.get("realization", "r1i1p1f1")]
    )
    target_parts, target_first = [], None
    for r in realizations:
        annual, first, _ = load_target_annual(
            scratch,
            r,
            scenario_cfg.target.scenario,
            var_idx,
            dataset_path=hydra_cfg.module.dataset_path,
        )
        target_parts.append(annual)
        if target_first is None:
            target_first = first
    target_annual = torch.cat(target_parts, dim=0)  # (n_realizations, n_years, 144, 144)

    model_labels = {k: scenario_cfg.models[k].get("label", k) for k in scenario_cfg.models}
    # Keyed by the model's own display label (e.g. "AC-SSP") against the
    # shared cross-figure palette in analysis/plot_style.py, not
    # positionally -- so the same named model gets the same color in every
    # figure it appears in. Falls back to a fixed (still non-positional)
    # color for any label plot_style doesn't recognize.
    model_colors = {
        k: MODEL_COLORS.get(label, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        for i, (k, label) in enumerate(model_labels.items())
    }
    model_annuals = {}
    for key, model_cfg in scenario_cfg.models.items():
        members = [
            load_run(f"{scratch}/generated_data/{name}", var_idx, data_mean, data_std)
            for name in model_cfg.runs
        ]
        t_min = min(m.shape[1] for m in members)
        raw = torch.cat([m[:, :t_min] for m in members], dim=0)  # (n_members, T, 144, 144)
        max_members = model_cfg.get("max_members", None)
        if max_members is not None:
            # Cap member count -- e.g. to match the target's own available
            # realization count for an apples-to-apples spread comparison,
            # rather than silently plotting more model members than target
            # members.
            raw = raw[:max_members]
        model_annuals[key] = to_annual(raw)  # (n_members, n_years, 144, 144)

    T_yrs = min([target_annual.shape[1]] + [v.shape[1] for v in model_annuals.values()])
    target_annual = target_annual[:, :T_yrs]
    model_annuals = {k: v[:, :T_yrs] for k, v in model_annuals.items()}
    start_year = target_first.year if target_first is not None else 2015
    dates = pd.date_range(start=f"{start_year}-01-01", periods=T_yrs, freq="YS")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), constrained_layout=True)

    lats = torch.linspace(-90, 90, target_annual.shape[-2])
    for key, annual in model_annuals.items():
        plot_ranges(
            ax1, lat_weighted_mean(annual, lats), dates, model_colors[key], model_labels[key]
        )
    ipsl_label = f"IPSL {scenario_cfg.label}"
    plot_ranges(ax1, lat_weighted_mean(target_annual, lats), dates, "black", ipsl_label)

    if branch_year is not None:
        import datetime

        ax1.axvline(x=datetime.datetime(branch_year, 1, 1), color="black", linestyle=":")
    ax1.set_xlabel("Year", fontsize=20)
    ax1.set_ylabel(f"{variable} (K)" if variable == "tas" else variable, fontsize=20)
    target_mean = lat_weighted_mean(target_annual, lats)
    ax1.set_ylim(target_mean.min().item() - 0.3, target_mean.max().item() + 1)

    legend_elements = [
        Line2D([0], [0], color=model_colors[k], lw=2, label=model_labels[k])
        for k in scenario_cfg.models
    ] + [Line2D([0], [0], color="black", lw=2, label=ipsl_label)]
    ax1.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=20)
    ax1.tick_params(axis="both", labelsize=20)

    for key, annual in model_annuals.items():
        sns.kdeplot(
            annual[:, -final_decade_years:].reshape(-1),
            fill=True,
            label=model_labels[key],
            color=model_colors[key],
            ax=ax2,
        )
    sns.kdeplot(
        target_annual[:, -final_decade_years:].reshape(-1),
        fill=True,
        label=ipsl_label,
        color="black",
        ax=ax2,
    )

    ax2.set_xlabel(f"{variable} (K)" if variable == "tas" else variable, fontsize=20)
    ax2.set_ylabel("Density", fontsize=20)
    ax2.legend(fontsize=20)
    ax2.tick_params(axis="both", labelsize=20)
    if variable == "tas" and cfg.get("annotate_kde_modes", True):
        # Modes are detected and labeled from the IPSL target's own pooled
        # distribution (the physical ground truth defining each regime), not
        # the model's -- labels come from each mode's actual grid-cell
        # latitude composition (_mode_label_from_geography), not an assumed
        # ascending-temperature order.
        final_decade_target = target_annual[:, -final_decade_years:]
        annotate_kde_modes(
            ax2,
            final_decade_target.reshape(-1).numpy(),
            pooled=final_decade_target,
            lats=lats,
        )

    out_path = output_dir / f"extrapolation_{scenario_cfg.key}.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved extrapolation figure to {out_path}")


if __name__ == "__main__":
    main()
