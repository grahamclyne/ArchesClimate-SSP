r"""Global-mean tas rollout vs IPSL target, one line per model in a scenario.

A lighter-weight companion to paper_figures_global_mean.py (which needs both
"434" and "534" scenarios and plots four variables over a single latitude
band): this script plots one variable, lat-weighted-averaged over the whole
globe, as an annual-mean time series, for every model declared under a
scenario in a paper-figures YAML config -- built for the "Forcing All-Present
Probability" ablation (deterministic_damip_pf8_ap{000,020,040,080,100}), but
generic over any scenario/config with that structure.

--scenario accepts a single key (one panel, one output file) or a
comma-separated list (one figure, side-by-side panels sharing a legend).

Usage:
    python -m analysis.paper_figures_ap_global_mean \\
        --config analysis/configs/paper_figures_table_ap_ablations.yaml \\
        --scenario 434

    python -m analysis.paper_figures_ap_global_mean \\
        --config analysis/configs/paper_figures_table_seed_stability.yaml \\
        --scenario 434,585
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run

MODEL_COLORS = ["cyan", "green", "darkorange", "magenta", "blue", "gold"]


def to_annual(monthly):
    """(..., T, lat, lon) -> (..., T // 12, lat, lon), trimming the remainder."""
    t = monthly.shape[-3]
    n_years = t // 12
    trimmed = monthly[..., : n_years * 12, :, :]
    shape = trimmed.shape
    return trimmed.view(*shape[:-3], n_years, 12, *shape[-2:]).mean(dim=-3)


def lat_weighted_global_mean(annual, lats):
    """(member, n_years, lat, lon) -> (member, n_years), cos(lat)-weighted."""
    weights = torch.cos(torch.deg2rad(lats)).view(1, 1, -1, 1)
    mask = ~torch.isnan(annual)
    w = weights * mask.float()
    filled = torch.nan_to_num(annual, nan=0.0)
    return (filled * w).sum(dim=(-1, -2)) / w.sum(dim=(-1, -2))


def plot_panel(
    ax,
    scenario_cfg,
    scratch,
    dataset_path,
    var_idx,
    variable,
    data_mean,
    data_std,
    lats,
    show_ylabel,
    target_label,
    xaxis_start=2015,
    xlabel="Year",
    max_years=None,
):
    realization = scenario_cfg.target.get("realization", "r1i1p1f1")
    target_path = (
        f"{scratch}/{dataset_path}/"
        f"{realization}_{scenario_cfg.target.scenario}_interpolation.memmap"
    )
    target_td = TensorDict.load_memmap(target_path)
    target_surface = target_td["surface"][var_idx][None]  # (1, T, lat, lon)
    target_surface = torch.where(target_surface.abs() < 1e30, target_surface, torch.nan)
    target_annual = to_annual(target_surface)
    target_global_mean = lat_weighted_global_mean(target_annual, lats)[0]  # (n_years,)
    if max_years is not None:
        target_global_mean = target_global_mean[:max_years]

    years = xaxis_start + torch.arange(target_global_mean.shape[0])
    ax.plot(years, target_global_mean.numpy(), color="black", lw=2, label=target_label, zorder=5)

    for i, (model_key, model_cfg) in enumerate(scenario_cfg.models.items()):
        members = []
        for name in model_cfg.runs:
            run_dir = os.path.join(scratch, "generated_data", name)
            members.append(load_run(run_dir, var_idx, data_mean, data_std))
        t_min = min(m.shape[1] for m in members)
        stacked = torch.cat([m[:, :t_min] for m in members], dim=0)  # (n_members, T, lat, lon)
        annual = to_annual(stacked)
        model_global_mean = lat_weighted_global_mean(annual, lats)  # (n_members, n_years)
        if max_years is not None:
            model_global_mean = model_global_mean[:, :max_years]
        mean = model_global_mean.nanmean(dim=0)
        std = model_global_mean.std(dim=0)
        n = years[: mean.shape[0]]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        label = model_cfg.get("label", model_key)
        ax.plot(n, mean.numpy(), color=color, lw=1.8, label=label)
        if model_global_mean.shape[0] > 1:
            ax.fill_between(
                n, (mean - std).numpy(), (mean + std).numpy(), color=color, alpha=0.15, lw=0
            )

    ax.set_xlabel(xlabel, fontsize=13)
    if show_ylabel:
        ax.set_ylabel(f"Global-mean {variable} (K)", fontsize=13)
    ax.set_title(scenario_cfg.label, fontsize=14)
    ax.grid(alpha=0.3, linestyle=":")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a paper-figures YAML config.")
    parser.add_argument(
        "--scenario",
        default="434",
        help="Scenario key to plot, or comma-separated keys for side-by-side "
        "panels (default: 434).",
    )
    args = parser.parse_args()
    scenario_keys = [s.strip() for s in args.scenario.split(",")]

    cfg = OmegaConf.load(args.config)
    scratch = cfg.get("scratch") or os.environ.get("SCRATCH", "/scratch/gclyne")
    output_dir = Path("plots") / cfg.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_cfgs = [next(s for s in cfg.scenarios if str(s.key) == key) for key in scenario_keys]

    dataloader = cfg.get("dataloader", "cmip_random_lead_times")
    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train",
        config_module=cfg.config_module,
        dataloader=dataloader,
    )
    dataset_path = hydra_cfg.module.dataset_path
    surface_variables = list(hydra_cfg.module.surface_variables)
    variable = cfg.get("variable", "tas")
    var_idx = surface_variables.index(variable)
    data_mean = train_dataset.data_mean["surface"].squeeze(1)
    data_std = train_dataset.data_std["surface"].view(-1)
    lats = torch.linspace(-90, 90, data_mean.shape[-2])
    target_label = cfg.get("target_label", "IPSL")
    xaxis_start = cfg.get("xaxis_start", 2015)
    xlabel = cfg.get("xlabel", "Year")
    max_years = cfg.get("max_years", None)

    fig, axes = plt.subplots(
        1, len(scenario_cfgs), figsize=(7 * len(scenario_cfgs), 6), squeeze=False
    )
    axes = axes[0]
    for i, (ax, scenario_cfg) in enumerate(zip(axes, scenario_cfgs)):
        plot_panel(
            ax,
            scenario_cfg,
            scratch,
            dataset_path,
            var_idx,
            variable,
            data_mean,
            data_std,
            lats,
            show_ylabel=(i == 0),
            target_label=target_label,
            xaxis_start=xaxis_start,
            xlabel=xlabel,
            max_years=max_years,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=10,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.06 if len(scenario_cfgs) > 1 else 1.0),
    )
    fig.suptitle(f"Global-mean {variable}", fontsize=14, y=1.14 if len(scenario_cfgs) > 1 else 0.98)

    out_path = output_dir / f"global_mean_{variable}_{'_'.join(scenario_keys)}.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved global-mean figure to {out_path}")


if __name__ == "__main__":
    main()
