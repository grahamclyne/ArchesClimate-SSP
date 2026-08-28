"""Spatially explicit version of compare_aerosol_forcing.py / compare_stratO3_forcing.py.

For a chosen forcing-isolation experiment (aerosol: hist-piAer, or
stratospheric ozone: hist-stratO3), computes the "historical minus X"
global tas difference map, averaged over the first 10 and last 10 years of
the rollout, and plots a 2x3 grid (rows: first 10yr / last 10yr; columns):

  1. pred   -- model: mean(generated historical) - mean(generated X)
  2. target -- actual: mean(real historical) - mean(real X)
  3. diff   -- pred - target (the model's bias in this forcing signal)

Same run-location/loading logic as compare_aerosol_forcing.py /
compare_stratO3_forcing.py -- pass the same module=/name=/cluster=/
inference.ckpt_fname=/inference.use_ema=/inference.num_rollout_steps=
overrides those rollouts were submitted with. inference.target is ignored
(both targets are built internally from the +experiment= choice).

Usage:
    python -m analysis.compare_forcing_spatial +experiment=aerosol \\
        module=deterministic_damip_pf4_energy_score_w050_80_10 \\
        name=deterministic_damip_pf4_energy_score_w050_80_10_0 \\
        cluster=cleps inference.ckpt_fname='step-step=040000.ckpt' \\
        inference.use_ema=True inference.num_rollout_steps=1978

    python -m analysis.compare_forcing_spatial +experiment=stratO3 ...

Writes analysis/outputs/compare_forcing_spatial/{experiment}_spatial_tas.png.
"""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig, OmegaConf

from analysis.compare_runs import load_run, load_target
from analysis.long_rollout import compute_rollout_out_dir

REALIZATION = "r1i1p1f1"

EXPERIMENTS = {
    "aerosol": {"target": "hist_piAer", "actual_exp": "hist-piAer", "label": "hist-piAer"},
    "stratO3": {"target": "hist_stratO3", "actual_exp": "hist-stratO3", "label": "hist-stratO3"},
}


def _run_dir_for_target(cfg: DictConfig, target: str) -> str:
    cfg = OmegaConf.merge(cfg, {"inference": {"target": target}})
    return compute_rollout_out_dir(cfg)


def _last_n_years_mean(monthly: torch.Tensor, n_years: int = 10) -> torch.Tensor:
    """monthly: (T, lat, lon) -> (lat, lon), mean over the last n_years*12 months."""
    n_months = n_years * 12
    return torch.nanmean(monthly[-n_months:], dim=0)


def _first_n_years_mean(monthly: torch.Tensor, n_years: int = 10, lead_in: int = 0) -> torch.Tensor:
    """monthly: (T, lat, lon) -> (lat, lon), mean over the first n_years*12 months.

    lead_in skips leading rows before averaging -- model rollouts carry a
    2-row NaN lead-in (see compare_runs.load_run's docstring) that would
    otherwise get included (as NaN, harmlessly, but wastefully) or shift the
    window; real memmap data has no lead-in (lead_in=0).
    """
    n_months = n_years * 12
    return torch.nanmean(monthly[lead_in : lead_in + n_months], dim=0)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    experiment = cfg.experiment
    if experiment not in EXPERIMENTS:
        raise ValueError(f"experiment must be one of {list(EXPERIMENTS)}, got {experiment!r}")
    spec = EXPERIMENTS[experiment]

    surface_variables = list(cfg.module.surface_variables)
    var_idx = surface_variables.index("tas")

    dataset = hydra.utils.instantiate(cfg.dataloader.dataset, domain="historical")
    data_mean = dataset.data_mean["surface"].squeeze(1)
    data_std = dataset.data_std["surface"].view(-1)

    # --- model: generated rollouts ----------------------------------------
    hist_dir = _run_dir_for_target(cfg, "historical")
    x_dir = _run_dir_for_target(cfg, spec["target"])
    pred_hist = load_run(hist_dir, var_idx, data_mean, data_std)[0]  # (T, 144, 144)
    pred_x = load_run(x_dir, var_idx, data_mean, data_std)[0]

    # --- actual: real driving data -----------------------------------------
    actual_hist, _ = load_target(cfg.cluster.data_path, REALIZATION, "historical", var_idx)
    actual_x, _ = load_target(cfg.cluster.data_path, REALIZATION, spec["actual_exp"], var_idx)

    windows = [
        (
            "first 10yr",
            _first_n_years_mean(pred_hist, lead_in=2) - _first_n_years_mean(pred_x, lead_in=2),
            _first_n_years_mean(actual_hist) - _first_n_years_mean(actual_x),
        ),
        (
            "last 10yr",
            _last_n_years_mean(pred_hist) - _last_n_years_mean(pred_x),
            _last_n_years_mean(actual_hist) - _last_n_years_mean(actual_x),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4))
    for row, (window_label, pred_diff, actual_diff) in zip(axes, windows):
        bias = pred_diff - actual_diff
        vmax = torch.nan_to_num(torch.abs(torch.stack([pred_diff, actual_diff]))).max().item()
        bias_vmax = torch.nan_to_num(torch.abs(bias)).max().item()
        panels = [
            (pred_diff, f"pred ({window_label}): historical - {spec['label']} (model)", vmax),
            (actual_diff, f"target ({window_label}): historical - {spec['label']} (actual)", vmax),
            (bias, f"diff ({window_label}): pred - target", bias_vmax),
        ]
        for ax, (data, title, v) in zip(row, panels):
            im = ax.imshow(data.numpy(), cmap="RdBu_r", vmin=-v, vmax=v, origin="lower")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="K")
    fig.suptitle(f"{cfg.name}: {spec['label']} forcing signal, first/last 10yr mean tas")
    fig.tight_layout()

    out_dir = Path("analysis/outputs/compare_forcing_spatial")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment}_spatial_tas.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
