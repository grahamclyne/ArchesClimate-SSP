"""Compare the model's own aerosol-forcing signal against the real one.

Computes two global-mean tas difference time series and plots them together:

  - model:  generated historical rollout  minus  generated hist-piAer rollout
  - actual: real IPSL-CM6A-LR historical (r1i1p1f1)  minus  real hist-piAer (r1i1p1f1)

Both differences isolate the same thing -- the historical minus
aerosol-held-at-piControl effect -- once for the model's own free-running
rollouts, once for the real driving data those rollouts are conditioned on.
Locates the two generated rollouts by recomputing their out_dir from cfg
(same as plot_global_mean_rollout.py), using the *unmasked* historical/
hist_piAer runs (no zero_spatial_forcing_indices override) -- pass the same
module=/name=/cluster=/inference.ckpt_fname=/inference.use_ema= overrides
those rollouts were submitted with; inference.target is ignored (both
targets are built internally).

Usage:
    python -m analysis.compare_aerosol_forcing \\
        module=deterministic_damip_pf4_energy_score_w050_80_10 \\
        name=deterministic_damip_pf4_energy_score_w050_80_10_0 \\
        cluster=cleps inference.ckpt_fname='step-step=040000.ckpt' inference.use_ema=True

Writes analysis/outputs/compare_aerosol_forcing/aerosol_forcing_tas.png.
"""

from pathlib import Path

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from analysis.compare_runs import lat_weighted_mean, load_run, load_target, to_annual
from analysis.long_rollout import compute_rollout_out_dir

REALIZATION = "r1i1p1f1"


def _run_dir_for_target(cfg: DictConfig, target: str) -> str:
    cfg = OmegaConf.merge(cfg, {"inference": {"target": target}})
    return compute_rollout_out_dir(cfg)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    surface_variables = list(cfg.module.surface_variables)
    var_idx = surface_variables.index("tas")

    dataset = hydra.utils.instantiate(cfg.dataloader.dataset, domain="historical")
    data_mean = dataset.data_mean["surface"].squeeze(1)
    data_std = dataset.data_std["surface"].view(-1)

    # --- model: generated rollouts ----------------------------------------
    hist_dir = _run_dir_for_target(cfg, "historical")
    piaer_dir = _run_dir_for_target(cfg, "hist_piAer")
    pred_hist = load_run(hist_dir, var_idx, data_mean, data_std)[0]  # (T, 144, 144)
    pred_piaer = load_run(piaer_dir, var_idx, data_mean, data_std)[0]
    t_common = min(pred_hist.shape[0], pred_piaer.shape[0])
    # Row 0 of both series is a NaN lead-in (see load_run's docstring) --
    # drop it before annualizing so it doesn't contaminate year 0.
    model_hist_annual = to_annual(pred_hist[2:t_common][None])[0] - 273.15
    model_piaer_annual = to_annual(pred_piaer[2:t_common][None])[0] - 273.15
    model_diff = lat_weighted_mean(model_hist_annual) - lat_weighted_mean(model_piaer_annual)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(model_diff.numpy(), label="model (generated historical - generated hist-piAer)")
    if cfg.get("include_actual", True):
        # --- actual: real driving data -------------------------------------
        actual_hist, _ = load_target(cfg.cluster.data_path, REALIZATION, "historical", var_idx)
        actual_piaer, _ = load_target(cfg.cluster.data_path, REALIZATION, "hist-piAer", var_idx)
        t_common_actual = min(actual_hist.shape[0], actual_piaer.shape[0])
        actual_hist_annual = to_annual(actual_hist[:t_common_actual][None])[0] - 273.15
        actual_piaer_annual = to_annual(actual_piaer[:t_common_actual][None])[0] - 273.15
        actual_diff = lat_weighted_mean(actual_hist_annual) - lat_weighted_mean(actual_piaer_annual)
        ax.plot(
            actual_diff.numpy(),
            label="actual (IPSL historical - IPSL hist-piAer)",
            linestyle="--",
        )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Year since rollout start")
    ax.set_ylabel("Global-mean tas difference (K)")
    ax.set_title(f"{cfg.name}: aerosol forcing signal (historical minus hist-piAer)")
    ax.legend(fontsize=8)

    out_dir = Path("analysis/outputs/compare_aerosol_forcing")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aerosol_forcing_tas.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
