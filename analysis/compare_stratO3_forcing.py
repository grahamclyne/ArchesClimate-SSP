"""Compare the model's own stratospheric-ozone-forcing signal against the real one.

Analogous to compare_aerosol_forcing.py, but for hist-stratO3 instead of
hist-piAer. Computes two global-mean tas difference time series and plots
them together:

  - model:  generated historical rollout  minus  generated hist-stratO3 rollout
  - actual: real IPSL-CM6A-LR historical (r1i1p1f1)  minus  real hist-stratO3 (r1i1p1f1)

Caveat (unlike the aerosol case): hist-stratO3 is DAMIP's *additive*
single-forcing run -- only stratospheric ozone follows its real historical
trajectory, every other forcing (GHGs, aerosol, tropospheric ozone, solar)
is held at piControl, not just the one being isolated. hist-piAer (used by
compare_aerosol_forcing.py) is the opposite construction: everything real
except the one forcing being isolated. So "historical minus hist-stratO3"
here reflects the combined difference of ALL forcings between the two runs,
not a clean stratospheric-ozone-only signal the way "historical minus
hist-piAer" was. There is no AerChemMIP-style subtractive counterpart
("hist-piO3"-like) downloaded for our two source models -- see the
piClim-O3/histSST-piO3 ESGF checks earlier in this project's history, which
found neither available for IPSL-CM6A-LR or CanESM5.

Locates the two generated rollouts by recomputing their out_dir from cfg
(same as plot_global_mean_rollout.py), using the *unmasked* historical/
hist_stratO3 runs (no zero_spatial_forcing_indices override) -- pass the
same module=/name=/cluster=/inference.ckpt_fname=/inference.use_ema=
overrides those rollouts were submitted with; inference.target is ignored
(both targets are built internally).

Usage:
    python -m analysis.compare_stratO3_forcing \\
        module=deterministic_damip_pf4_energy_score_w050_80_10 \\
        name=deterministic_damip_pf4_energy_score_w050_80_10_0 \\
        cluster=cleps inference.ckpt_fname='step-step=040000.ckpt' inference.use_ema=True

Writes analysis/outputs/compare_stratO3_forcing/stratO3_forcing_tas.png.
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
    strato3_dir = _run_dir_for_target(cfg, "hist_stratO3")
    pred_hist = load_run(hist_dir, var_idx, data_mean, data_std)[0]  # (T, 144, 144)
    pred_strato3 = load_run(strato3_dir, var_idx, data_mean, data_std)[0]
    t_common = min(pred_hist.shape[0], pred_strato3.shape[0])
    # Row 0 of both series is a NaN lead-in (see load_run's docstring) --
    # drop it before annualizing so it doesn't contaminate year 0.
    model_hist_annual = to_annual(pred_hist[2:t_common][None])[0] - 273.15
    model_strato3_annual = to_annual(pred_strato3[2:t_common][None])[0] - 273.15
    model_diff = lat_weighted_mean(model_hist_annual) - lat_weighted_mean(model_strato3_annual)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(model_diff.numpy(), label="model (generated historical - generated hist-stratO3)")
    if cfg.get("include_actual", True):
        # --- actual: real driving data -------------------------------------
        actual_hist, _ = load_target(cfg.cluster.data_path, REALIZATION, "historical", var_idx)
        actual_strato3, _ = load_target(cfg.cluster.data_path, REALIZATION, "hist-stratO3", var_idx)
        t_common_actual = min(actual_hist.shape[0], actual_strato3.shape[0])
        actual_hist_annual = to_annual(actual_hist[:t_common_actual][None])[0] - 273.15
        actual_strato3_annual = to_annual(actual_strato3[:t_common_actual][None])[0] - 273.15
        actual_diff = lat_weighted_mean(actual_hist_annual) - lat_weighted_mean(actual_strato3_annual)
        ax.plot(
            actual_diff.numpy(),
            label="actual (IPSL historical - IPSL hist-stratO3)",
            linestyle="--",
        )
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Year since rollout start")
    ax.set_ylabel("Global-mean tas difference (K)")
    ax.set_title(f"{cfg.name}: stratospheric-ozone forcing signal (historical minus hist-stratO3)")
    ax.legend(fontsize=8)

    out_dir = Path("analysis/outputs/compare_stratO3_forcing")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stratO3_forcing_tas.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
