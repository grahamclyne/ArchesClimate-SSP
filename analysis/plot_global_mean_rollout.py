"""Plot a rollout's global-mean tas against the real CMIP target it's conditioned to
reproduce -- the simplest possible sanity check that a rollout ran correctly.

Locates the rollout by recomputing its out_dir from cfg (see
compute_rollout_out_dir in long_rollout.py) -- so this takes the exact same
module=/cluster=/name=/inference.* overrides the rollout itself was
submitted with, nothing extra.

Usage (same overrides as the "Minimal inference recipe" in INFERENCE.md):
    python -m analysis.plot_global_mean_rollout \\
        module=<module_config_name> name=<checkpoint_run_name> cluster=<your cluster.yaml> \\
        inference.ckpt_fname='step-step=NNNNNN.ckpt' inference.target=val inference.use_ema=True

Writes <out_dir>/global_mean_tas.png.
"""

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from analysis.compare_runs import lat_weighted_mean, load_run, load_target, to_annual
from analysis.long_rollout import compute_rollout_out_dir

# See dataloaders/cmip_random_lead_time.py's filename_filters -- val/test come
# from the module config's own filter lists, the rest are fixed scenario names.
_FIXED_TARGET_SCENARIOS = {
    "val1": "ssp119",
    "val2": "ssp370",
    "val3": "ssp534-over",
    "abrupt": "abrupt-4xCO2",
}
REALIZATION = "r1i1p1f1"  # every scenario currently published on the hub uses this one


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    target = cfg.inference.target
    if target in _FIXED_TARGET_SCENARIOS:
        scenario = _FIXED_TARGET_SCENARIOS[target]
    elif target == "val":
        scenario = cfg.module.val_filter[0]
    elif target == "test":
        scenario = cfg.module.test_filter[0]
    else:
        raise ValueError(f"Don't know which CMIP scenario inference.target={target!r} maps to.")

    surface_variables = list(cfg.module.surface_variables)
    var_idx = surface_variables.index("tas")

    run_dir = compute_rollout_out_dir(cfg)
    dataset = hydra.utils.instantiate(cfg.dataloader.dataset, domain=target)
    data_mean = dataset.data_mean["surface"].squeeze(1)
    data_std = dataset.data_std["surface"].view(-1)

    pred_monthly = load_run(run_dir, var_idx, data_mean, data_std)[0]  # (T, 144, 144)
    target_monthly, _ = load_target(
        cfg.cluster.data_path, REALIZATION, scenario, var_idx, dataset_path=cfg.module.dataset_path
    )
    t_common = min(pred_monthly.shape[0], target_monthly.shape[0] - 1)
    pred_annual = to_annual(pred_monthly[:t_common][None])[0] - 273.15
    target_annual = to_annual(target_monthly[1 : 1 + t_common][None])[0] - 273.15
    # Row 0 is contaminated by load_run's NaN lead-in -- see its own docstring.
    global_pred = lat_weighted_mean(pred_annual[1:])
    global_target = lat_weighted_mean(target_annual[1:])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(global_pred, label="AC-SSP")
    ax.plot(global_target, label="IPSL-CM6A-LR", linestyle="--")
    ax.set_xlabel("Year since rollout start")
    ax.set_ylabel("Global-mean tas (°C)")
    ax.set_title(f"{cfg.name}, {scenario}")
    ax.legend()

    out_path = f"{run_dir}/global_mean_tas.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
