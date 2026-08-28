import hydra
import lightning as L
from omegaconf import DictConfig

from ArchesClimate.model.base_module import load_module


def compute_rollout_out_dir(cfg: DictConfig) -> str:
    """Reconstruct a rollout's out_dir from cfg alone -- the same computation
    main() below uses to pick where to write, exposed so other scripts (e.g.
    analysis/plot_global_mean_rollout.py) can locate an existing rollout's
    output from the identical module=/cluster=/name=/inference.* overrides
    it was generated with, without needing to duplicate this logic (which
    has drifted before) or pass a separate --run-dir by hand.
    """
    use_ema = cfg.inference.get("use_ema", True)
    ema_suffix = "_ema" if use_ema else "_noema"

    zero_spatial = list(getattr(cfg.inference, "zero_spatial_forcing_indices", None) or [])
    zero_non_spatial = list(getattr(cfg.inference, "zero_non_spatial_forcing_indices", None) or [])
    zero_forcing_source = cfg.inference.get("zero_forcing_source", "null")
    clamp_spatial = list(getattr(cfg.inference, "clamp_spatial_forcing_indices", None) or [])
    clamp_std = cfg.inference.get("clamp_spatial_forcing_std", None)
    clamp_polar_tas_std = cfg.inference.get("clamp_polar_tas_std", None)
    clamp_polar_n_rows = cfg.inference.get("clamp_polar_n_rows", 5)
    teacher_force = cfg.inference.get("teacher_force", False)
    energy_score_noise_scale = cfg.inference.get("energy_score_noise_scale", 1.0)

    forcing_suffix = ""
    if zero_spatial:
        forcing_suffix += "_zs" + "-".join(str(i) for i in sorted(zero_spatial))
    if zero_non_spatial:
        forcing_suffix += "_zns" + "-".join(str(i) for i in sorted(zero_non_spatial))
    if (zero_spatial or zero_non_spatial) and zero_forcing_source == "pi":
        # Distinguish "held at pre-industrial level" runs from the default
        # null-token ablations so they don't share an output directory.
        forcing_suffix += "_pi"
    if clamp_spatial:
        forcing_suffix += (
            "_clamp" + "-".join(str(i) for i in sorted(clamp_spatial)) + f"std{clamp_std}"
        )
    if clamp_polar_tas_std is not None:
        forcing_suffix += f"_polarclamp{clamp_polar_n_rows}std{clamp_polar_tas_std}"
    if energy_score_noise_scale != 1.0:
        # Keeps a scaled-noise run from colliding with (or being mistaken
        # for) the standard unit-variance ensemble in the same directory.
        forcing_suffix += f"_nscale{energy_score_noise_scale}"
    if teacher_force:
        # Otherwise a teacher-forced rollout silently reuses (and can
        # overwrite) the free-running rollout's directory for the same
        # forcing/clamp overrides, since teacher_force isn't reflected
        # anywhere else in out_dir below.
        forcing_suffix += "_teacherforce"

    return (  # noqa: E501
        f"{cfg.cluster.output_path}/generated_data/{cfg.name}"
        f"_{cfg.inference.num_inference_steps}"
        f"_{cfg.inference.num_members}"
        f"_{cfg.inference.scale_input_noise}"
        f"_{cfg.inference.num_rollout_steps}"
        f"_{cfg.inference.cf_guidance}"
        f"_{cfg.inference.target}"
        f"_{cfg.inference.year}"
        f"_{cfg.inference.ckpt_fname}"
        f"{ema_suffix}"
        f"{forcing_suffix}/"
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run long rollout inference for a climate model experiment.

    Args:
        cfg: Hydra DictConfig containing inference, model, cluster,
            and dataloader settings.
    """
    L.seed_everything(cfg.seed)
    dataset = hydra.utils.instantiate(cfg.dataloader.dataset, domain=cfg.inference.target)
    print(cfg)
    use_ema = cfg.inference.get("use_ema", True)
    pl_module, config = load_module(
        cfg.model_dir,
        cfg=cfg,
        ckpt_fname=cfg.inference.ckpt_fname,
        use_ema=use_ema,
        device=cfg.inference.get("device", "auto"),
    )

    index = 120 * cfg.inference.year

    seed_index = getattr(cfg.inference, "seed_index", None)
    zero_spatial = list(getattr(cfg.inference, "zero_spatial_forcing_indices", None) or [])
    zero_non_spatial = list(getattr(cfg.inference, "zero_non_spatial_forcing_indices", None) or [])
    zero_forcing_source = cfg.inference.get("zero_forcing_source", "null")
    clamp_spatial = list(getattr(cfg.inference, "clamp_spatial_forcing_indices", None) or [])
    clamp_std = cfg.inference.get("clamp_spatial_forcing_std", None)
    clamp_polar_tas_std = cfg.inference.get("clamp_polar_tas_std", None)
    clamp_polar_n_rows = cfg.inference.get("clamp_polar_n_rows", 5)
    teacher_force = cfg.inference.get("teacher_force", False)
    batch_seeds = cfg.inference.get("batch_seeds", False)
    energy_score_noise_scale = cfg.inference.get("energy_score_noise_scale", 1.0)

    out_dir = compute_rollout_out_dir(cfg)
    print(out_dir)
    pl_module.generate_rollouts(
        cfg,
        index,
        dataset,
        target_name=cfg.inference.target,
        out_dir=out_dir,
        start_member=cfg.inference.start_member,
        end_member=cfg.inference.end_member,
        flat_forcings=cfg.inference.flat_forcings,
        num_rollout_steps=cfg.inference.num_rollout_steps,
        num_perturbations_per_member=cfg.inference.num_seeds,
        debug=cfg.inference.debug,
        seed_index=seed_index,
        zero_spatial_forcing_indices=zero_spatial or None,
        zero_non_spatial_forcing_indices=zero_non_spatial or None,
        zero_forcing_source=zero_forcing_source,
        clamp_spatial_forcing_indices=clamp_spatial or None,
        clamp_spatial_forcing_std=clamp_std,
        clamp_polar_tas_std=clamp_polar_tas_std,
        clamp_polar_n_rows=clamp_polar_n_rows,
        save_all_vars=cfg.inference.get("save_all_vars", False),
        teacher_force=teacher_force,
        energy_score_noise_scale=energy_score_noise_scale,
        batch_seeds=batch_seeds,
    )


if __name__ == "__main__":
    main()
