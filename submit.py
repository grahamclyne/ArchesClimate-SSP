# submitit file
import hydra
import submitit
from omegaconf import DictConfig

from ArchesClimate.main_hydra import main as hydra_main


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Submit a job using submitit.

    Args:
        cfg: Hydra configuration object.
    """
    aex = submitit.AutoExecutor(
        folder=cfg.cluster.folder,
        cluster="slurm",
    )
    aex.update_parameters(**cfg.cluster.launcher)  # original launcher
    aex.submit(hydra_main, cfg)


if __name__ == "__main__":
    main()
