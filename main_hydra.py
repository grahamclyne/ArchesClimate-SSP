import os
import warnings
from pathlib import Path
from typing import Any

# Must be set before any CUDA context is created (i.e. before the torch
# import below), so setting it via the submitting shell's environment isn't
# reliable -- submitit doesn't propagate the launching shell's exported vars
# into the job's environment. Reduces allocator fragmentation; observed
# fixing an OOM on the very first training step after a fresh warm-start
# with pushforward active from step 0 (heaviest code path hit cold, no
# warmed-up allocator pool from prior non-PF steps).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import hydra
import lightning as L
import torch
import torch._dynamo
import torch.multiprocessing
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from tensordict import stack

from ArchesClimate.model.diffusion import DCPPDiffusion

torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = True
# Default of 8 is blown through immediately by DropPath's per-layer drop_prob
# constants (droppath_coeff scaled across ~20+ backbone layers), each forcing
# its own graph specialization -- observed causing a recompile storm and GPU
# OOM on the very first training step after a fresh warm-start.
torch._dynamo.config.cache_size_limit = 64
# torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
# torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
# torch.serialization.add_safe_globals([typing.Any])
# torch.serialization.add_safe_globals([list])
# torch.serialization.add_safe_globals([collections.defaultdict])
# torch.serialization.add_safe_globals([dict])
# torch.serialization.add_safe_globals([int])
# torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
# import tracemalloc

# class CPUMemoryMonitor(L.pytorch.Callback):
#     def on_train_start(self, *args, **kwargs):
#         tracemalloc.start()

#     def on_train_epoch_end(self, trainer, pl_module):
#         snapshot = tracemalloc.take_snapshot()
#         top_stats = snapshot.statistics('lineno')
#         print("[Top 3 CPU memory lines:]")
#         for stat in top_stats[:3]:
#             print(stat)
# import tracemalloc

# class CPUMemoryMonitor(L.pytorch.Callback):
#     def on_train_start(self, *args, **kwargs):
#         tracemalloc.start()


#     def on_train_epoch_end(self, trainer, pl_module):
#         snapshot = tracemalloc.take_snapshot()
#         top_stats = snapshot.statistics('lineno')
#         print("[Top 3 CPU memory lines:]")
#         for stat in top_stats[:3]:
#             print(stat)
# Custom resolver
def calculate_length(surface_variables: Any, level_variables: Any) -> Any:
    """Calculate the total input length for surface and level variables.

    Args:
        surface_variables: List of surface variable names.
        level_variables: List of level variable names.

    Returns:
        Total number of input channels.
    """
    # three pressure levels, three times added: prev, noise, actual
    return (len(surface_variables) + len(level_variables) * 3) * 3


def get_random_code() -> Any:
    """Generate a random alphanumeric code of alternating letters and digits.

    Returns:
        A 6-character string alternating letters and digits.
    """
    import random
    import string

    # generate random code that alternates letters and numbers
    letters = random.choices(string.ascii_lowercase, k=3)
    n = random.choices(string.digits, k=3)
    return "".join([f"{a}{b}" for a, b in zip(letters, n)])


# def collate_fn(lst):
#     # print(lst)
#     keys = list(lst[0]) if not hasattr(lst[0], "keys") else list(lst[0].keys())  # noqa: E501

#     return {k: torch.stack([x[k] for x in lst]) for k in keys}
#     # print(lst)
#     keys = list(lst[0]) if not hasattr(lst[0], "keys") else list(lst[0].keys())  # noqa: E501


#     return {k: torch.stack([x[k] for x in lst]) for k in keys}
def collate_fn(lst: Any) -> Any:
    """Collate a list of TensorDicts into a single stacked TensorDict.

    Args:
        lst: List of TensorDict objects to stack.

    Returns:
        A single stacked TensorDict.
    """
    # Much faster C-level implementation for TensorDicts
    return stack(lst)


class CheckpointEveryNSteps(L.Callback):
    """Save a checkpoint every N steps.

    Instead of Lightning's default that checkpoints based on validation loss.
    """

    def __init__(
        self,
        dirpath="./",
        save_step_frequency=50000,
        prefix="checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """Initialize CheckpointEveryNSteps.

        Args:
            dirpath: Directory to save checkpoints.
            save_step_frequency: How often to save in steps.
            prefix: Prefix added to the filename when
                use_modelcheckpoint_filename=False.
            use_modelcheckpoint_filename: Use ModelCheckpoint callback's
                default filename instead of custom naming.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.dirpath = dirpath

    def on_train_batch_end(self, trainer: L.Trainer, *args, **kwargs) -> None:
        """Check if we should save a checkpoint after every train batch."""
        if not hasattr(self, "trainer"):
            self.trainer = trainer

        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            self.save()

    # def save(self, *args, trainer=None, **kwargs):
    #     if trainer is None and not hasattr(self, "trainer"):
    #         print("No trainer !")
    #         return
    #     if trainer is None:
    #         trainer = self.trainer

    #     global_step = trainer.global_step
    #     if self.use_modelcheckpoint_filename:
    #         filename = trainer.checkpoint_callback.filename
    #     else:
    #         filename = f"{self.prefix}_{global_step=}.ckpt"
    #     ckpt_path = Path(self.dirpath) / "checkpoints"
    #     ckpt_path.mkdir(exist_ok=True, parents=True)
    #     trainer.save_checkpoint(ckpt_path / filename)
    def save(self, *args, trainer=None, **kwargs) -> None:
        """Save a checkpoint using the current trainer state."""
        if trainer is None and not hasattr(self, "trainer"):
            print("No trainer !")
            return
        if trainer is None:
            trainer = self.trainer

        global_step = trainer.global_step

        # Try to get ModelCheckpoint callback
        mc_callbacks = [cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)]
        ckpt_cb = mc_callbacks[0] if mc_callbacks else None

        if self.use_modelcheckpoint_filename and ckpt_cb is not None:
            # Use ModelCheckpoint's filename template
            filename = ckpt_cb.filename or f"epoch={trainer.current_epoch}-step={global_step}.ckpt"
        else:
            # Fallback to custom naming
            filename = f"{self.prefix}_step={global_step}.ckpt"

        ckpt_dir = Path(self.dirpath) / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True, parents=True)

        ckpt_path = ckpt_dir / filename
        trainer.save_checkpoint(str(ckpt_path))

        print(f"Checkpoint saved at {ckpt_path}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig, cli_overrides: Any = None) -> None:
    """Entry point for training or testing a climate model with Hydra config.

    Args:
        cfg: Hydra DictConfig with experiment configuration.
        cli_overrides: Optional list of CLI override strings.
    """
    OmegaConf.register_new_resolver("calculate_length", calculate_length)
    # torch.multiprocessing.set_sharing_strategy('file_system')

    warnings.simplefilter(action="ignore", category=FutureWarning)
    print("Working dir", os.getcwd())

    main_node = int(os.environ.get("SLURM_PROCID", 0)) == 0
    print("is main node", main_node)

    # init some variables
    logger = None
    ckpt_path = None

    # first, check if exp exists
    if Path(cfg.exp_dir).exists():
        print("Experiment already exists. Trying to resume it.")
        exp_cfg = OmegaConf.load(Path(cfg.exp_dir) / "config.yaml")
        if cfg.resume or cfg.mode == "test":
            # we just copy cluster info
            cfg.module = exp_cfg.module
            # cfg.dataloader = exp_cfg.dataloader
            # # then we update we cli overrides
            # cli_overrides = cli_overrides #or HydraConfig.get().overrides.task
            # cli_overrides = [x for x in cli_overrides if x.startswith("+")]

            # cli_cfg = compose(overrides=cli_overrides)
            # OmegaConf.set_struct(cfg, False)  # to merge
            # cfg = OmegaConf.merge(cfg, cli_cfg)

        else:
            # check that new config and old config match
            if OmegaConf.to_yaml(cfg.module, resolve=True) != OmegaConf.to_yaml(exp_cfg.module):
                print("Module config mismatch. Exiting")
                print("Old config", OmegaConf.to_yaml(exp_cfg.module))
                print("New config", OmegaConf.to_yaml(cfg.module))

            if OmegaConf.to_yaml(cfg.dataloader, resolve=True) != OmegaConf.to_yaml(
                exp_cfg.dataloader
            ):
                print("Dataloader config mismatch. Exiting.")
                print("Old config", OmegaConf.to_yaml(exp_cfg.dataloader))
                print("New config", OmegaConf.to_yaml(cfg.dataloader))
                return

        # trying to find checkpoints
        ckpt_dir = Path(cfg.model_dir).joinpath("checkpoints")
        ckpt_dir = Path(cfg.model_dir).joinpath("checkpoints")
        if ckpt_dir.exists():
            ckpts = list(sorted(ckpt_dir.glob("*.ckpt"), key=os.path.getmtime))
            if len(ckpts):
                print("Found checkpoints", ckpts)
                ckpt_path = ckpts[-1]
                print(ckpt_path)

    # warm-start from a checkpoint of a different model
    # (restores weights + global_step, new experiment)
    nonstrict_warmstart_path = None
    warmstart_global_step = None
    if ckpt_path is None and getattr(cfg, "warmstart_ckpt_path", None):
        if cfg.get("warmstart_strict", True):
            ckpt_path = cfg.warmstart_ckpt_path
            print(f"Warm-starting from {ckpt_path}")
        else:
            # Architecture differs from the source checkpoint (e.g. new
            # submodules) -- can't go through trainer.fit(ckpt_path=...),
            # which restores optimizer state by position in param_groups and
            # will error/misalign. Applied manually below once pl_module
            # exists (weights, non-strict) and trainer exists (global_step).
            nonstrict_warmstart_path = cfg.warmstart_ckpt_path
            print(f"Warm-starting (non-strict, weights only) from {nonstrict_warmstart_path}")

    if cfg.log:
        os.environ["WANDB_DISABLE_SERVICE"] = "False"
        if cfg.cluster.wandb_mode != "online":
            # WandbLogger(offline=True) alone doesn't stop the SDK from still
            # attempting some network calls (e.g. version/update checks) --
            # on a compute node with no internet access, that call can hang
            # indefinitely. Observed directly: nopf_energy_score_v100 (job
            # 585512) logged "wandb: Network error (ConnectTimeout), entering
            # retry loop" right as rank 0 stalled entering validation, which
            # then timed out the whole 8-GPU DDP job's NCCL collective after
            # 30 min. Setting WANDB_MODE=offline as an actual env var (which
            # the SDK checks in more places than the offline= kwarg) is the
            # robust way to suppress every network attempt, not just syncing.
            os.environ["WANDB_MODE"] = "offline"
        print("wandb mode", cfg.cluster.wandb_mode)
        print(
            "wandb service",
            os.environ.get("WANDB_DISABLE_SERVICE", "variable unset"),
        )
        run_id = cfg.name + "-" + get_random_code() if cfg.cluster.use_custom_requeue else cfg.name
        # W&B derives internal artifact names (e.g. for logged Tables) as
        # "run-{id}-{key}" and rejects names over 128 chars, so long config
        # names must be shortened here. Hashing keeps the id deterministic
        # across resumes of the same run.
        if len(run_id) > 80:
            import hashlib

            run_id = run_id[:71] + "-" + hashlib.md5(run_id.encode()).hexdigest()[:8]
        logger = L.pytorch.loggers.WandbLogger(
            project=cfg.project,
            name=cfg.name,
            id=run_id,
            save_dir=cfg.cluster.wandb_dir,
            offline=(cfg.cluster.wandb_mode != "online"),
            resume="allow",
        )

    if cfg.log and main_node and not Path(cfg.exp_dir).exists():
        print("registering exp on main node")
        hparams = OmegaConf.to_container(cfg, resolve=True)
        print(hparams)
        logger.log_hyperparams(hparams)
        Path(cfg.exp_dir).mkdir(parents=True)
        with open(Path(cfg.exp_dir) / "config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg, resolve=True))

    if cfg.mode == "train":
        valset = instantiate(cfg.dataloader.dataset, domain="val")
        ssp585_testset = instantiate(cfg.dataloader.dataset, domain="test")
        trainset = instantiate(cfg.dataloader.dataset)  # will automatically pickup cfg split

        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=1,  # to make sample size
            num_workers=cfg.cluster.cpus,
            shuffle=False,
            collate_fn=collate_fn,
        )  # to viz shuffle samples

        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            num_workers=cfg.cluster.cpus,
            # num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=6,  # bumped from 3 -- Lustre memmap reads are the
            # bottleneck (workers sit in cl_sync_io_wait), not GPU compute;
            # deeper prefetch gives workers more runway to buffer ahead of a
            # slow/bursty Lustre read instead of the training step stalling
            # on a near-empty queue. Not persistent_workers -- keep workers
            # torn down and rebuilt each epoch.
        )

    pl_module = instantiate(cfg.module.module, cfg.module)
    if cfg.mode == "train":
        pl_module._ssp585_test_dataset = ssp585_testset

    # if hasattr(cfg, "load_ckpt"):
    #     # load weights w/o resuming run
    #     pl_module.init_from_ckpt(path='/scratch/gclyne/model_output/cmip_diffusion/flow_spectral_no_grad_2/',ckpt_fname=cfg.load_ckpt)  # noqa: E501

    # 4. Fit
    # pl_module.backbone = torch.compile(pl_module.backbone,mode="reduce-overhead")  # noqa: E501
    if cfg.compile:
        pl_module.backbone = torch.compile(pl_module.backbone, dynamic=True)

    if nonstrict_warmstart_path is not None:
        # Applied after torch.compile (when enabled) since a compiled
        # backbone's state_dict keys gain a "_orig_mod." prefix (OptimizedModule
        # wrapping) -- the source checkpoint was itself saved from a compiled
        # model, so loading before compile here would mismatch nearly every
        # backbone key instead of just the genuinely new ones.
        raw_ckpt = torch.load(nonstrict_warmstart_path, map_location="cpu")
        missing, unexpected = pl_module.load_state_dict(raw_ckpt["state_dict"], strict=False)
        print(f"  non-strict warmstart missing keys: {missing}")
        print(f"  non-strict warmstart unexpected keys: {unexpected}")
        warmstart_global_step = raw_ckpt.get("global_step", 0)
        print(f"  non-strict warmstart seeding global_step={warmstart_global_step}")

    # checkpointer = CheckpointEveryNSteps(
    #     dirpath=cfg.exp_dir, save_step_frequency=cfg.save_step_frequency
    # )
    # checkpointer = ModelCheckpoint(
    #     save_on_train_epoch_end=True,
    #     # save_top_k=3,
    #     filename="epoch-{epoch:02d}",
    #     save_last=False,
    #     save_top_k=-1,
    # )
    # checkpointer = CheckpointEveryNSteps(
    #     dirpath=cfg.exp_dir, save_step_frequency=cfg.save_step_frequency
    # )
    # checkpointer = ModelCheckpoint(
    #     save_on_train_epoch_end=True,
    #     # save_top_k=3,
    #     filename="epoch-{epoch:02d}",
    #     save_last=False,
    #     save_top_k=-1,
    # )
    checkpointer = ModelCheckpoint(
        dirpath=str(Path(cfg.exp_dir) / "checkpoints"),
        every_n_train_steps=2000,  # Save a checkpoint every 2000 steps
        save_on_train_epoch_end=False,  # Disable saving at the end of the epoch
        filename="step-{step:06d}",  # Recommendation: use step in the filename
        save_last=False,
        save_top_k=-1,
    )
    # Keep the 5 best checkpoints by long-rollout skill (lower = better).
    # Use a separate dirpath so best/ and step/ checkpoints don't collide.
    # This relies on the expensive autoregressive rollout run in
    # ForecastModule.validation_step, which the flow-matching diffusion model
    # does not (and should not) run, so it's skipped entirely for diffusion.
    best_checkpointer = None
    if not isinstance(pl_module, DCPPDiffusion):
        best_checkpointer = ModelCheckpoint(
            dirpath=str(Path(cfg.exp_dir) / "checkpoints" / "best"),
            monitor="val_rollout_skill",
            mode="min",
            save_top_k=5,
            filename="best-step={step:06d}-skill={val_rollout_skill:.4f}",
            save_last=False,
            save_on_train_epoch_end=False,
        )
    L.seed_everything(cfg.seed)
    # use_energy_score models (see forecast.py's n_energy_score_members
    # ensemble) hit "parameters that were not used in producing the loss"
    # from DDP's first-backward bucket rebuild on multi-GPU jobs -- some
    # registered parameter isn't reached by every use_energy_score
    # training_step's set of forward() calls (single-device training never
    # hits this, since DDP's bucket check doesn't run at all there).
    # find_unused_parameters=True is the standard fix (small per-step
    # overhead from the extra unused-parameter traversal), scoped to
    # energy-score configs only so it doesn't tax every other model.
    n_devices = cfg.cluster.get("gpus", 1) * cfg.cluster.launcher.get("nodes", 1)
    if cfg.module.module.get("use_energy_score", False) and n_devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"
    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=strategy,
        num_nodes=cfg.cluster.launcher.get("nodes", 1),
        precision=cfg.cluster.precision,
        log_every_n_steps=cfg.log_freq,
        profiler=getattr(cfg, "profiler", None),
        gradient_clip_val=1,
        max_steps=cfg.max_steps,
        max_epochs=cfg.get("max_epochs", None),
        # limit_train_batches=0.01,
        callbacks=[
            c
            for c in (
                TQDMProgressBar(refresh_rate=100 if cfg.mode == "train" else 1),
                checkpointer,
                best_checkpointer,
            )
            if c is not None
        ],
        logger=logger,
        plugins=[],
        limit_val_batches=cfg.limit_val_batches,  # max 5 samples
        limit_test_batches=cfg.limit_val_batches,  # max 5 samples
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
        num_sanity_val_steps=0,
    )

    if warmstart_global_step is not None:
        # Fast-forwards trainer.global_step (used by pf_warmup_steps,
        # noise_embedder_warmup_steps, checkpoint filenames, Trainer's
        # max_steps stopping condition, etc.) without going through
        # trainer.fit(ckpt_path=...)'s full state restore -- see the
        # non-strict warmstart branch above. The LR scheduler is NOT
        # fast-forwarded: it's freshly constructed in configure_optimizers
        # and starts its own warmup/decay from its own step 0, which is
        # intentional here (new optimizer, new param groups).
        #
        # trainer.global_step / LightningModule.global_step is NOT backed by
        # fit_loop.epoch_loop._batches_that_stepped (that's a separate,
        # unrelated counter -- confirmed by reading
        # lightning/pytorch/loops/training_epoch_loop.py: its global_step
        # property reads automatic_optimization.optim_progress.optimizer_steps
        # instead, itself a read-only property backed by
        # optim_progress.optimizer.step.total.completed). Setting
        # _batches_that_stepped alone silently does nothing to global_step --
        # confirmed the hard way: a run seeded this way still produced
        # checkpoints named step-002000/004000/006000.ckpt instead of
        # step-012000/014000/016000.ckpt, i.e. global_step actually started
        # at 0, not the intended warmstart_global_step.
        step_progress = (
            trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step
        )
        step_progress.total.ready = warmstart_global_step
        step_progress.total.completed = warmstart_global_step
        step_progress.current.ready = warmstart_global_step
        step_progress.current.completed = warmstart_global_step

    if cfg.debug:
        breakpoint()
    if cfg.mode == "train":
        trainer.fit(pl_module, train_loader, val_loader, ckpt_path=ckpt_path)
    elif cfg.mode == "validate":
        valset = instantiate(cfg.dataloader.dataset, domain="val")
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=1,
            num_workers=cfg.cluster.cpus,
            shuffle=False,
            collate_fn=collate_fn,
        )
        trainer.validate(pl_module, val_loader, ckpt_path=ckpt_path)
    elif cfg.mode == "test":
        trainer.test(pl_module, test_loader, ckpt_path=ckpt_path)  # noqa: F821


if __name__ == "__main__":
    main()


# python main_hydra.py log=False limit_examples=100 debug=True
