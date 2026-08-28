from pathlib import Path

import lightning as L  # noqa N812
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ArchesClimate.model.optimizers import OptimizerMixin


def load_module(
    path: str,
    device: str = "auto",
    dotlist: list = [],
    return_config: bool = True,
    ckpt_fname: str | None = None,
    cfg=None,
    use_ema: bool = True,
    **kwargs,
):
    """Load a trained module from a checkpoint directory.

    Args:
    path: Path under `modelstore` directory, holding hydra config `config.yaml`
        and lightning module checkpoint(s) under `checkpoints/*.chkpt`.
    device: Device to move the loaded module to; "auto" picks cuda if available,
        else cpu.
    dotlist: list of config overrides.
    return_config: Whether to return cfg along with module, or just the instantiated module.
    ckpt_fname: Optional. Checkpoint filename under `checkpoints/`, otherwise
        chooses most recent file.
    cfg: Optional pre-loaded Hydra config; loaded from `path`/config.yaml when not given.
    use_ema: Whether to load EMA weights when the checkpoint has them,
        otherwise loads the regular state_dict.
    **kwargs: Extra keyword arguments forwarded to the module's instantiate() call.
    """
    if Path("modelstore").joinpath(path).exists():
        path = Path("modelstore").joinpath(path)
    else:
        path = Path(path)
    if cfg is None:
        cfg = OmegaConf.load(path / "config.yaml")
        cfg.merge_with_dotlist(dotlist)
    module = instantiate(cfg.module.module, cfg.module, **kwargs)
    module.init_from_ckpt(path, ckpt_fname=ckpt_fname, missing_warning=False, use_ema=use_ema)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    module = module.to(device).eval()
    if not return_config:
        return module
    return module, cfg


class BaseLightningModule(OptimizerMixin, L.LightningModule):
    def __init__(self):
        super().__init__()

    def mylog(self, dct={}, mode="auto", **kwargs):
        if mode == "auto":
            mode = "train_" if self.training else "val_"
        dct.update(kwargs)
        for k, v in dct.items():
            self.log(mode + k, v, prog_bar=True, sync_dist=True, add_dataloader_idx=True)

    def init_from_ckpt(
        self,
        path: str,
        ckpt_fname: str | None = None,
        ignore_keys: list = list(),
        missing_warning: bool = True,
        use_ema: bool = True,
    ):
        if Path(path).is_dir():
            ckpt_dir = Path(path) / "checkpoints"
            if not ckpt_dir.exists():
                ckpt_dir = Path(path)  # path itself is the checkpoints dir (e.g. .../best)
            path = ckpt_dir
            paths = list(Path(path).glob("**/*.ckpt"))
            if ckpt_fname is not None:
                ckpt_fname = str(ckpt_fname)
                filtered = [p for p in paths if ckpt_fname in str(p.relative_to(path))]
                if not filtered:
                    # Try to convert between step and epoch naming (1 epoch ≈ 2000 steps)
                    import re

                    step_match = re.search(r"step[=-](\d+)", ckpt_fname)
                    epoch_match = re.search(r"epoch[=-](\d+)", ckpt_fname)
                    if step_match:
                        target_epoch = int(step_match.group(1)) // 2000
                        filtered = [
                            p
                            for p in paths
                            if re.search(rf"epoch[=-]0*{target_epoch}(?!\d)", p.name)
                        ]
                        if filtered:
                            print(
                                f"No step checkpoint matching '{ckpt_fname}', "
                                f"using epoch {target_epoch} equivalent: {filtered[0].name}"
                            )
                    elif epoch_match:
                        target_step = int(epoch_match.group(1)) * 2000
                        filtered = [
                            p for p in paths if re.search(rf"step[=-]0*{target_step}(?!\d)", p.name)
                        ]
                        if filtered:
                            print(
                                f"No epoch checkpoint matching '{ckpt_fname}', "
                                f"using step {target_step} equivalent: {filtered[0].name}"
                            )
                if not filtered:
                    raise FileNotFoundError(
                        f"No checkpoint matching '{ckpt_fname}' found in {path}. "
                        f"Available: {[p.name for p in paths]}"
                    )
                paths = filtered
            path = sorted(paths, key=lambda x: x.stat().st_mtime)[-1]

        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        sd = ckpt.get("ema_state_dict", ckpt["state_dict"]) if use_ema else ckpt["state_dict"]

        # --- FIX FOR TORCH.COMPILE ---
        # Create a new state dict with cleaned keys
        new_sd = {}
        for k, v in sd.items():
            # Strip the prefix added by torch.compile
            name = k.replace("_orig_mod.", "")
            new_sd[name] = v
        sd = new_sd
        # -----------------------------

        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]

        # Now load the cleaned state dict
        self.load_state_dict(sd, strict=False)

        # Update missing keys check to use the cleaned 'sd'
        current_model_keys = self.state_dict().keys()
        missing_keys = set(
            [".".join(k.split(".")[:2]) for k in current_model_keys if k not in sd.keys()]
        )

        if len(missing_keys) and missing_warning:
            print("Missing keys:", missing_keys)
        print(f"Restored from {path}")


class AvgModule(L.LightningModule):
    """Wrapper around several lightning modules to run forward and compute average prediction."""

    def __init__(self, module_paths, ckpt_fname: str | None = None):
        super().__init__()
        path = module_paths[0]
        if Path("modelstore").joinpath(path).exists():
            path = Path("modelstore").joinpath(path)
        else:
            path = Path(path)
        self.cfg = OmegaConf.load(path / "config.yaml")
        self.core = nn.ModuleList(
            [load_module(p, return_config=False, ckpt_fname=ckpt_fname) for p in module_paths]
        )

    def forward(self, *args, **kwargs):
        return torch.stack([m.forward(*args, **kwargs) for m in self.core], dim=0).mean(dim=0)
