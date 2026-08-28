import os
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig
from tensordict.tensordict import TensorDict

from ArchesClimate.model.base_module import AvgModule, load_module

root = os.environ.get("ARCHESCLIMATE_ROOT", "/home/gclyne/ArchesClimate/ArchesClimate")


def nanvar(tensor: Any, dim: Any, keepdim: Any) -> Any:
    """Compute variance while handling NaN values.

    Args:
        tensor: Input tensor.
        dim: Dimension(s) to reduce over.
        keepdim: Whether to keep the reduced dimension.

    Returns:
        Tensor with NaN values handled.
    """
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).apply(lambda x: x.square())
    output = TensorDict(
        {
            "surface": output["surface"].nanmean(dim=0, keepdim=keepdim),
            "level": output["level"].nanmean(dim=0, keepdim=keepdim),
            "lev": output["lev"].nanmean(dim=0, keepdim=keepdim),
        }
    )
    return output


def nanstd(tensor: Any, dim: Any, keepdim: Any) -> Any:
    """Compute standard deviation while handling NaN values.

    Args:
        tensor: Input tensor.
        dim: Dimension(s) to reduce over.
        keepdim: Whether to keep the reduced dimension.

    Returns:
        Tensor with NaN values handled.
    """
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.apply(lambda x: x.sqrt())
    return output


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Generate delta statistics for CMIP data.

    Args:
        cfg: Hydra configuration object.
    """
    train = hydra.utils.instantiate(cfg.dataloader.dataset, domain="train")

    if isinstance(cfg.model_dir, (list, ListConfig)):
        pl_module = AvgModule(list(cfg.model_dir), ckpt_fname=cfg.inference.ckpt_fname)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pl_module = pl_module.to(device).eval()
        config = pl_module.cfg
    else:
        pl_module, config = load_module(
            cfg.model_dir,
            cfg=cfg,
            ckpt_fname=cfg.inference.ckpt_fname,
        )

    print(cfg)
    batch = train.__getitem__(0, skip_pf=True)
    del batch["next_state"]["non_spatial_forcings"]
    del batch["next_state"]["spatial_forcings"]
    keys = batch["next_state"].keys()

    # Bucket accumulation by lead time: a dataset with lead_times=[1, 12]
    # (say) draws a random lead time per sample, and the residual
    # (next_state - det prediction) has a genuinely different scale at
    # different lead times -- a single pooled mean/M2 blends those
    # distributions together, mis-scaling the normalized residual target
    # at every lead time it wasn't dominated by. Accumulating separately
    # per lead time and saving under "per_lead_time" lets DCPPDiffusion
    # (model/diffusion.py's _get_state_scaler) pick the right scaler per
    # sample instead. Bucket key is int(lead_time_months) (batch's raw
    # per-sample lead time, in months).
    per_lead: dict[int, dict[str, Any]] = {}

    with torch.no_grad():
        for i in range(2000):
            print(i)
            sample_idx = torch.randint(len(train), size=(1,)).item()
            batch = train.__getitem__(sample_idx, skip_pf=True)
            lt = int(batch["lead_time_months"])
            batch = {k: v.to("cuda") for k, v in batch.items()}
            batch = {k: v[None] for k, v in batch.items()}
            del batch["next_state"]["non_spatial_forcings"]
            del batch["next_state"]["spatial_forcings"]

            f_x = pl_module.forward(batch, use_condition=True)
            delta = {k: (batch["next_state"][k] - f_x[k]).squeeze().cpu() for k in keys}

            bucket = per_lead.setdefault(
                lt, {"n": 0, "mean": {k: 0 for k in keys}, "M2": {k: 0 for k in keys}}
            )
            bucket["n"] += 1
            n = bucket["n"]
            for k in keys:
                delta_diff = delta[k] - bucket["mean"][k]
                bucket["mean"][k] += delta_diff / n
                bucket["M2"][k] += delta_diff * (delta[k] - bucket["mean"][k])

    per_lead_time_std = {
        lt: {k: torch.sqrt(bucket["M2"][k] / (bucket["n"] - 1)) for k in keys}
        for lt, bucket in per_lead.items()
        if bucket["n"] > 1
    }
    print(
        "sample counts per lead time (months): "
        + ", ".join(f"{lt}: {per_lead[lt]['n']}" for lt in sorted(per_lead))
    )
    if len(per_lead_time_std) == 1:
        # Only one lead time was ever drawn (the common single-lead_time
        # case) -- save the old flat format so existing delta_std_name
        # files / DCPPDiffusion's single-scaler path are unaffected.
        std = next(iter(per_lead_time_std.values()))
        torch.save(std, f"{root}/stats/cmip_delta_std_{cfg.name}")
    else:
        torch.save(
            {"per_lead_time": per_lead_time_std},
            f"{root}/stats/cmip_delta_std_{cfg.name}",
        )


if __name__ == "__main__":
    main()
