import os
import sys

import hydra
import lightning as L
import torch
from hydra import compose, initialize_config_dir
from tensordict.tensordict import TensorDict

root = os.environ.get("ARCHESCLIMATE_ROOT", "/home/gclyne/ArchesClimate/ArchesClimate")
with initialize_config_dir(config_dir=f"{root}/configs"):
    default_overrides = [
        "module=forcing_dropout_co2log",
        "name=new_ozone_co2log",
        "dataloader.dataset.norm_scheme=zeroes",
        "dataloader.dataset.path=/scratch/gclyne/memmap_filled_in",
        "cluster=cleps",
        # 'dataloader.dataset.aerosol_log=True'
    ]
    # extra hydra overrides can be passed on the command line, e.g.:
    #   python generate_stats.py module=forcing_dropout_no_random_lt_co2log_canesm5 \
    #       name=new_ozone_co2log_canesm5 \
    #       dataloader.dataset.path=/scratch/gclyne/memmap_filled_in_canesm5
    # later overrides win, so CLI args on top of default_overrides replace matching keys.
    cfg = compose(config_name="config", overrides=default_overrides + sys.argv[1:])

import time  # noqa: E402

L.seed_everything(cfg.seed)
print("Instantiating dataset...", flush=True)
train = hydra.utils.instantiate(cfg.dataloader.dataset, domain="train")
print(f"Dataset ready ({len(train)} samples).", flush=True)
from collections import defaultdict  # noqa: E402
from typing import Any

num_samples = 1000
batch_size = 50

# Randomly sample indices for a batch
sample_indices = torch.randint(len(train), size=(num_samples,))

# Stats containers
counts = defaultdict(int)
means = {}
M2 = {}


def update_running_stats(name: Any, x: Any) -> None:
    """Update running mean and M2 for variance (per key), preserving NaNs."""
    x = x.float()
    if name not in means:
        means[name] = torch.zeros_like(x, dtype=torch.float32)
        M2[name] = torch.zeros_like(x, dtype=torch.float32)
        counts[name] = torch.zeros_like(x, dtype=torch.int32)

    mask = ~torch.isnan(x)  # valid entries
    n = counts[name]
    # only update non-NaN positions
    delta = torch.zeros_like(x, dtype=torch.float32)
    delta2 = torch.zeros_like(x, dtype=torch.float32)

    # Welford’s update for masked values
    n_new = n + mask.int()
    delta[mask] = x[mask] - means[name][mask]
    means[name][mask] += delta[mask] / n_new[mask].clamp_min(1)
    delta2[mask] = x[mask] - means[name][mask]
    M2[name][mask] += delta[mask] * delta2[mask]
    counts[name] = n_new


n_batches = num_samples // batch_size
t0 = time.time()
for i in range(0, num_samples, batch_size):
    batch_num = i // batch_size + 1
    elapsed = time.time() - t0
    eta = (elapsed / batch_num * n_batches - elapsed) if batch_num > 1 else float("nan")
    print(
        f"Batch {batch_num}/{n_batches} | elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
        flush=True,
    )

    # Get batch indices
    batch_indices = sample_indices[i : i + batch_size]

    batch_data = [
        train.__getitem__(idx.item(), normalize=False, state_only=True) for idx in batch_indices
    ]
    for b in batch_data:
        s = b["state"].squeeze()

        # update running stats for each sub-key
        for key, tensor in s.items():
            if tensor is not None:
                update_running_stats(key, tensor)
        s = None
    batch_data = None

output = {}
for key in means:
    mean = means[key].clone()
    var = torch.full_like(mean, float("nan"))
    std = torch.full_like(mean, float("nan"))

    mask = counts[key] == num_samples
    mean[~mask] = torch.nan
    var[mask] = M2[key][mask] / (counts[key][mask] - 1)
    std[mask] = torch.sqrt(var[mask])

    output[f"{key}_mean"] = mean
    output[f"{key}_std"] = std

output = TensorDict(output)

torch.save(output, f"{root}/stats/cmip_stats_{cfg.name}.pt")
