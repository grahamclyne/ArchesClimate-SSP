"""Global RMSE and SSR (spread-skill ratio) as a function of
energy_score_noise_scale, for deterministic_damip_pf4_energy_score_w050_80_10
step-040000, ssp434 (val), 5-member ensembles, surface tas, full available
rollout length. Feeds the noise-scale appendix table.
"""

import torch
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook, load_cmip_experiment

work, scratch, cfg, train_dataset = initialize_notebook(
    domain="val", config_module="deterministic_damip_pf4_energy_score_w050_80_10"
)

NOISE_SCALES_MEMBERS = [("1.0", 5), ("0.8", 5), ("0.6", 5), ("0.5", 5), ("0.4", 5)]
TARGET_PATH = f"{scratch}/memmap_filled_in/r1i1p1f1_ssp434_interpolation.memmap"

domain = "surface"
num_decades = 8
var_index, plev_index = 0, 0  # tas


def load_experiment_alt_naming(experiment_name, num_members, scratch, train, domain, limit):
    """Same as load_cmip_experiment's non-flow branch, but for the
    nscale0.5 rollout whose files are named
    rollout_{domain}_{i+1}_0{ens_member}.pt (no underscore before the
    member digit) instead of the usual rollout_{domain}_{i+1}_0_{ens_member}.pt.
    """
    indices = [119, 239, 359, 479, 599, 719, 839, 959, 1018]
    path = f"{scratch}/generated_data/{experiment_name}"
    ensemble_members = []
    for ens_member in range(num_members):
        stacked = []
        for i in indices[:limit]:
            data = torch.load(
                f"{path}/rollout_{domain}_{i + 1}_0{ens_member}.pt",
                map_location=torch.device("cpu"),
            )
            data = (data * train.data_std[domain]) + train.data_mean[domain]
            stacked.append(data)
        ensemble_members.append(torch.concat(stacked, dim=1))
    return torch.stack(ensemble_members)


def ssr_and_rmse(members_subset, tgt):
    M = members_subset.shape[0]
    ens_mean = members_subset.mean(dim=0)
    ens_var = members_subset.var(dim=0, unbiased=True)
    spread = torch.sqrt(ens_var.mean() * (M + 1) / M)
    err = ens_mean - tgt
    rmse = torch.sqrt((err**2).mean())
    ssr = (spread / rmse).item()
    return ssr, spread.item(), rmse.item()


target_full = TensorDict.load_memmap(TARGET_PATH)[domain]
print("\n=== val (ssp434) ===")
for nscale, num_members in NOISE_SCALES_MEMBERS:
    suffix = "" if nscale == "1.0" else f"_nscale{nscale}"
    experiment_name = (
        f"deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_1020_1_val_0_"
        f"step-step=040000.ckpt_ema{suffix}"
    )
    if nscale == "0.5":
        data = load_experiment_alt_naming(
            experiment_name,
            num_members,
            scratch,
            train_dataset,
            domain,
            num_decades,
        )
    else:
        data = load_cmip_experiment(
            experiment_name,
            num_members,
            scratch,
            train_dataset,
            domain,
            num_decades,
            is_yearly=False,
            denormalize=True,
        )
    data = data[:, 0]
    T = min(data.shape[1], target_full.shape[1] - 2)
    fc = data[:num_members, :T, var_index, plev_index]
    tgt = target_full[var_index, 2 : 2 + T]
    ssr, spread, rmse = ssr_and_rmse(fc, tgt)
    print(
        f"  nscale={nscale:<5} members={fc.shape[0]} rmse={rmse:.3f}K spread={spread:.3f}K ssr={ssr:.3f}"
    )
