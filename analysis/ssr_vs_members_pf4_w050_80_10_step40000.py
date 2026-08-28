"""SSR (spread-skill ratio) and RMSE vs. ensemble size.

For deterministic_damip_pf4_energy_score_w050_80_10 step-040000, ssp434 (val),
20-member noise-seeded ensemble.

Reports SSR = sqrt(finite-sample-corrected ensemble variance) / RMSE(ensemble
mean, target) and RMSE(ensemble mean, target) for M in {1..20} members,
surface tas, full available rollout length. Simple cumulative approach (not
exhaustive/combinatoric): just adds members 0..M-1 one at a time. Ideal
calibrated SSR is 1.0. Mirrors analysis/ssr_vs_members_pf8_emafix_step22000.py.
"""

import torch
from tensordict.tensordict import TensorDict

from analysis.analysis_utils import initialize_notebook, load_cmip_experiment

work, scratch, cfg, train_dataset = initialize_notebook(
    domain="val", config_module="deterministic_damip_pf4_energy_score_w050_80_10"
)

experiment_name = (
    "deterministic_damip_pf4_energy_score_w050_80_10_0_12_10_1_1020_1_val_0_"
    "step-step=040000.ckpt_ema"
)
num_members = 20
domain = "surface"
num_decades = 8

data = load_cmip_experiment(
    experiment_name,
    num_members,
    scratch,
    train_dataset,
    domain,
    num_decades,
    is_yearly=False,
    denormalize=True,
)  # (members, 1, time, var, plev, lat, lon)
data = data[:, 0]  # (members, time, var, plev, lat, lon)

target = TensorDict.load_memmap(f"{scratch}/memmap_filled_in/r1i1p1f1_ssp434_interpolation.memmap")[
    domain
]  # (var, time, lat, lon)

var_index, plev_index = 0, 0  # tas
T = min(data.shape[1], target.shape[1] - 2)
fc = data[:, :T, var_index, plev_index]  # (members, T, lat, lon)
tgt = target[var_index, 2 : 2 + T]  # (T, lat, lon), +2 offset matches notebook convention

print(f"members={num_members} T={T} field shape={fc.shape[-2:]}")


def ssr_and_rmse(members_subset):
    M = members_subset.shape[0]
    ens_mean = members_subset.mean(dim=0)  # (T, lat, lon)
    if M > 1:
        ens_var = members_subset.var(dim=0, unbiased=True)
        spread = torch.sqrt(ens_var.mean() * (M + 1) / M)
    else:
        spread = torch.tensor(float("nan"))
    err = ens_mean - tgt
    rmse = torch.sqrt((err**2).mean())
    ssr = (spread / rmse).item()
    return ssr, spread.item(), rmse.item()


print(f"{'M':>3} {'SSR':>10} {'RMSE':>10} {'spread':>10}")
for M in range(1, num_members + 1):
    subset = fc[:M]
    ssr, spread, rmse = ssr_and_rmse(subset)
    print(f"{M:>3} {ssr:>10.3f} {rmse:>10.4f} {spread:>10.4f}")
