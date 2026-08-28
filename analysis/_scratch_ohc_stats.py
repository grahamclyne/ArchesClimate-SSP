"""Print summary stats for the OHC panel of abrupt4xco2_det_vs_es.py, to
quote exact numbers in the paper's analysis paragraph."""

import torch
from tensordict.tensordict import TensorDict

from analysis.abrupt4xco2_det_vs_es import (
    MAX_YEARS,
    RUNS,
    TARGET_REALIZATION,
    area_weighted_mean,
    to_annual_series,
)
from analysis.analysis_utils import initialize_notebook
from analysis.compare_runs import load_run_lev
from analysis.energy_hydrostatic_balance import ocean_heat_content


def main():
    scratch = "/scratch/gclyne"
    _, _, hydra_cfg, train_dataset = initialize_notebook(
        domain="train",
        config_module="deterministic_damip_pf4_energy_score_w050",
        dataloader="cmip_random_lead_times",
    )
    depth_levels = [float(d) for d in hydra_cfg.module.depth_levels]
    data_mean_lev = train_dataset.data_mean["lev"].squeeze(0)
    data_std_lev = train_dataset.data_std["lev"].view(-1)
    lats = torch.linspace(-90, 90, hydra_cfg.module.get("lat_dim", 144))
    dataset_path = hydra_cfg.module.dataset_path

    target_path = f"{scratch}/{dataset_path}/{TARGET_REALIZATION}_abrupt-4xCO2_interpolation.memmap"
    target_td = TensorDict.load_memmap(target_path)
    target_lev = target_td["lev"][0]
    target_lev = torch.where(target_lev.abs() < 1e30, target_lev, torch.nan)
    target_ohc = ocean_heat_content(target_lev, depth_levels)[None]
    target_annual = to_annual_series(area_weighted_mean(target_ohc, lats))[0][:MAX_YEARS]
    print("TARGET OHC yr1-5 mean:", target_annual[:5].mean().item())
    print("TARGET OHC yr76-80 mean:", target_annual[75:80].mean().item())
    print("TARGET OHC delta:", (target_annual[75:80].mean() - target_annual[:5].mean()).item())

    for label, (run_name, _) in RUNS.items():
        run_dir = f"{scratch}/generated_data/{run_name}"
        lev = load_run_lev(run_dir, data_mean_lev, data_std_lev)
        ohc = torch.stack([ocean_heat_content(m, depth_levels) for m in lev], dim=0)
        gm_annual = to_annual_series(area_weighted_mean(ohc, lats))
        mean = gm_annual.mean(dim=0)
        print(f"{label} OHC yr1-5 mean:", mean[:5].mean().item())
        print(f"{label} OHC yr76-80 mean:", mean[75:80].mean().item())
        print(f"{label} OHC delta:", (mean[75:80].mean() - mean[:5].mean()).item())


if __name__ == "__main__":
    main()
