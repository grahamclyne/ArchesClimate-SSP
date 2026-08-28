import os
from typing import Any

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from tensordict.tensordict import TensorDict


def load_config(
    work: Any,
    config_module: Any,
    dataloader: str = "cmip_random_lead_times",
    cluster: str = "jean_zay_a100",
) -> Any:
    """Initialize Hydra config and compose the config object.

    dataloader defaults to the IPSL group for backward compatibility --
    pass dataloader="cmip_random_lead_times_canesm5_native" for CanESM5
    native-grid config_modules (e.g. flow_canesm5_damip_fixed), which need
    that group's own forcings_path and a different path template. cluster
    defaults to "jean_zay_a100" (data_path=/lustre/fsn1/.../udy16au/, where
    every dataset -- IPSL and CanESM5-native alike -- actually lives now);
    "jean_zay_v100" resolves to the same data_path and works equally well.
    "cleps"/"cleps_local" are a stale, pre-migration cluster's config
    (/scratch/gclyne/ paths, "arches" SLURM account) and no longer resolve
    to real data or a valid account on this cluster.
    """
    with initialize_config_dir(config_dir=f"{work}/ArchesClimate/ArchesClimate/configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"module={config_module}",
                f"dataloader={dataloader}",
                f"cluster={cluster}",
            ],
        )
    return cfg


def instantiate_dataset(cfg: Any, domain: Any) -> Any:
    """Instantiate the dataset from the Hydra config.

    path/forcings_path come from cfg as composed (via the dataloader group
    and ${cluster.data_path}${module.dataset_path}), not hardcoded here --
    hardcoding them to the IPSL memmap_filled_in/reference_data paths used
    to silently break every non-IPSL config_module (e.g. CanESM5 native
    grid, whose data lives under memmap_filled_in_canesm5_native /
    reference_data_canesm5_native instead).
    """
    dataset = hydra.utils.instantiate(cfg.dataloader.dataset, domain=domain)
    return dataset


def set_plot_style() -> None:
    """Set a custom matplotlib plotting style."""
    plt.style.use("seaborn-v0_8-paper")

    small_size = 12
    medium_size = 13
    bigger_size = 14

    mpl.rcParams["font.serif"] = ["Times New Roman"]

    plt.rc("font", size=small_size)
    plt.rc("axes", titlesize=medium_size)
    plt.rc("axes", labelsize=small_size)
    plt.rc("xtick", labelsize=small_size)
    plt.rc("ytick", labelsize=small_size)
    plt.rc("legend", fontsize=small_size)
    plt.rc("figure", titlesize=bigger_size)
    plt.rc("lines", linewidth=1.1)

    # Optional: override font family globally
    plt.rcParams["font.family"] = ["sans"]


# --- Convenience function to call at the start of a notebook ---
def initialize_notebook(
    domain="train",
    config_module="new_ozone",
    environment="cluster",
    dataloader="cmip_random_lead_times",
    cluster="cleps",
):
    """Initialize config, dataset, and plot style for a notebook session.

    Args:
        domain: Dataset domain to load (e.g. "train", "val", "test").
        config_module: Hydra config module name to compose.
        environment: Execution environment; "local" or "cluster".
        dataloader: Dataloader config group to compose (default: the IPSL
            group). Pass "cmip_random_lead_times_canesm5_native" for CanESM5
            native-grid config_modules -- see load_config.
        cluster: Hydra cluster config group to compose -- see load_config.

    Returns:
        Tuple of (work path, scratch path, cfg, dataset).
    """
    if environment == "local":
        work = "/Users/gclyne/"
        scratch = "/Users/gclyne/"
    else:
        work = os.environ.get("WORK", "/home/gclyne/")
        scratch = os.environ.get("SCRATCH", "/scratch/gclyne")
    cfg = load_config(work, config_module, dataloader=dataloader, cluster=cluster)
    dataset = instantiate_dataset(cfg, domain)
    set_plot_style()
    return work, scratch, cfg, dataset


def load_cmip_experiment(
    experiment_name,
    num_members,
    scratch,
    train,
    domain,
    limit,
    is_yearly=False,
    remove_clima=False,
    denormalize=True,
    bottom_limit=0,
    seeds=2,
):
    """Load and stack CMIP experiment rollout data for ensemble members.

    Args:
        experiment_name: Name of the experiment directory under
            generated_data/.
        num_members: Number of ensemble members to load.
        scratch: Path to the scratch storage directory.
        train: Dataset object providing normalization statistics.
        domain: Dataset domain key (e.g. "train", "val").
        limit: Upper index into the rollout segment list.
        is_yearly: If True, average monthly data to yearly means.
        remove_clima: If True, subtract a historical climatology.
        denormalize: If True, apply inverse normalization using train
            statistics.
        bottom_limit: Lower index into the rollout segment list.
        seeds: Number of diffusion seeds per ensemble member (flow
            experiments only).

    Returns:
        Stacked tensor of shape (num_members * seeds, ...) for flow
        experiments or (num_members, ...) otherwise.
    """
    ensemble_members = []
    indices = [
        120,
        240,
        360,
        480,
        600,
        720,
        840,
        960,
        1080,
        1200,
        1320,
        1440,
        1560,
        1680,
        1800,
        1920,

    ]  # note the last one is 1029 as it is only 5 years
    if remove_clima:
        historical = TensorDict.load_memmap(
            "/scratch/gclyne/memmap_filled_in/r1i1p1f1_historical_interpolation.memmap"
        )[domain]
        climatology = (
            historical[:, -251:-11]
            .view(20, 12, historical.shape[0], historical.shape[2], 144, 144)
            .mean(0)
        )  # shape: 12,144,144, take the last two decades only for clima

    for ens_member in range(num_members):
        path = f"{scratch}/generated_data"
        stacked = []

        if "flow" in experiment_name:
            for seed_member in range(0, seeds):
                stacked = []

                seed = (seed_member) + ens_member

                for i in indices[bottom_limit:limit]:
                    # if (i == 719) and "val3" in experiment_name:
                    #     i = 719
                    data = torch.load(
                        (
                            f"{path}/{experiment_name}"
                            f"/rollout_{domain}_{i - 1}"
                            f"_{ens_member}_{seed}.pt"
                        ),
                        map_location=torch.device("cpu"),
                    )  # torch.Size([1, 120, 8, 1, 144, 144])
                    if i == 1018 and is_yearly:
                        data = data[:48]
                    if i == 715 and is_yearly:
                        data = data[:108]

                    data = torch.where(data > 1e20, torch.nan, data)
                    # how to not repeat this twice
                    if denormalize:
                        data = (data * train.data_std[domain]) + train.data_mean[domain]
                    if remove_clima:
                        climatology_tiled = climatology.tile(data.shape[1] // 12, 1, 1, 1, 1)

                        data = data - climatology_tiled

                    if is_yearly:
                        # print(data.shape)
                        data = data.view(-1, 12, data.shape[-4], data.shape[-3], 144, 144).mean(
                            dim=1
                        )
                    stacked.append(data)
                if is_yearly:
                    ensemble_members.append(torch.concat(stacked, dim=0))
                else:
                    ensemble_members.append(
                        torch.concat(stacked, dim=0)
                    )  # append the whole 1032 steps
        else:
            for i in indices[:limit]:
                data = torch.load(
                    (f"{path}/{experiment_name}/rollout_{domain}_{i}_0_{ens_member}.pt"),
                    map_location=torch.device("cpu"),
                )

                if denormalize:
                    data = (data * train.data_std[domain]) + train.data_mean[domain]
                if is_yearly:
                    data = data.view(-1, 12, data.shape[-4], data.shape[-3], 144, 144).mean(dim=1)

                stacked.append(data)
            if is_yearly:
                ensemble_members.append(torch.concat(stacked, dim=0))  # append the whole 1032 steps
            else:
                ensemble_members.append(torch.concat(stacked, dim=1))  # append the whole 1032 steps

        # if is_yearly and not ('flow' in experiment_name):
        #     ensemble_members.append(torch.concat(stacked, dim=0))
        # else:
        #     ensemble_members.append(torch.concat(stacked, dim=1))
        # if remove_clima:
        #     clima = torch.load(
        #         "/lustre/fswork/projects/rech/mlr/udy16au/ArchesClimate/ArchesClimate/utils/climatology_stats_cmip_piControl_included_1900_2000_oro_new_masking.pt"  # noqa: E501
        #     )
        #     clima_extended = TensorDict(
        #         {k: v.repeat(13, 1, 1, 1, 1) for k, v in clima.items()}
        #     )

        #     data = (
        #         data - clima_extended[f"{domain}_mean"][None, 2 : data.shape[1] + 2]  # noqa: E501
        #     )

        # stacked.append(data)

    data = torch.stack(ensemble_members)
    return data


def load_target_cmip_experiment(
    target_ssp_number,
    num_members,
    scratch,
    train,
    domain,
    is_yearly=False,
    remove_clima=False,
    denormalize=True,
    full_ocean=False,
):
    """Load and stack target CMIP SSP or abrupt experiment data.

    Args:
        target_ssp_number: SSP scenario number (e.g. 245, 585) or
            "abrupt" for abrupt-4xCO2 experiments.
        num_members: Number of ensemble members to load.
        scratch: Unused; kept for API consistency.
        train: Dataset object providing normalization statistics.
        domain: Dataset domain key (e.g. "train", "val").
        is_yearly: If True, average monthly data to yearly means.
        remove_clima: If True, subtract a pre-computed climatology.
        denormalize: If True, apply inverse normalization using train
            statistics.
        full_ocean: If True, load the full-ocean variant of the data.

    Returns:
        Stacked tensor of shape (num_members, ...) containing
        denormalized target data.
    """
    target_ssp = []
    for i in range(num_members):
        if full_ocean:
            data = torch.load(
                f"/lustre/fswork/projects/rech/mlr/udy16au/ArchesClimate/ArchesClimate/generated_data/cmip_ssp{target_ssp_number}_{i}_full_ocean"  # noqa: E501
            )[domain]
        else:
            data = torch.load(
                f"/lustre/fswork/projects/rech/mlr/udy16au/ArchesClimate/ArchesClimate/generated_data/cmip_ssp{target_ssp_number}_{i}"  # noqa: E501
            )[domain]
        if denormalize:
            data = (data * train.data_std[domain]) + train.data_mean[domain]
        data = data[: ((data.shape[0] // 12) * 12)]
        if remove_clima:
            clima = torch.load(
                "/lustre/fswork/projects/rech/mlr/udy16au/ArchesClimate/ArchesClimate/utils/climatology_stats_cmip_piControl_included_1900_2000_oro_new_masking.pt"  # noqa: E501
            )
            clima_extended = TensorDict({k: v.repeat(100, 1, 1, 1, 1) for k, v in clima.items()})
            data = data - clima_extended[f"{domain}_mean"][None, 2 : data.shape[0] + 2]
        if is_yearly:
            data = data.view(-1, 12, data.shape[-4], data.shape[-3], 144, 144).mean(dim=1)

        target_ssp.append(data)
    target_ssp = torch.stack(target_ssp)
    return target_ssp
