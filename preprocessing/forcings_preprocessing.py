"""Regrid raw CMIP6 forcing sources to the model's 144x144 grid and save as .pt.

This script's outputs already exist under reference_data/ for every experiment
currently used (GHG: processed_spatial_monthly_ghg_forcings/*.pt; ozone:
ozone_{exp}.pt) -- it does not need to be rerun for those. It's kept as a
record of how those tensors were built and as a starting point if a new
experiment/scenario needs the same treatment.

The ozone-regridding step below reads from a hardcoded path on Jean-Zay
(/lustre/fsn1/projects/rech/mlr/udy16au/ozone_forcings_{exp}/climoz_*), a
cluster this project no longer runs on -- the raw ozone concentration forcing
files (CMIP6 "climoz" ozone climatology inputs) would need to be re-sourced
from ESGF/input4MIPs and this path updated before rerunning it for a new
experiment.

See reference_data/README.md for what's already been built and where to get
other/additional forcing scenarios.
"""

import glob
import os
import subprocess
from typing import Any

import torch
import xarray as xr


def prep_aerosol_forcings(directory: str) -> Any:
    """Concatenate aerosol NetCDF files in `directory` into a single tensor."""
    files = sorted(glob.glob(directory))
    data = None
    for file in files:
        ds = xr.open_dataset(file)
        if data is None:
            data = torch.tensor(ds.to_array().values)
        else:
            data = torch.concatenate([data, torch.tensor(ds.to_array().values)], dim=1)
        ds.close()
    return data


def process_mole_file(file: str) -> Any:
    """Load a single spatial GHG mole-fraction NetCDF and slice to its valid time range."""
    if "ssp" in file:
        start_index = 0
        end_index = 85 * 12 + (12)  # 2100 is 85 years after 2015
    else:
        start_index = 1850 * 12 + (1 - 1)
        end_index = 2014 * 12 + (12)
    ds = xr.open_dataset(file, decode_times=False)
    ds = ds.isel(time=slice(start_index, end_index))
    ds = ds.drop_vars(["time_bnds"])
    ds = ds.to_array().values[0]
    ds = torch.swapaxes(torch.tensor(ds), -2, -1)
    return ds


def convert_spatial_ghg_to_tensor() -> None:
    """Convert regridded spatial GHG NetCDFs into per-gas, per-scenario .pt tensors."""
    scratch = os.environ["SCRATCH"]
    mole_files = sorted(
        glob.glob(f"{scratch}/reference_data/raw_spatial_monthly_ghg_forcings/*_preprocessed.nc")
    )
    for file in mole_files:
        gas = file.split("-")[3]
        ssp = next((s for s in file.split("/")[-1].split("-") if "ssp" in s), None)
        if ssp is None:
            ssp = "historical"
        ds = process_mole_file(file)
        file_name = gas + "_" + ssp + ".pt"
        torch.save(
            ds,
            f"{scratch}/reference_data/processed_spatial_monthly_ghg_forcings/{file_name}",
        )


def run_command(command: Any) -> None:
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)


def main() -> None:
    scratch = os.environ["SCRATCH"]

    # INTERPOLATE OZONE
    # cdo remapcon,r144x144 ozone_forcings_ssp370/tro3_2049.nc tro3_2049_regridded.nc
    experiments = [
        "ssp119",
        "ssp126",
        "ssp245",
        "ssp370",
        "ssp434",
        "ssp460",
        "ssp534-over",
        "ssp585",
        "historical",
    ]
    for exp in experiments:
        print("regridding ozone....")
        ozone_files = glob.glob(
            f"/lustre/fsn1/projects/rech/mlr/udy16au/ozone_forcings_{exp}/climoz*"
        )
        for file in ozone_files:
            regridded_dir = f"/lustre/fsn1/projects/rech/mlr/udy16au/ozone_forcings_{exp}_regridded"
            out_file = f"{regridded_dir}/{file.split('/')[-1]}"
            if os.path.exists(out_file):
                print(f"{out_file} exists")
                continue

            regrid_command = f"cdo remapcon,r144x144 {file} {out_file}"
            run_command(regrid_command)
        print("saving ozone....")
        ozone_files = glob.glob(
            f"/lustre/fsn1/projects/rech/mlr/udy16au/ozone_forcings_{exp}_regridded/climoz_*"
        )
        ozone_files.sort()
        data = None
        for ozone_file in ozone_files:
            ds = xr.open_dataset(ozone_file, decode_times=False)
            ds = ds["tro3"].isel(time=slice(1, -1))
            if "rlatu" in ds.coords:
                ds = ds.rename({"rlatu": "latitude", "rlonv": "longitude"})
            else:
                ds = ds.rename({"lat": "latitude", "lon": "longitude"})
            ds = ds.transpose("time", "plev", "longitude", "latitude")
            if data is None:
                data = torch.tensor(ds.values)
            else:
                data = torch.concatenate([data, torch.tensor(ds.values)], dim=0)
            ds.close()
        torch.save(data, f"{scratch}/reference_data/ozone_{exp}.pt")


if __name__ == "__main__":
    main()
