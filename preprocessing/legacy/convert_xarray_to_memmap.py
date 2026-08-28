import glob
import os

import numpy as np
import torch
import xarray as xr
from tensordict.tensordict import TensorDict

scratch = os.environ["SCRATCH"]
files = glob.glob(f"{scratch}/full_ocean_dataset_forced/*")

variables = {
    "surface": [
        "tos",
        "siconc",
        "sithick",
        "zos",
        "net_flux",
        "evspsbl",
        "huss",
    ],
    "level": [],
    "lev": ["thetao", "so", "vo", "uo"],
    "spatial_forcings": ["methane", "carbon", "nitrous"],
    "non_spatial_forcings": [
        "ssi_0",
        "ssi_1",
        "ssi_2",
        "ssi_3",
        "ssi_4",
        "ssi_5",
    ],
}

out_dir = "full_ocean_dataset_memmap"
for file in files:
    print("processing file:", file)
    if os.path.exists(f"{scratch}/{out_dir}/{file.split('/')[-1].replace('.nc', '.memmap')}"):
        fname = file.split("/")[-1].replace(".nc", ".memmap")
        print(f"{fname} memmap already exists, skipping")
        continue

    ds = xr.open_dataset(file, engine="netcdf4")
    # ds = ds.sel(plev=[100000.,  92500.,  85000.,  70000.,  60000.,  50000.,  40000.,  30000., 25000.,  20000.,  15000.,  10000.,   7000.,   5000.,   3000.,   2000.,1000.])  # noqa: E501
    # ds = ds.sel(lev=[0.50576,3.8562799,8.092519,13.991038,22.757616,35.740204,53.850636,77.61116,108.03028, 147.40625,])  # noqa: E501
    np_arrays = {}
    for key, variable in variables.items():
        if variable:  # non-empty
            np_arrays[key] = ds[list(variable)].to_array().to_numpy()
        else:  # empty list -> create an empty array
            np_arrays[key] = np.empty((0,))

    tdict = TensorDict(
        {key: torch.from_numpy(np_array).float() for key, np_array in np_arrays.items()}
    )
    tdict["time"] = ds.time.values
    tdict.memmap(f"{scratch}/{out_dir}/{file.split('/')[-1].replace('.nc', '.memmap')}")
