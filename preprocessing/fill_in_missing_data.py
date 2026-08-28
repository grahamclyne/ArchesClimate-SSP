import glob
import os

import numpy as np
import xarray as xr

scratch = os.environ["SCRATCH"]
for file in glob.glob(f"{scratch}/interpolation_project_datasets/*"):
    scratch = os.environ["SCRATCH"]
    ds = xr.open_dataset(file)
    for var in ["hus", "ta", "ua", "va", "zg"]:
        ds_var = ds[var].where(ds[var] < 1e30)
        mask = ds_var.mean(dim="time").isnull()
        masked = ds_var.where(~mask)
        stacked = masked.stack(z=("lat", "lon"))
        stacked = stacked.assign_coords(z_num=("z", np.arange(stacked.sizes["z"])))
        interp = stacked.swap_dims({"z": "z_num"}).interpolate_na(dim="z_num", method="linear")
        interp = interp.swap_dims({"z_num": "z"})
        data_interp = interp.unstack("z")
        ds[var] = data_interp.where(~mask, ds_var)

    ds.to_netcdf(f"{scratch}/interpolation_project_datasets_filled_in/{file.split('/')[-1]}")
    ds.close()
    print(f"Finished {file}")
