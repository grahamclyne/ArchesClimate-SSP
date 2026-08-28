"""Merge GHG/aerosol/ozone/solar forcings into the interpolated IPSL dataset.

NOT CURRENTLY RUNNABLE: reads from f"{scratch}/full_ocean_dataset/*{exp}*",
which no longer exists on disk (nor does its default output,
full_ocean_dataset_forced/) -- this reflects an earlier stage of the pipeline
before those intermediate NetCDF directories were cleaned up after
memmap_filled_in was produced. It also predates common/forcings.py's
consolidated, verified-correct forcing logic (see add_forcings_canesm5.py):
it uses only the old 6-band ozone selection (see fix_ozone.py, which patches
memmap_filled_in's ozone channels after the fact) and includes a 4th spatial
gas, cfc11, that isn't in any current module config's spatial_forcing_variables.
Kept as a historical reference of how memmap_filled_in was originally built;
rewriting it to use common/forcings.py would need `full_ocean_dataset` to
exist again to verify against, which would mean re-deriving it from
interpolation_project_datasets_filled_in (itself no longer on disk either).

experiments is left hardcoded to a single scenario below, matching how this
was actually last run per-experiment rather than in one pass over the full
list -- same pattern as fix_ozone.py before its cleanup.
"""

import glob
import os

import torch
import xarray as xr

scratch = os.environ["SCRATCH"]

experiments = [
    "piControl",
    "historical",
    "ssp119",
    "ssp126",
    "ssp245",
    "ssp370",
    "ssp434",
    "ssp460",
    "ssp534-over",
    "ssp585",
    "abrupt-4xCO2",
]
experiments = ["historical"]
spatial_gases = ["methane", "cfc11", "carbon", "nitrous"]
aerosol_vars = [
    "load_ASNO3M",
    "load_CSNO3M",
    "load_CINO3M",
    "load_SO4",
    "load_AIBCM",
    "load_ASBCM",
]

# non_spatial_gases = ['CO2','CFC12eq','CH4','N2O']
solar_forcings = torch.load(
    f"{scratch}/reference_data/solar_forcings_monthly_interpolated.pt"
)  # need to generate this first

full_ozone = False
rotate_ssps = False
yearly_temperature = False
for index, exp in enumerate(experiments):
    print("is_rotated: ", rotate_ssps, "current index: ", index, exp)
    exp_files = glob.glob(f"{scratch}/full_ocean_dataset/*{exp}*")
    for file in exp_files:
        out_file = file.split("/")[-1]
        print(out_file)
        if rotate_ssps:
            # assign forcings for next ssp to current ssp
            if "ssp" in exp:
                if index == (len(experiments) - 1):
                    exp = "ssp119"
                else:
                    exp = experiments[index + 1]
        ds = xr.open_dataset(file)
        if yearly_temperature:
            ds["tas_yearly"] = ds["tas"].rolling(time=12, center=True).mean()
            ds["tas_5year"] = ds["tas"].rolling(time=60, center=True).mean()

            ds["ta_yearly"] = ds["ta"].rolling(time=12, center=True).mean()
            ds["ta_5year"] = ds["ta"].rolling(time=60, center=True).mean()

        # first, spatial ghg
        for gas in spatial_gases:
            if exp == "piControl":
                data = torch.load(
                    f"{scratch}/reference_data/processed_spatial_monthly_ghg_forcings/{gas}_historical.pt"
                )
                data = torch.tile(data[:12], (250, 1, 1))
            elif exp == "abrupt-4xCO2":
                data = torch.load(
                    f"{scratch}/reference_data/processed_spatial_monthly_ghg_forcings/{gas}_historical.pt"
                )
                if gas == "carbon":
                    data = data * 4
                data = torch.tile(data[:12], (150, 1, 1))

            elif exp == "ssp534-over":
                # need to start at 2040
                data = torch.load(
                    f"{scratch}/reference_data/processed_spatial_monthly_ghg_forcings/{gas}_ssp534.pt"
                )[(12 * 25) :]
            else:
                data = torch.load(
                    f"{scratch}/reference_data/processed_spatial_monthly_ghg_forcings/{gas}_{exp}.pt"
                )

            data = data.swapdims(-1, -2)  # this is a patch for an upstream bug
            gas_array = xr.DataArray(
                data[: len(ds.coords["time"])],
                dims=("time", "lat", "lon"),
                coords={
                    "time": ds.coords["time"],
                    "lat": ds.coords["lat"],
                    "lon": ds.coords["lon"],
                },
                attrs={"long_name": f"{gas.capitalize()} concentration"},  # optional metadata
            )
            ds[gas] = gas_array

        # aerosol
        for aerosol_var_index in range(len(aerosol_vars)):
            if exp == "piControl":
                data = torch.load(f"{scratch}/reference_data/aerosol_historical.pt")
                data = torch.tile(data[:, :12], (1, 250, 1, 1))[aerosol_var_index]
            elif exp == "abrupt-4xCO2":
                data = torch.load(f"{scratch}/reference_data/aerosol_historical.pt")
                data = torch.tile(data[:, :12], (1, 150, 1, 1))[aerosol_var_index]
            elif exp == "ssp534-over":
                data = torch.load(f"{scratch}/reference_data/aerosol_{exp}.pt").numpy()[
                    :, 12 * 25 :
                ]
                data = data[aerosol_var_index]
            else:
                data = torch.load(f"{scratch}/reference_data/aerosol_{exp}.pt").numpy()
                data = data[aerosol_var_index]
            aero_array = xr.DataArray(
                data[: len(ds.coords["time"])],
                dims=("time", "lat", "lon"),
                coords={
                    "time": ds.coords["time"],
                    "lat": ds.coords["lat"],
                    "lon": ds.coords["lon"],
                },
                attrs={
                    "long_name": (f"{aerosol_vars[aerosol_var_index].capitalize()} concentration")
                },  # optional metadata
            )
            ds[aerosol_vars[aerosol_var_index]] = aero_array

        # nonspatial ssi
        for ssi_band_index in range(6):
            if exp == "historical":
                ssi = solar_forcings[: (12 * 165)]
            elif exp == "piControl":
                # ssi = solar_forcings[:12].repeat(250)  # repeat same year over and over  # noqa: E501
                ssi = (
                    torch.tensor(solar_forcings[:1])
                    .unsqueeze(0)
                    .repeat(1, 3000, 1)
                    .reshape(-1, 6)
                    .numpy()
                )  # repeat same 25 years over and over
            elif exp == "abrupt-4xCO2":
                ssi = (
                    torch.tensor(solar_forcings[:1])
                    .unsqueeze(0)
                    .repeat(1, 1800, 1)
                    .reshape(-1, 6)
                    .numpy()
                )  # repeat same 25 years over and over
            elif exp == "ssp534-over":
                ssi = solar_forcings[((12 * 165) + (12 * 25)) :]
            else:
                ssi = solar_forcings[(12 * 165) :]

            ssi_array = xr.DataArray(
                ssi[: len(ds.coords["time"]), ssi_band_index],
                dims=("time"),
                coords={"time": ds.coords["time"]},
                attrs={"long_name": f"ssi {ssi_band_index}"},  # optional metadata
            )
            ds[f"ssi_{ssi_band_index}"] = ssi_array

        # ozone
        if exp == "piControl":
            data = torch.load(f"{scratch}/reference_data/ozone_historical.pt")
            data = torch.tile(data[:12], (250, 1, 1, 1))
        elif exp == "abrupt-4xCO2":
            data = torch.load(f"{scratch}/reference_data/ozone_historical.pt")
            data = torch.tile(data[:12], (150, 1, 1, 1))
        elif exp == "ssp534-over":
            data = torch.load(f"{scratch}/reference_data/ozone_{exp}.pt")[(12 * 25) :]
        else:
            data = torch.load(f"{scratch}/reference_data/ozone_{exp}.pt")
            # last_year_data = data[-12:]  # dataset doesnt include 2100, repeat 2099  # noqa: E501
        # if(exp in ['ssp126','ssp245','ssp370','ssp585']):
        #     data = torch.concatenate([data,last_year_data],axis=0)
        # elif(exp in ['ssp119','ssp434','ssp460','ssp534-over']):
        #     data = data[12:] #this data has 2014_hybrid
        data = data.swapdims(-1, -2)  # this is a patch for an upstream bug6
        ozone_array = xr.DataArray(
            data[: len(ds.coords["time"])],
            dims=("time", "ozone_lev", "lat", "lon"),
            coords={
                "time": ds.coords["time"],
                "ozone_lev": [x for x in range(66)],
                "lat": ds.coords["lat"],
                "lon": ds.coords["lon"],
            },
            attrs={"long_name": "Ozone"},  # optional metadata
        )
        if full_ozone:
            ds["ozone"] = ozone_array
        else:
            for i in range(6):
                ds[f"ozone_{i}"] = ozone_array.isel(ozone_lev=i)

        # exp_id for basic forcing experiment
        exp_id = experiments.index(exp)
        exp_id_tensor = torch.tensor([exp_id])
        exp_id_tensor = exp_id_tensor.repeat(len(ds.coords["time"]))

        exp_id_array = xr.DataArray(
            exp_id_tensor,
            dims=("time"),
            coords={"time": ds.coords["time"]},
            attrs={"long_name": f"exp {exp_id}"},  # optional metadata
        )
        ds["exp_id"] = exp_id_array
        if rotate_ssps:
            ds.to_netcdf(f"{scratch}/interpolation_dataset_final_rotated/{out_file}")
        elif full_ozone:
            ds.to_netcdf(f"{scratch}/interpolation_dataset_full_ozone/{out_file}")
        elif yearly_temperature:
            ds = ds.isel(time=slice(30, -30))  # remove edge effects
            ds.to_netcdf(f"{scratch}/interpolation_dataset_filled_in_yearly_temperature/{out_file}")
        else:
            ds.to_netcdf(f"{scratch}/full_ocean_dataset_forced/{out_file}")
