"""Shared forcing-loading logic (GHG, aerosol, ozone, solar).

Forcings are driven purely by experiment/scenario, not by the source ESM, so
this is identical across every source model's add_forcings_*.py script.
Used by both add_forcings_canesm5.py and add_forcings.py (IPSL).

Verified against real data (see preprocessing/verify_ozone_and_ssp534_patches.py
and the ssp534-over aerosol/GHG/ozone/SSI checks run against
memmap_filled_in/r1i1p1f1_ssp534-over_interpolation.memmap): this is the
correct, current forcing-assembly logic.
"""

import torch

# fixed order so exp_id lines up with memmap_filled_in's encoding, even
# though canesm5 doesn't have piControl/abrupt-4xCO2 files yet.
EXPERIMENTS = [
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
    "1pctCO2",
    "hist-GHG",
    "hist-aer",
    "hist-stratO3",
    "hist-piAer",
]

# hist-GHG/hist-aer/hist-stratO3 (DAMIP) vary exactly one forcing group on its
# real historical trajectory and hold every other forcing at pre-industrial
# (piControl) climatology. The DAMIP runs themselves span 1850-2020, but our
# reference_data/*_historical.pt files only cover 1850-2014 (the standard
# CMIP6 `historical` period, same truncation already used everywhere else in
# this pipeline) -- so these experiments are truncated to that same 165-year
# span rather than the full DAMIP length. 1pctCO2 (CO2 ramping 1%/yr,
# compounding, from its pre-industrial level, everything else held fixed)
# spans its own native 150 years (1850-1999), matching abrupt-4xCO2.
HIST_SINGLE_FORCING_YEARS = 165
PCT_CO2_YEARS = 150

SPATIAL_GASES = ["methane", "carbon", "nitrous"]
AEROSOL_VARS = [
    "load_ASNO3M",
    "load_CSNO3M",
    "load_CINO3M",
    "load_SO4",
    "load_AIBCM",
    "load_ASBCM",
]

# Exact ozone-band indices (out of 66) that memmap_filled_in's spatial_forcings
# channels 9-18 correspond to -- confirmed by exact-value fingerprinting
# against a real memmap_filled_in file (not a contiguous/evenly-spaced
# subsample, so this must be reproduced verbatim, not re-derived).
OZONE_BAND_INDICES = [65, 63, 56, 49, 45, 40, 33, 26, 12, 1]

# hist-stratO3 varies stratospheric ozone only -- tropospheric ozone must stay
# at piControl climatology like every other non-active forcing. The original
# ozone source file's plev coordinate (which band is strat vs. tropo) is gone
# -- forcings_preprocessing.py's docstring notes it lived on a decommissioned
# cluster -- so this split is inferred empirically from each band's 1850-2014
# time series (see preprocessing/verification/): bands 65, 63, 56 rise
# steadily with no depletion signature (tropospheric, pollution-driven).
# Bands 49, 45, 40, 33, 26, 12 dip through the 1980s-90s and partially recover
# post-2000 -- the CFC/ozone-hole/Montreal-Protocol signature (stratospheric).
# Band 1 (the highest-altitude of the 10) shows a weak, ambiguous signal
# that doesn't clearly match either pattern; held at piControl here as the
# conservative default rather than guessed into either bucket -- revisit if
# the raw climoz plev values ever resurface.
TROPOSPHERIC_OZONE_BAND_INDICES = [65, 63, 56]
STRATOSPHERIC_OZONE_BAND_INDICES = [49, 45, 40, 33, 26, 12]


def load_solar_forcings(scratch: str, reference_data_dir: str = "reference_data") -> torch.Tensor:
    return torch.load(f"{scratch}/{reference_data_dir}/solar_forcings_monthly_interpolated.pt")


def spatial_ghg_tensor(
    scratch: str, gas: str, exp: str, n_time: int, reference_data_dir: str = "reference_data"
) -> torch.Tensor:
    if exp == "piControl":
        data = torch.load(
            f"{scratch}/{reference_data_dir}/processed_spatial_monthly_ghg_forcings/{gas}_historical.pt"
        )
        data = torch.tile(data[:12], (250, 1, 1))
    elif exp == "abrupt-4xCO2":
        data = torch.load(
            f"{scratch}/{reference_data_dir}/processed_spatial_monthly_ghg_forcings/{gas}_historical.pt"
        )
        if gas == "carbon":
            data = data * 4
        data = torch.tile(data[:12], (150, 1, 1))
    elif exp == "1pctCO2":
        # CO2 ramps 1%/yr compounding from its pre-industrial level; every
        # other forcing (including methane/nitrous here) held fixed.
        data = torch.load(
            f"{scratch}/{reference_data_dir}/processed_spatial_monthly_ghg_forcings/{gas}_historical.pt"
        )
        data = torch.tile(data[:12], (PCT_CO2_YEARS, 1, 1))
        if gas == "carbon":
            year_index = torch.arange(PCT_CO2_YEARS).repeat_interleave(12)[: data.shape[0]]
            data = data * (1.01**year_index).view(-1, 1, 1)
    elif exp in ("hist-GHG", "hist-piAer"):
        # all well-mixed GHGs follow their real historical trajectory here --
        # active forcing for hist-GHG, held-fixed-elsewhere backdrop for
        # hist-piAer (AerChemMIP counterpart of hist-aer: aerosol is the one
        # forcing held at piControl, everything else -- GHGs, ozone, solar --
        # follows the real historical trajectory). Truncated to 1850-2014 via
        # the final data[:n_time] below, same as everywhere else
        # `historical.pt` is used.
        data = torch.load(
            f"{scratch}/{reference_data_dir}/processed_spatial_monthly_ghg_forcings/{gas}_historical.pt"
        )
    elif exp in ("hist-aer", "hist-stratO3"):
        # GHGs held at piControl climatology here -- aerosol (hist-aer) or
        # ozone (hist-stratO3) is the active forcing instead.
        data = torch.load(
            f"{scratch}/{reference_data_dir}/processed_spatial_monthly_ghg_forcings/{gas}_historical.pt"
        )
        data = torch.tile(data[:12], (HIST_SINGLE_FORCING_YEARS, 1, 1))
    elif exp == "ssp534-over":
        data = torch.load(
            f"{scratch}/{reference_data_dir}/processed_spatial_monthly_ghg_forcings/{gas}_ssp534.pt"
        )[(12 * 25) :]
    else:
        data = torch.load(
            f"{scratch}/{reference_data_dir}/processed_spatial_monthly_ghg_forcings/{gas}_{exp}.pt"
        )
    data = data.swapdims(-1, -2)  # patch for upstream bug, same as add_forcings.py
    return data[:n_time]


def aerosol_tensor(
    scratch: str, var_index: int, exp: str, n_time: int, reference_data_dir: str = "reference_data"
) -> torch.Tensor:
    if exp == "piControl":
        data = torch.load(f"{scratch}/{reference_data_dir}/aerosol_historical.pt")
        data = torch.tile(data[:, :12], (1, 250, 1, 1))[var_index]
    elif exp == "abrupt-4xCO2":
        data = torch.load(f"{scratch}/{reference_data_dir}/aerosol_historical.pt")
        data = torch.tile(data[:, :12], (1, 150, 1, 1))[var_index]
    elif exp == "1pctCO2":
        data = torch.load(f"{scratch}/{reference_data_dir}/aerosol_historical.pt")
        data = torch.tile(data[:, :12], (1, PCT_CO2_YEARS, 1, 1))[var_index]
    elif exp == "hist-aer":
        # aerosol is the active forcing here -- real historical trajectory
        # (truncated to 1850-2014 via the final data[:n_time] below).
        data = torch.load(f"{scratch}/{reference_data_dir}/aerosol_historical.pt")
        data = data[var_index]
    elif exp in ("hist-GHG", "hist-stratO3", "hist-piAer"):
        # aerosol held at piControl climatology here -- for hist-piAer this
        # is the active (counterfactual) forcing: real AerChemMIP hist-piAer
        # holds aerosol emissions fixed at pre-industrial while GHG/ozone/
        # solar follow their real historical trajectory (see
        # spatial_ghg_tensor/ozone_bands_tensor/ssi_tensor's hist-piAer
        # branches) -- the exact opposite of hist-aer.
        data = torch.load(f"{scratch}/{reference_data_dir}/aerosol_historical.pt")
        data = torch.tile(data[:, :12], (1, HIST_SINGLE_FORCING_YEARS, 1, 1))[var_index]
    elif exp == "ssp534-over":
        data = torch.load(f"{scratch}/{reference_data_dir}/aerosol_{exp}.pt")[:, (12 * 25) :]
        data = data[var_index]
    else:
        data = torch.load(f"{scratch}/{reference_data_dir}/aerosol_{exp}.pt")
        data = data[var_index]
    return data[:n_time]


def ozone_bands_tensor(
    scratch: str, exp: str, n_time: int, reference_data_dir: str = "reference_data"
) -> torch.Tensor:
    if exp == "piControl":
        data = torch.load(f"{scratch}/{reference_data_dir}/ozone_historical.pt")
        data = torch.tile(data[:12], (250, 1, 1, 1))
    elif exp == "abrupt-4xCO2":
        data = torch.load(f"{scratch}/{reference_data_dir}/ozone_historical.pt")
        data = torch.tile(data[:12], (150, 1, 1, 1))
    elif exp == "1pctCO2":
        data = torch.load(f"{scratch}/{reference_data_dir}/ozone_historical.pt")
        data = torch.tile(data[:12], (PCT_CO2_YEARS, 1, 1, 1))
    elif exp == "hist-stratO3":
        # Only the stratospheric bands are the active forcing; tropospheric
        # bands (and the one unclassified band) held at piControl climatology
        # -- see STRATOSPHERIC_OZONE_BAND_INDICES's comment above.
        real = torch.load(f"{scratch}/{reference_data_dir}/ozone_historical.pt")
        data = torch.tile(real[:12], (HIST_SINGLE_FORCING_YEARS, 1, 1, 1))
        data[:, STRATOSPHERIC_OZONE_BAND_INDICES] = real[:, STRATOSPHERIC_OZONE_BAND_INDICES]
    elif exp in ("hist-GHG", "hist-aer"):
        # ozone held at piControl climatology here.
        data = torch.load(f"{scratch}/{reference_data_dir}/ozone_historical.pt")
        data = torch.tile(data[:12], (HIST_SINGLE_FORCING_YEARS, 1, 1, 1))
    elif exp == "hist-piAer":
        # ozone follows its real historical trajectory here (only aerosol is
        # held at piControl for hist-piAer) -- same source file as
        # exp="historical" itself, but that case reaches this same file via
        # the generic else branch below (its filename literally matches
        # ozone_historical.pt); hist-piAer needs it named explicitly.
        data = torch.load(f"{scratch}/{reference_data_dir}/ozone_historical.pt")
    elif exp == "ssp534-over":
        data = torch.load(f"{scratch}/{reference_data_dir}/ozone_{exp}.pt")[(12 * 25) :]
    else:
        data = torch.load(f"{scratch}/{reference_data_dir}/ozone_{exp}.pt")
    data = data.swapdims(-1, -2)  # patch for upstream bug, same as add_forcings.py
    return data[:n_time]


def ozone_subset_and_full(
    scratch: str, exp: str, n_time: int, reference_data_dir: str = "reference_data"
):
    full = ozone_bands_tensor(scratch, exp, n_time, reference_data_dir)  # [time, 66, lat, lon]
    return full[:, OZONE_BAND_INDICES], full


def ssi_tensor(solar_forcings: torch.Tensor, exp: str, n_time: int) -> torch.Tensor:
    if exp in ("historical", "hist-piAer"):
        ssi = solar_forcings[: (12 * 165)]
    elif exp == "piControl":
        ssi = solar_forcings[:1].repeat(3000, 1)
    elif exp == "abrupt-4xCO2":
        ssi = solar_forcings[:1].repeat(1800, 1)
    elif exp == "1pctCO2":
        ssi = solar_forcings[:1].repeat(12 * PCT_CO2_YEARS, 1)
    elif exp in ("hist-GHG", "hist-aer", "hist-stratO3"):
        # solar cycle held at piControl (constant) here -- DAMIP single-forcing
        # runs hold everything but the named forcing fixed, including solar.
        ssi = solar_forcings[:1].repeat(12 * HIST_SINGLE_FORCING_YEARS, 1)
    elif exp == "ssp534-over":
        ssi = solar_forcings[((12 * 165) + (12 * 25)) :]
    else:
        ssi = solar_forcings[(12 * 165) :]
    return ssi[:n_time]


def build_forcings(
    scratch: str,
    exp: str,
    n_time: int,
    solar_forcings: torch.Tensor,
    reference_data_dir: str = "reference_data",
):
    spatial = []
    for gas in SPATIAL_GASES:
        spatial.append(spatial_ghg_tensor(scratch, gas, exp, n_time, reference_data_dir))
    for var_index in range(len(AEROSOL_VARS)):
        spatial.append(aerosol_tensor(scratch, var_index, exp, n_time, reference_data_dir))
    ozone_subset, ozone_full = ozone_subset_and_full(scratch, exp, n_time, reference_data_dir)
    for i in range(ozone_subset.shape[1]):
        spatial.append(ozone_subset[:, i])
    spatial_forcings = torch.stack(spatial, dim=0).float()  # [19, time, lat, lon]

    non_spatial_forcings = ssi_tensor(solar_forcings, exp, n_time).float().T  # [6, time]

    return spatial_forcings, non_spatial_forcings, ozone_full.float()
