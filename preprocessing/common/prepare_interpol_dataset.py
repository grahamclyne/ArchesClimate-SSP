"""Shared esmvaltool diagnostic logic: merge per-variable cubes for one
(experiment, ensemble) group and derive net_flux.

Used identically by both prepare_interpol_dataset.py (IPSL, writes NetCDF)
and prepare_interpol_dataset_canesm5.py (CanESM5, writes memmap directly) --
this step is driven entirely by esmvaltool's input_data metadata and variable
names, not by source model, so there's nothing model-specific here.
"""

from typing import Any

import iris
import xarray as xr
from ncdata.iris_xarray import cubes_to_xarray

FLUX_VARS = ["rsus", "rsds", "rlus", "rlds", "hfss", "hfls"]
DROPPABLE_AUX_VARS = ["height", "time_bnds", "lat_bnds", "lon_bnds"]


def merge_cubes_and_compute_net_flux(entries: list) -> Any:
    """Merge one (experiment, ensemble) group's per-variable esmvaltool
    input_data entries into a single xarray Dataset, with net_flux derived
    from the six surface radiation/heat-flux variables.

    Args:
        entries: list of esmvaltool input_data dicts (all sharing the same
            exp/ensemble), each with a "filename" key pointing at a
            single-variable NetCDF file.

    Returns:
        xarray.Dataset with net_flux added and the raw flux/aux variables
        dropped.
    """
    cubes = [cubes_to_xarray(iris.load_cube(entry["filename"])) for entry in entries]
    dataset = xr.merge(cubes, compat="override")

    drop_vars = [v for v in DROPPABLE_AUX_VARS if v in dataset]
    if drop_vars:
        dataset = dataset.drop_vars(drop_vars)

    dataset["net_flux"] = (
        dataset["rsus"]
        - dataset["rsds"]
        + dataset["rlus"]
        - dataset["rlds"]
        + dataset["hfss"]
        + dataset["hfls"]
    )
    return dataset.drop_vars(FLUX_VARS)


def group_by_exp_ensemble(input_data: Any) -> dict:
    """Group esmvaltool input_data entries by (exp, ensemble)."""
    grouped: dict = {}
    for entry in input_data:
        key = (entry["exp"], entry["ensemble"])
        grouped.setdefault(key, []).append(entry)
    return grouped
