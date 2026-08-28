import os
from typing import Any

import numpy as np
import torch
from common.prepare_interpol_dataset import (
    group_by_exp_ensemble,
    merge_cubes_and_compute_net_flux,
)
from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic
from tensordict.tensordict import TensorDict

VARIABLES = {
    "surface": ["tas", "psl", "pr", "uas", "vas", "ps", "net_flux", "evspsbl", "huss"],
    "level": ["hus", "ta", "ua", "va", "zg"],
    "lev": ["thetao"],
}


def run_my_diagnostic(cfg: Any) -> None:
    scratch = os.environ["SCRATCH"]
    # Overridable for one-off comparison runs (e.g. testing an alternate
    # regrid scheme against the production output) without touching the
    # default directory the real pipeline reads from.
    out_dir = os.environ.get(
        "CANESM5_INTERPOLATION_OUT_DIR", f"{scratch}/interpolation_canesm5_datasets"
    )
    os.makedirs(out_dir, exist_ok=True)

    group_metadata(cfg["input_data"].values(), "ensemble")
    grouped = group_by_exp_ensemble(cfg["input_data"].values())

    for (exp, ensemble), entries in grouped.items():
        out_path = f"{out_dir}/{ensemble}_{exp}_interpolation.memmap"
        if os.path.exists(out_path):
            print(f"{out_path} already exists, skipping")
            continue

        print(exp, ensemble)
        for entry in entries:
            print(entry["filename"])
        dataset = merge_cubes_and_compute_net_flux(entries)

        np_arrays = {}
        for key, variable_list in VARIABLES.items():
            if variable_list:
                np_arrays[key] = dataset[variable_list].to_array().to_numpy()
            else:
                np_arrays[key] = np.empty((0,))

        tdict = TensorDict({key: torch.from_numpy(arr).float() for key, arr in np_arrays.items()})
        tdict["time"] = dataset.time.values
        tdict.memmap(out_path)
        print(f"Written memmap to: {out_path}")


if __name__ == "__main__":
    with run_diagnostic() as config:
        run_my_diagnostic(config)
