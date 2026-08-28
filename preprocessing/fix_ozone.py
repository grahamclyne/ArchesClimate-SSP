"""Retroactively patch spatial_forcings' ozone channels in memmap_filled_in.

Historically, add_forcings.py wrote only 6 raw ozone bands (ozone_lev 0-5) into
spatial_forcings, giving 15 total channels (3 GHGs + 6 aerosols + 6 ozone).
This replaces those with the 10-band selection (OZONE_BAND_INDICES) confirmed
by exact-value fingerprinting to be the ones actually used, giving 19 total
channels (3 GHGs + 6 aerosols + 10 ozone) -- the layout memmap_filled_in's
consumers (e.g. dataloaders/cmip_random_lead_time.py) and add_forcings_canesm5.py
now assume.

Idempotent and safe to rerun: files that already have 19 spatial_forcings
channels are skipped, so this only needs to touch files add_forcings.py
produced before this fix existed.
"""

import os
from pathlib import Path

import torch
from tensordict import TensorDict

SCRATCH = os.environ["SCRATCH"]
MEMMAP_DIR = Path(f"{SCRATCH}/memmap_filled_in")
OZONE_DIR = Path(f"{SCRATCH}/reference_data")

SCENARIOS = [
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

# Same 10 ozone-band indices as add_forcings_canesm5.py's OZONE_BAND_INDICES
# (there: [65, 63, 56, 49, 45, 40, 33, 26, 12, 1], equivalent modulo the
# negative-vs-positive indexing convention).
OZONE_INDICES = [-1, -3, -10, -17, -21, -26, -33, -40, -54, -65]


def process_memmaps() -> None:
    current_ozone_tensor = None
    current_sc = None

    # Sort files to group scenarios and minimize re-loading.
    for mm_file in sorted(MEMMAP_DIR.glob("*.memmap")):
        matching_sc = next((sc for sc in SCENARIOS if sc in mm_file.name), None)
        if not matching_sc:
            continue

        try:
            td = TensorDict.load_memmap(mm_file)
            forcings = td["spatial_forcings"]
        except Exception as e:  # noqa: BLE001
            print(f"Error opening {mm_file.name}: {e}")
            continue

        if forcings.shape[0] == 19:
            print(f"Skipping {mm_file.name}: already has 19 spatial_forcings channels.")
            continue

        print(f"\nProcessing {mm_file.name}...")

        if matching_sc != current_sc:
            del current_ozone_tensor
            current_ozone_tensor = None

            if matching_sc == "piControl":
                print("Handling piControl: loading historical and tiling...")
                data = torch.load(OZONE_DIR / "ozone_historical.pt", map_location="cpu")
                current_ozone_tensor = torch.tile(data[:12], (250, 1, 1, 1))
            elif matching_sc == "abrupt-4xCO2":
                # No dedicated ozone_abrupt-4xCO2.pt exists; matches
                # add_forcings.py's handling (tile historical's first year to
                # 150 years). The original fix_ozone.py never had this branch
                # (only piControl), which is exactly why these files were
                # never patched.
                print("Handling abrupt-4xCO2: loading historical and tiling...")
                data = torch.load(OZONE_DIR / "ozone_historical.pt", map_location="cpu")
                current_ozone_tensor = torch.tile(data[:12], (150, 1, 1, 1))
            else:
                ozone_path = OZONE_DIR / f"ozone_{matching_sc}.pt"
                if not ozone_path.exists():
                    print(f"Skipping: {ozone_path} not found.")
                    current_sc = None
                    continue
                print(f"Loading ozone source for {matching_sc}...")
                current_ozone_tensor = torch.load(ozone_path, map_location="cpu")

            current_sc = matching_sc

        try:
            selected_ozone = current_ozone_tensor[:, OZONE_INDICES].swapdims(0, 1).swapdims(-1, -2)
            new_forcings = torch.concat([forcings[:9], selected_ozone], dim=0)
            td.set("spatial_forcings", new_forcings, inplace=False)
            td.memmap_()
            print(f"Updated {mm_file.name} successfully.")
            del td
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {mm_file.name}: {e}")


if __name__ == "__main__":
    process_memmaps()
