from collections.abc import Callable
from pathlib import Path

import torch
from tensordict.tensordict import TensorDict
from tqdm import tqdm


class XarrayDataset(torch.utils.data.Dataset):
    """Dataset to read a list of xarray files and iterate through it by timestamp.

    constraint: it should be indexed by at least one dimension named "time".

    Child classes that inherit this class, should implement convert_to_tensordict()
    which converts an xarray dataset into a tensordict (to feed into the model).
    """

    def __init__(
        self,
        path: str,
        variables: dict[str, list[str]],
        filename_filter: Callable = lambda _: True,  # condition to keep file in dataset
        limit_examples: int | None = None,
        timestamp_key: Callable = lambda x: x[-1],
    ):
        """Initialize the dataset.

        Args:
        path: Single filepath or directory holding xarray files.
        variables: Dict holding xarray data variable lists mapped by their keys
            to be processed into tensordict.
            e.g. {surface: [data_var1, datavar2, ...], level: [...]}
            Used in convert_to_tensordict() to read data arrays in the xarray
            dataset and convert to tensordict.
        filename_filter: To filter files within `path` based on filename.
        limit_examples: Return set number of examples in dataset.
        timestamp_key: Extracts the sort/index timestamp from a parsed
            filename tuple; defaults to the last element.
        """
        self.filename_filter = filename_filter
        self.variables = variables

        if not Path(path).exists():
            raise ValueError("Path does not exist:", path)

        if Path(path).is_file() and "." in path.split("/")[-1]:
            self.files = [path]
        else:
            files = list(Path(path).glob("*"))
            if len(files) == 0:
                raise ValueError("No files found under path:", path)

            self.files = sorted(
                [str(x) for x in files if filename_filter(x.name)],
                key=lambda x: x.replace("_6h", "_06h").replace("_0h", "_00h"),
            )
            if len(self.files) == 0:
                raise ValueError("filename_filter filtered all files under path:", path)

        self.timestamps = []

        for fid, f in tqdm(enumerate(self.files)):
            obs = TensorDict.load_memmap(f)
            file_stamps = [(fid, i, t) for (i, t) in enumerate(obs["time"])]

            self.timestamps.extend(file_stamps)

            if (
                limit_examples and len(self.timestamps) > limit_examples
            ):  # get fraction of full dataset
                self.timestamps = self.timestamps[:limit_examples]
                break

        self.timestamps = sorted(self.timestamps, key=timestamp_key)

        self.id2pt = dict(enumerate(self.timestamps))

    def __len__(self):
        return len(self.id2pt)

    def _ensure_cache(self):
        if getattr(self, "_file_cache", None) is None:
            self._file_cache = {}

    def __getitem__(self, i):
        file_id, line_id, timestamp = self.id2pt[i]

        file_path = self.files[file_id]

        self._ensure_cache()

        if file_path not in self._file_cache:
            self._file_cache[file_path] = TensorDict.load_memmap(file_path)
        data = self._file_cache[file_path]

        spatial_forcings = data["spatial_forcings"][:, line_id]
        if "ozone_0" not in self.variables["spatial_forcings"]:
            spatial_forcings = spatial_forcings[:8]
        elif ("ozone_1") not in self.variables["spatial_forcings"]:
            spatial_forcings = spatial_forcings[:9]
        elif ("ozone_7") not in self.variables["spatial_forcings"]:
            spatial_forcings = spatial_forcings[:15]
        if "ozone" in data.keys():
            # memmap_filled_in_full_ozone: data['ozone'] is a MemoryMappedTensor
            # of shape [1, n_timesteps, 66, H, W] (full pressure-level ozone).
            # _ozone_band_indices is set by the child class to select 10 levels.
            #
            # Some realizations in this dataset (the DAMIP ones rerun through
            # Stage 3's add_forcings) already have those same 10 bands baked
            # into their on-disk spatial_forcings tensor -- confirmed bit-identical
            # to data['ozone'][..., _ozone_band_indices]. Others don't. Detect
            # that via the channel count so we don't append (and duplicate) the
            # bands for realizations that already have them, which was also
            # producing inconsistent spatial_forcings channel counts across
            # realizations within the same dataset instance.
            _indices = getattr(self, "_ozone_band_indices", None)
            _base_channels = getattr(self, "_base_spatial_forcing_channels", None)
            _already_has_ozone_bands = (
                _base_channels is not None and spatial_forcings.shape[0] > _base_channels
            )
            if _indices is not None and not _already_has_ozone_bands:
                ozone_full = data["ozone"][0, line_id]  # [66, H, W]
                bands = ozone_full[_indices]  # [10, H, W]
                spatial_forcings = torch.cat([spatial_forcings, bands], dim=0)
        if "level" in data.keys():
            out = TensorDict(
                {
                    "lev": data["lev"][:, line_id],
                    "level": data["level"][:, line_id][
                        : len(self.variables["level"])
                    ],  # Slice early
                    "surface": data["surface"][:, line_id, None],
                    "non_spatial_forcings": data["non_spatial_forcings"][:, line_id],
                    "spatial_forcings": spatial_forcings,
                }
            )
            out["level"] = out["level"][: len(self.variables["level"])]

        else:
            out = TensorDict(
                {
                    "lev": data["lev"][:, line_id],
                    "surface": data["surface"][:, line_id, None],
                    "non_spatial_forcings": data["non_spatial_forcings"][:, line_id],
                    "spatial_forcings": data["spatial_forcings"][
                        :, line_id
                    ],  # override ozone logic here
                }
            )
        # Mirrors 'level' slicing above -- the on-disk 'surface' tensor always has
        # the full physical channel layout (currently 8: tas, psl, pr, uas, vas,
        # ps, net_flux, evspsbl, in that order); self.variables['surface'] (from
        # CMIPForecastLeadTime's surface_variables) is only ever a prefix subset
        # of that layout in practice, so slicing to its length selects exactly
        # the requested leading variables. Previously unsliced -- every config
        # so far has requested all 8 (a no-op slice), so this was never
        # exercised until a config asked for fewer.
        out["surface"] = out["surface"][: len(self.variables["surface"])]
        # ORDER is very important here. first check if aero is in, which we cut
        # out of the middle if not. then we can remove the beginning which is
        # ghg, finally we check if we remove the end, ozone
        if "load_ASNO3M" not in self.variables["spatial_forcings"]:
            out["spatial_forcings"] = torch.concatenate(
                [out["spatial_forcings"][:3], out["spatial_forcings"][9:]], dim=0
            )
        if "methane" not in self.variables["spatial_forcings"]:
            out["spatial_forcings"] = out["spatial_forcings"][3:]
        if len(self.variables["non_spatial_forcings"]) == 0:
            out["non_spatial_forcings"] = torch.tensor([])
        return out
