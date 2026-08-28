import importlib.resources

import cftime
import torch
from tensordict.tensordict import TensorDict

from .. import stats as ArchesClimate_stats
from .netcdf import XarrayDataset

# Ozone level indices (out of 66) selected as spatial forcing channels.
# Index 65 = surface, index 0 = TOA. Same order as ozone_0..ozone_9 in
# spatial_forcing_variables. Confirmed against memmap_filled_in fingerprinting.
_OZONE_BAND_INDICES = [65, 63, 56, 49, 45, 40, 33, 26, 12, 1]


def _epoch_seconds(t) -> int:
    """Epoch seconds for a single timestamp, whatever its backing type.

    numpy.datetime64 (e.g. IPSL, standard/proleptic-Gregorian calendar)
    supports `.item()` directly. CMIP6 models on a non-standard calendar
    (e.g. CanESM5's 365-day/noleap) surface as cftime.datetime objects
    instead, which have no `.item()` and need calendar-aware conversion.
    """
    if hasattr(t, "item"):
        return t.item() // 10**9
    return int(cftime.date2num(t, units="seconds since 1970-01-01", calendar=t.calendar))


class CMIPForecastLeadTime(XarrayDataset):
    """Load DCPP data for the forecast task.

    Loads previous timestep and multiple future timesteps if configured.
    Also handles normalization.
    """

    def __init__(
        self,
        path="/path/to/data/",
        forcings_path="data/",
        domain="train",
        filename_filter=None,
        lead_time_months=1,
        multistep=1,
        lead_times=None,
        return_all_lead_times=False,
        load_prev=True,
        norm_scheme="spatial",
        limit_examples: int = 0,
        variables=None,
        surface_variables=None,
        level_variables=None,
        spatial_forcing_variables=[
            "methane",
            # "cfc11",
            "carbon",
            "nitrous",
            "load_ASNO3M",
            "load_CSNO3M",
            "load_CINO3M",
            "load_SO4",
            "load_AIBCM",
            "load_ASBCM",
        ],
        non_spatial_forcing_variables=[
            "CO2",
            "CFC12eq",
            "CFC12eq",
            "CH4",
            "N2O",
            "ssi_0",
            "ssi_1",
            "ssi_2",
            "ssi_3",
        ],
        pressure_levels=[],
        depth_levels=[
            0.50576,
            3.8562799,
            8.092519,
            13.991038,
            22.757616,
            35.740204,
            53.850636,
            77.61116,
            108.03028,
            147.40625,
        ],
        train_filter=["historical", "ssp245", "ssp585", "ssp126", "ssp434", "piControl"],
        test_filter=["ssp370"],
        val_filter=["ssp245"],
        pf_n_steps=8,
        skip_pf=False,
    ):
        """Initialize the dataset.

        Args:
        path: Single filepath or directory holding files.
        forcings_path: Directory holding the forcing tensors (spatial/non-spatial
            forcings, ozone) referenced alongside `path`'s state data.
        domain: Specify data split for the filename filters (eg. train, val, test, testz0012..)
        filename_filter: To filter files within data directory based on filename.
        lead_time_months: Time difference between current state and previous and future states.
        multistep: How many future states to load. By default, loads one
            (current time + lead_time_months).
        lead_times: Explicit list of lead times (months) to load; overrides
            the multistep-derived default when given.
        return_all_lead_times: Whether to return every configured lead time per
            item, rather than a single randomly-sampled one.
        load_prev: Whether to load state at previous timestamp (current time - lead_time_months).
        norm_scheme: Which normalization stats variant to load (matches a
            `cmip_stats_<norm_scheme>.pt` file under ArchesClimate/stats).
        limit_examples: Return set number of examples in dataset.
        variables: Optional override dict of {surface, level, lev, spatial_forcings,
            non_spatial_forcings} variable name lists; defaults built from the
            variables below when not given.
        surface_variables: Surface-level state variable names to load.
        level_variables: Pressure-level state variable names to load.
        spatial_forcing_variables: Declared spatial-forcing channel names/order
            (see module docstring in preprocessing/common/forcings.py for the
            known aerosol-channel mislabeling caveat).
        non_spatial_forcing_variables: Declared non-spatial (scalar) forcing
            channel names/order.
        pressure_levels: Pressure levels (hPa) corresponding to level_variables'
            vertical axis.
        depth_levels: Ocean depth levels (m) corresponding to the 'lev'
            (thetao) vertical axis.
        train_filter: Experiment names included in the "train" domain filter.
        test_filter: Experiment names included in the "test" domain filter.
        val_filter: Experiment names included in the "val" domain filter.
        pf_n_steps: Number of pushforward steps to build `pf_future_states` for.
        skip_pf: Skip building pf_future_states entirely (see __getitem__'s
            state_only for the related per-item fast path).
        """
        self.__dict__.update(locals())  # concise way to update self with input arguments
        self.timedelta = 1
        # Normalize lead_times: allow user to pass a list of lead times (months)
        # Backwards compatible behaviour:
        # - if `lead_times` is provided it is used (list of ints)
        # - else if `multistep` > 1, create lead_times as
        #   [lead_time_months * (i+1) for i in range(multistep)]
        # - else use [lead_time_months]
        if lead_times is not None:
            # ensure a sorted list of ints
            self.lead_times = sorted([int(x) for x in lead_times])
        filename_filters = dict(
            all=(lambda _: True),
            train=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in train_filter]
            ),
            val=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in val_filter]
            ),
            test=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in test_filter]
            ),
            val1=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in ["ssp119"]]
            ),
            val2=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in ["ssp370"]]
            ),
            val3=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in ["ssp534-over"]]
            ),
            abrupt=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in ["abrupt-4xCO2"]]
            ),
            historical=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in ["historical"]]
            ),
            hist_aer=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in ["hist-aer"]]
            ),
            hist_piAer=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in ["hist-piAer"]]
            ),
            hist_stratO3=lambda x: any(
                substring in x
                for substring in [f"{str(y)}_interpolation.memmap" for y in ["hist-stratO3"]]
            ),
            empty=lambda x: False,
        )
        if filename_filter is None:
            filename_filter = filename_filters[domain]
        if variables is None:
            variables = dict(
                surface=surface_variables,
                level=level_variables,
                lev=["thetao"],
                spatial_forcings=spatial_forcing_variables,
                non_spatial_forcings=non_spatial_forcing_variables,
            )

        self.master_list_spatial_forcing_variables = [
            "methane",
            # "cfc11",
            "carbon",
            "nitrous",
            "load_ASNO3M",
            "load_CSNO3M",
            "load_CINO3M",
            "load_SO4",
            "load_AIBCM",
            "load_ASBCM",
        ]
        # Tell netcdf.py.__getitem__ which of the 66 pressure levels to select
        # when the memmap has a separate full-ozone tensor (memmap_filled_in_full_ozone).
        self._ozone_band_indices = _OZONE_BAND_INDICES
        # Channel count of spatial_forcings BEFORE ozone bands are added. Some
        # memmap_filled_in_full_ozone realizations (the DAMIP ones rerun through
        # Stage 3's add_forcings) already have the ozone bands baked into their
        # on-disk spatial_forcings tensor; others don't. netcdf.py uses this to
        # avoid appending the ozone.memmap bands a second time for the former.
        self._base_spatial_forcing_channels = len(self.master_list_spatial_forcing_variables)
        self.master_list_spatial_forcing_variables = self.master_list_spatial_forcing_variables + [
            "ozone_0",
            "ozone_1",
            "ozone_2",
            "ozone_3",
            "ozone_4",
            "ozone_5",
            "ozone_6",
            "ozone_7",
            "ozone_8",
            "ozone_9",
            "ozone_10",
        ]
        self.master_list_non_spatial_forcing_variables = [
            "ssi_0",
            "ssi_1",
            "ssi_2",
            "ssi_3",
            "ssi_4",
            "ssi_5",
            "exp_id",
        ]
        super().__init__(
            path,
            filename_filter=filename_filter,
            variables=variables,
            limit_examples=limit_examples,
            timestamp_key=lambda x: (x[0], x[1]),
        )
        if self.norm_scheme == "zeroes":
            self.data_mean = TensorDict(
                surface=torch.tensor(0),
                level=torch.tensor(0),
                lev=torch.tensor(0),
                spatial_forcings=torch.tensor(0),
                non_spatial_forcings=torch.tensor(0),
            )
            self.data_std = TensorDict(
                surface=torch.tensor(1),
                level=torch.tensor(1),
                lev=torch.tensor(1),
                spatial_forcings=torch.tensor(0),
                non_spatial_forcings=torch.tensor(0),
            )
        else:
            stats_file_path = f"cmip_stats_{self.norm_scheme}.pt"
            ArchesClimate_stats_path = importlib.resources.files(ArchesClimate_stats)
            norm_file_path = ArchesClimate_stats_path / stats_file_path
            spatial_norm_stats = torch.load(norm_file_path, weights_only=False)
            # a hack to add exp_id without needing to normalize it.
            spatial_norm_stats["non_spatial_forcings_mean"] = torch.concat(
                [spatial_norm_stats["non_spatial_forcings_mean"], torch.tensor([0])]
            )
            spatial_norm_stats["non_spatial_forcings_std"] = torch.concat(
                [spatial_norm_stats["non_spatial_forcings_std"], torch.tensor([1])]
            )
            if "full_ocean" in path:
                self.data_mean = TensorDict(
                    surface=spatial_norm_stats["surface_mean"],
                    lev=spatial_norm_stats["lev_mean"],
                    spatial_forcings=spatial_norm_stats["spatial_forcings_mean"],
                    non_spatial_forcings=spatial_norm_stats["non_spatial_forcings_mean"],
                )
                self.data_std = TensorDict(
                    surface=spatial_norm_stats["surface_std"].nanmean(axis=(-1, -2), keepdim=True),
                    lev=spatial_norm_stats["lev_std"].nanmean(axis=(-1, -2), keepdim=True),
                    spatial_forcings=spatial_norm_stats["spatial_forcings_std"].nanmean(
                        axis=(-1, -2), keepdim=True
                    ),
                    non_spatial_forcings=spatial_norm_stats["non_spatial_forcings_std"],
                )
            else:
                self.data_mean = TensorDict(
                    surface=spatial_norm_stats["surface_mean"],
                    level=spatial_norm_stats["level_mean"],
                    lev=spatial_norm_stats["lev_mean"],
                    spatial_forcings=spatial_norm_stats["spatial_forcings_mean"],
                    non_spatial_forcings=spatial_norm_stats["non_spatial_forcings_mean"],
                )
                self.data_std = TensorDict(
                    surface=spatial_norm_stats["surface_std"].nanmean(axis=(-1, -2), keepdim=True),
                    level=spatial_norm_stats["level_std"].nanmean(axis=(-1, -2), keepdim=True),
                    lev=spatial_norm_stats["lev_std"].nanmean(axis=(-1, -2), keepdim=True),
                    spatial_forcings=spatial_norm_stats["spatial_forcings_std"].nanmean(
                        axis=(-1, -2), keepdim=True
                    ),
                    non_spatial_forcings=spatial_norm_stats["non_spatial_forcings_std"],
                )
                # A stats file is always computed for the full surface_variables/
                # level_variables layout it was generated with (see
                # utils/generate_stats.py) -- mirrors netcdf.py's __getitem__
                # slicing the raw 'surface'/'level' tensors to
                # self.variables['surface'/'level']: truncating the loaded
                # mean/std to len(surface_variables)/len(level_variables) here
                # lets a module request a PREFIX subset of an existing stats
                # file's variable order (e.g. surface_variables=['tas'] against
                # an 8-surface-variable stats file whose first variable is
                # 'tas') without regenerating stats. Only correct for a prefix
                # subset in the same order the stats file was generated with --
                # dropping/reordering variables in the middle would silently
                # grab the wrong slice, since this truncates by position, not
                # by variable name (no variable-name metadata is stored in the
                # stats file to look up by name instead).
                if surface_variables is not None:
                    self.data_mean["surface"] = self.data_mean["surface"][: len(surface_variables)]
                    self.data_std["surface"] = self.data_std["surface"][: len(surface_variables)]
                if level_variables is not None:
                    self.data_mean["level"] = self.data_mean["level"][: len(level_variables)]
                    self.data_std["level"] = self.data_std["level"][: len(level_variables)]
            if len(self.spatial_forcing_variables) == 0:
                self.data_mean["spatial_forcings"] = 0
                self.data_std["spatial_forcings"] = 1
            else:
                self.data_mean["spatial_forcings"] = [
                    self.data_mean["spatial_forcings"][
                        self.master_list_spatial_forcing_variables.index(val)
                    ]
                    for i, val in enumerate(self.spatial_forcing_variables)
                    if val in self.master_list_spatial_forcing_variables
                ]

                self.data_std["spatial_forcings"] = [
                    self.data_std["spatial_forcings"][
                        self.master_list_spatial_forcing_variables.index(val)
                    ]
                    for i, val in enumerate(self.spatial_forcing_variables)
                    if val in self.master_list_spatial_forcing_variables
                ]

            self.data_mean["spatial_forcings"] = self.data_mean["spatial_forcings"][:-1]
            self.data_std["spatial_forcings"] = self.data_std["spatial_forcings"][:-1]
            # Drop the trailing exp_id placeholder pad.
            self.data_mean["non_spatial_forcings"] = self.data_mean["non_spatial_forcings"][:-1]
            self.data_std["non_spatial_forcings"] = self.data_std["non_spatial_forcings"][:-1]
        if self.norm_scheme != "zeroes":
            self.data_std["non_spatial_forcings"] = torch.tensor(
                [
                    self.data_std["non_spatial_forcings"][
                        self.master_list_non_spatial_forcing_variables.index(val)
                    ]
                    for i, val in enumerate(self.non_spatial_forcing_variables)
                    if val in self.master_list_non_spatial_forcing_variables
                ]
            )
            self.data_mean["non_spatial_forcings"] = torch.tensor(
                [
                    self.data_mean["non_spatial_forcings"][
                        self.master_list_non_spatial_forcing_variables.index(val)
                    ]
                    for i, val in enumerate(self.non_spatial_forcing_variables)
                    if val in self.master_list_non_spatial_forcing_variables
                ]
            )
        self.surface_variables = surface_variables
        times_seconds = [_epoch_seconds(v[2]) for k, v in self.id2pt.items()]
        self.next_timestamp_map = {k: v for k, v in list(zip(times_seconds, times_seconds[1:]))}

        # override netcdf functionality
        self.timestamps = sorted(self.timestamps, key=lambda x: (x[0], x[1]))  # sort by timestamp
        self.orography = torch.load(f"{forcings_path}/orography.pt")[
            None
        ]  # add dim for stacking later

        self.orography = ((self.orography - self.orography.mean()) / self.orography.std()).float()

    def convert_to_tensordict(self, xr_dataset):
        """Input xarr should be a single time slice."""
        tdict = super().convert_to_tensordict(xr_dataset)
        tdict["surface"] = tdict["surface"].unsqueeze(-3)
        return tdict

    def __len__(self):
        # Take into account previous and/or future timestamps loaded for one example.
        # Convert lead times (months) to index steps using timedelta
        ref_step = min(self.lead_times) // self.timedelta
        max_future_step = max(self.lead_times) // self.timedelta
        offset_steps = max_future_step + (self.load_prev * ref_step)
        return super().__len__() - (offset_steps + self.multistep)

    def __getitem__(self, i, normalize=True, state_only=False, skip_pf=None):
        """Return one dataset item.

        Args:
        i: Dataset index of the item to return.
        normalize: Whether to normalize state/next_state/prev_state (and,
            when built, pf_future_states) before returning.
        state_only: Skip building next_state/prev_state/pf_future_states
            (and their memmap reads/clones) -- only "state" and
            "timestamp" are populated. The processing below (nan
            masking, normalize, orography) already only
            touches keys present in `out`, so this alone drops the
            ~11x-per-call read/clone/normalize overhead. Used by
            inference rollouts (forward_multistep), which only need
            state's spatial_forcings/non_spatial_forcings.
        skip_pf: Skip building pf_future_states/has_pf (8 extra memmap
            reads/clones) while still returning next_state/prev_state.
            Use for callers that need a real next_state (e.g. one-step
            eval/delta stats) but never consume pushforward targets,
            which are only read by training_step. Defaults to
            self.skip_pf (set at construction time) so the standard
            DataLoader indexing path (`dataset[idx]`, which can't pass
            extra kwargs) also benefits for modules -- e.g. DCPPDiffusion
            -- that never consume pushforward targets.
        """
        if skip_pf is None:
            skip_pf = self.skip_pf

        # Shift index if previous state is requested. Use reference step (smallest lead).
        ref_step = min(self.lead_times) // self.timedelta
        i = i + self.load_prev * ref_step

        out = TensorDict()

        out["state"] = super().__getitem__(i).clone()

        out["timestamp"] = torch.tensor(
            _epoch_seconds(self.id2pt[i][2]),
            dtype=torch.int64,
        )

        if not state_only:
            # Choose a single lead time at random from the configured lead_times.
            lead_steps = [lt // self.timedelta for lt in self.lead_times]
            if self.return_all_lead_times:
                next_states = []
                for step in lead_steps:
                    next_states.append(super().__getitem__(i + step).clone())

                out["next_state"] = torch.stack(next_states, dim=0)
                # Expose which lead time (months) was selected for this example
                out["lead_time_months"] = torch.tensor(self.lead_times, dtype=torch.int64)
                out["lead_time"] = torch.tensor(lead_steps, dtype=torch.int64)
            else:
                if len(lead_steps) == 1:
                    chosen_idx = 0
                else:
                    chosen_idx = int(torch.randint(low=0, high=len(lead_steps), size=(1,)).item())
                chosen_step = lead_steps[chosen_idx]
                out["next_state"] = super().__getitem__(i + chosen_step).clone()
                out["lead_time_months"] = torch.tensor(
                    self.lead_times[chosen_idx], dtype=torch.int64
                )
                out["lead_time"] = torch.tensor(chosen_step, dtype=torch.int64)

                if not skip_pf:
                    # Pushforward targets: states at t+2, t+3, ..., t+(pf_n_steps+1).
                    # Always emit the key (with a has_pf flag) so all samples in a
                    # batch have identical keys and TensorDict.stack succeeds.
                    pf_n_steps = self.pf_n_steps
                    max_pf_idx = i + (pf_n_steps + 1) * chosen_step
                    valid_pf = chosen_step == ref_step and max_pf_idx < len(self.id2pt)
                    pf_future = []
                    for k in range(2, pf_n_steps + 2):
                        idx = i + k * chosen_step if valid_pf else i + chosen_step
                        pf_future.append(super().__getitem__(idx).clone())
                    out["pf_future_states"] = torch.stack(pf_future, dim=0)
                    out["has_pf"] = torch.tensor(valid_pf, dtype=torch.bool)

            # Previous state (use reference step)
            out["prev_state"] = super().__getitem__(i - ref_step).clone()

        if not state_only and self.multistep > 1:
            t = self.lead_time_months
            future_states = []
            future_timestamps = []
            for k in range(1, self.multistep + 1):
                idx = i + k * t // self.timedelta
                future_states.append(super().__getitem__(idx).clone())
                future_timestamps.append(_epoch_seconds(self.id2pt[idx][2]))
            out["future_states"] = torch.stack(future_states, dim=0)
            out["future_timestamps"] = torch.tensor(future_timestamps, dtype=torch.int64)

        # Replace inf/large values IN-PLACE instead of cloning
        for k, v in out.items():
            if "state" in k:
                for key, tensor in v.items():
                    # In-place replacement
                    mask = (tensor.abs() > 1e30) | torch.isinf(tensor)
                    tensor[mask] = torch.nan
        # Normalize in-place if possible
        if normalize:
            out = self.normalize(out)

            for k, v in out.items():
                if "state" in k:
                    for key, value in v.items():
                        # Ensure contiguous before in-place operation
                        if not value.is_contiguous():
                            v[key] = value.contiguous()
                            value = v[key]
                        torch.nan_to_num(value, out=value)
        # Handle orography
        for state_key in ["state", "prev_state", "next_state", "future_states", "pf_future_states"]:
            if state_key in out:
                old_spatial = out[state_key]["spatial_forcings"]
                if old_spatial.ndim == 3:
                    out[state_key]["spatial_forcings"] = torch.cat(
                        [old_spatial, self.orography], dim=0
                    ).contiguous()
                elif old_spatial.ndim == 4:
                    # Stacked case (S, C, H, W)
                    S = old_spatial.shape[0]
                    # self.orography is (1, H, W). Expand to (S, 1, H, W)
                    oro_expanded = self.orography.unsqueeze(0).expand(S, -1, -1, -1)
                    out[state_key]["spatial_forcings"] = torch.cat(
                        [old_spatial, oro_expanded], dim=1
                    ).contiguous()
                del old_spatial

        return out

    def normalize(self, batch, stateless=False):
        device = batch["state"].device

        if not hasattr(self, "_cached_means") or self._cached_means.device != device:
            self._cached_means = self.data_mean.to(device, non_blocking=True)
            self._cached_stds = self.data_std.to(device, non_blocking=True)

        means = self._cached_means
        stds = self._cached_stds

        if stateless:
            # Return normalized batch without modifying original
            return (batch - means) / stds
        else:
            # Normalize in-place where possible
            for k, v in batch.items():
                if "state" in k:
                    # Method 1: In-place operations (if v allows it)
                    if isinstance(v, dict) or isinstance(v, TensorDict):
                        for inner_k, inner_v in v.items():
                            # Compute normalized value
                            mean_key = means.get(inner_k, 0)
                            std_key = stds.get(inner_k, 1)

                            # Normalize with minimal intermediate tensors
                            normalized = inner_v.sub(mean_key).div_(std_key)
                            batch[k][inner_k] = normalized
                    else:
                        # Simple tensor case
                        normalized = v.sub(means).div_(stds)
                        batch[k] = normalized

            return batch

    def denormalize(self, batch, stateless=False):
        if stateless:
            device = batch.device
        else:
            device = list(batch.values())[0].device

        means = self.data_mean.to(device)
        stds = self.data_std.to(device)

        skip_keys = {"non_spatial_forcings", "spatial_forcings"}

        if stateless:
            result = {}
            for k, v in batch.items():
                if k not in skip_keys:
                    result[k] = (v * stds[k]) + means[k]
                else:
                    result[k] = v
            return result
        else:
            result = {}
            for k, v in batch.items():
                if "state" in k:
                    result[k] = {
                        kk: ((vv * stds[kk]) + means[kk]) if kk not in skip_keys else vv
                        for kk, vv in v.items()
                    }
                else:
                    result[k] = v
            return result
