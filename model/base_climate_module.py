import importlib.resources
from typing import Any

import lovely_tensors as lt
import torch
import torch.nn as nn
from hydra.utils import instantiate
from tensordict.tensordict import TensorDict

from ArchesClimate import stats as ArchesClimate_stats
from ArchesClimate.backbones.archesweather import SpatialForcingProjector
from ArchesClimate.backbones.dit import CO2LinearGain, TimestepEmbedder
from ArchesClimate.metrics.ensemble_metrics import EnsembleMetrics
from ArchesClimate.model.base_module import BaseLightningModule
from ArchesClimate.utils.tensordict_utils import tensordict_cat

lt.monkey_patch()
stats_resource = importlib.resources.files(ArchesClimate_stats)


class ForcingDropout(nn.Module):
    """Per-forcing independent dropout for SPADE conditioning.

    Owns the full four-way split so the probabilities are explicit and sum to 1:

        uncond_prob       → all forcings replaced by the learned "absent" token
                             (null_spatial_map / null_ssi_token) -- the model
                             still receives a specific, learned signal here,
                             not "no forcing information at all". This is
                             what every existing uncond_* run trains.
        true_uncond_prob  → all forcings replaced by a literal, non-learned
                             zero instead of the null token -- the closest
                             approximation to "no forcing conditioning
                             whatsoever" this architecture can express. Not
                             a mathematical guarantee of zero downstream
                             influence: in "both"/"spade" mode the zeroed
                             channels still pass through
                             spatial_forcing_projector, whose bias term can
                             inject a small nonzero contribution even from an
                             all-zero input. In "both"/"concat" mode the
                             zeroed channels are still physically
                             concatenated into the encoder's input (a fixed
                             channel count -- there's no way to omit a
                             channel), so "true unconditional" there means
                             "channel present but held at the dataset's
                             normalized zero (its mean), not learned".
        all_present_prob  → all forcings kept
        1 - all of these  → per-forcing independent Bernoulli at drop_p,
                             except channels covered by `groups` (see below),
                             which drop as a single block instead

    During eval the mask is always all-ones (the true_uncond regime only
    ever fires during training's random draw; nothing here changes rollout
    behaviour -- see zero_spatial_forcing_indices/zero_forcing_source in
    model/forecast.py for the separate inference-time ablation mechanism,
    which still substitutes the null token, not a true zero).

    Args:
        n_forcings: number of forcing channels (spatial + 1 for SSI group).
        drop_p: per-forcing drop probability in the partial-dropout regime.
        uncond_prob: probability of zeroing every forcing via the learned
            null token (replaces uncond_proba for the forcing path).
        true_uncond_prob: probability of zeroing every forcing via a literal
            (non-learned) zero instead of the null token.
        all_present_prob: probability of keeping every forcing.
        never_drop_indices: forcing channel indices (into the same F axis as
            the returned mask) that are always kept, overriding all four
            regimes above -- including the uncond_prob/true_uncond_prob
            all-absent draws.
        groups: list of channel-index lists (into the same F axis as the
            returned mask) that should be dropped as a single unit -- one
            Bernoulli draw per group per sample, applied to every channel in
            that group -- rather than each channel drawing independently.
            Physically-motivated: e.g. all 11 ozone bands or all 6 aerosol
            species are either jointly known or jointly missing in practice,
            so independent per-channel dropout within such a group produces
            partial states (some ozone bands present, others not) that never
            occur at inference. Channels not listed in any group keep the
            default independent per-channel behaviour.

    Returned mask encoding (not a plain 0/1 keep-mask any more): 1.0 = keep
    the real value, 0.0 = substitute the learned null token, -1.0 =
    substitute a literal zero. Callers that only ever compared against 1.0
    (e.g. "was this channel kept") are unaffected; shared_forward_logic is
    the only place that needs to distinguish 0.0 from -1.0.
    """

    def __init__(
        self,
        n_forcings: int,
        drop_p: float = 0.20,
        uncond_prob: float = 0.10,
        true_uncond_prob: float = 0.0,
        all_present_prob: float = 0.40,
        never_drop_indices: list[int] | None = None,
        groups: list[list[int]] | None = None,
    ) -> None:
        super().__init__()
        assert uncond_prob + true_uncond_prob + all_present_prob <= 1.0
        self.uncond_prob = uncond_prob
        self.true_uncond_prob = true_uncond_prob
        self.all_present_prob = all_present_prob
        self.never_drop_indices = list(never_drop_indices) if never_drop_indices else []
        self.groups = [list(g) for g in groups] if groups else []
        self.register_buffer("drop_probs", torch.full((n_forcings,), drop_p, dtype=torch.float32))

    def sample_mask(self, B: int, device: torch.device) -> torch.Tensor:
        """Return keep-mask (B, F).

        1 = keep, 0 = null-token substitute, -1 = true-zero substitute.
        """
        if not self.training:
            return torch.ones(B, len(self.drop_probs), device=device)
        r = torch.rand(B, device=device)
        all_absent = r < self.uncond_prob
        true_uncond = (r >= self.uncond_prob) & (r < self.uncond_prob + self.true_uncond_prob)
        lo = self.uncond_prob + self.true_uncond_prob
        all_present = (r >= lo) & (r < lo + self.all_present_prob)
        mask = (torch.rand(B, len(self.drop_probs), device=device) > self.drop_probs).float()
        for group in self.groups:
            group_drop_p = self.drop_probs[group[0]]
            group_keep = (torch.rand(B, device=device) > group_drop_p).float()
            mask[:, group] = group_keep.unsqueeze(1)
        mask[all_present] = 1.0
        mask[all_absent] = 0.0
        mask[true_uncond] = -1.0
        if self.never_drop_indices:
            mask[:, self.never_drop_indices] = 1.0
        return mask


class EMA:
    """Exponential moving average of all model parameters.

    Uses decay warmup so early shadow weights track fast-moving params rather
    than averaging near-random initialisations: effective_decay = min(decay,
    (1+step)/(10+step)).

    Args:
        parameters: iterable of (name, param) from model.named_parameters().
        decay: target EMA decay (e.g. 0.999).
    """

    def __init__(self, parameters, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {name: p.detach().clone() for name, p in parameters}
        self._backup: dict[str, torch.Tensor] | None = None

    def update(self, parameters, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for name, p in parameters:
                if name in self.shadow:
                    self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def copy_to(self, parameters) -> None:
        for name, p in parameters:
            if name in self.shadow:
                p.data.copy_(self.shadow[name])

    def swap_with(self, parameters) -> None:
        """Toggle model params between training and EMA weights, in-place.

        Call once to swap EMA weights into params (backs training weights up
        to CPU first); call again to restore the backed-up training weights.
        Either way, shadow itself is left untouched -- its own EMA average
        keeps accumulating independently of what validation happens to have
        swapped into params.
        """
        with torch.no_grad():
            if self._backup is None:
                self._backup = {}
                for name, p in parameters:
                    if name in self.shadow:
                        self._backup[name] = p.data.detach().to("cpu", copy=True)
                        p.data.copy_(self.shadow[name])
            else:
                for name, p in parameters:
                    if name in self._backup:
                        p.data.copy_(self._backup[name])
                self._backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.shadow = state_dict


# 2. Get the real system path as a string without using as_file
# Casting to Path then to string strips the "MultiplexedPath" wrapper
# ArchesClimate_stats_path = str(Path(str(stats_resource)).resolve())


class ClimateLightningModule(BaseLightningModule):
    """Base Lightning module for climate forecasting models."""

    def __init__(self) -> None:
        """Initialize ClimateLightningModule."""
        super().__init__()

    def get_cmip_stats(self, cfg: Any) -> Any:
        """Return cmip_stats based on cfg flags.

        Cached to avoid repeated disk loads.

        Args:
            cfg: Configuration object with norm_scheme attribute.

        Returns:
            Loaded CMIP statistics dictionary.
        """
        file_name = f"cmip_stats_{getattr(cfg, 'norm_scheme', None)}.pt"
        path = stats_resource.joinpath(file_name)

        return torch.load(
            path,
            weights_only=False,
        )
        # if getattr(cfg, "norm_scheme", None) == "full_ocean":
        #     return torch.load(  # noqa: E501
        #         f"{ArchesClimate_stats_path}/cmip_stats_full_ocean.pt")
        # elif getattr(cfg, "norm_scheme", None) == "longer_temps":
        #     return torch.load(  # noqa: E501
        #         f"{ArchesClimate_stats_path}/cmip_stats_longer_temp.pt")
        # else:
        #     return torch.load(  # noqa: E501
        #         f"{ArchesClimate_stats_path}/cmip_stats_test_welfords.pt")

    def get_cmip_masks(self, cfg: Any) -> Any:
        """Get masks for surface, level and lev variables.

        Returns a tuple of (surface_mask, level_mask, lev_mask) based on
        NaN values in stats. Level mask is truncated to length of
        level_variables if provided in cfg.

        Args:
            cfg: Configuration object.

        Returns:
            Tuple of (surface_mask, level_mask, lev_mask).
        """
        cmip_stats = self.get_cmip_stats(cfg)

        # TEMP LOGIC
        # if (len(cfg.surface_variables)> 8):
        #     cmip_stats['surface_mean'] = torch.cat(  # noqa: E501
        #         [cmip_stats['surface_mean'],
        #          cmip_stats['surface_mean'][0:1]], dim=0)
        #     cmip_stats['surface_std'] = torch.cat(  # noqa: E501
        #         [cmip_stats['surface_std'],
        #          cmip_stats['surface_std'][0:1]], dim=0)
        #     cmip_stats['level_mean'] = torch.cat(  # noqa: E501
        #         [cmip_stats['level_mean'],
        #          cmip_stats['level_mean'][1:2]], dim=0)
        #     cmip_stats['level_std'] = torch.cat(  # noqa: E501
        #         [cmip_stats['level_std'],
        #          cmip_stats['level_std'][1:2]], dim=0)
        surface_mask = torch.where(cmip_stats["surface_mean"].isnan(), 0, 1)
        if hasattr(cfg, "surface_variables"):
            # Mirrors level_mask's truncation below -- surface_mean/surface_mask
            # are always the full width of whatever stats file norm_scheme
            # points at, which a cfg.surface_variables that's a PREFIX subset of
            # that file's variable order (e.g. ['tas'] against an 8-variable
            # stats file) needs truncated to match. Previously missing (only
            # level had this): a use_energy_score module reusing an existing
            # multi-variable stats file with a reduced surface_variables list
            # got an untruncated (8,...) surface_mask multiplied against a
            # (1,...) actual surface tensor during forward_multistep, erroring
            # with a broadcast-shape mismatch.
            surface_mask = surface_mask[: len(cfg.surface_variables)]
        if len(cfg.level_variables) > 0:
            level_mask = torch.where(cmip_stats["level_mean"].isnan(), 0, 1)
        else:
            level_mask = torch.ones_like(surface_mask)
        lev_mask = torch.where(cmip_stats["lev_mean"].isnan(), 0, 1)

        if hasattr(cfg, "level_variables"):
            level_mask = level_mask[: len(cfg.level_variables)]

        return surface_mask, level_mask, lev_mask

    def prep_model(self) -> None:
        """Initialize backbone, encoder, embedders, and loss weights."""
        # Grid resolution: defaults to the shared IPSL-derived 144x144 grid;
        # override per-config for datasets on a different native grid (e.g.
        # CanESM5's own 64x128, see memmap_filled_in_canesm5).
        self.lat_dim = self.cfg.get("lat_dim", 144)
        self.lon_dim = self.cfg.get("lon_dim", 144)
        lat_coeffs_equi = torch.tensor(
            [
                torch.cos(x)
                for x in torch.arange(-torch.pi / 2 + 1e-2, torch.pi / 2, torch.pi / self.lat_dim)
            ]
        )
        # cos(lat) collapses to ~0 right at the poles, so the unweighted
        # scheme below gives the main training loss (loss_coeffs, used for
        # the primary step everywhere and for every pushforward step in
        # diffusion.py) almost no gradient signal there -- a likely cause of
        # the disproportionate polar bias documented for this model family.
        # polar_loss_min_weight (default 0.0 = old behaviour, unclamped) lets
        # a floor be applied, mirroring the clamp(min=0.2) already used for
        # loss_coeffs_pf below (forecast.py's pushforward-step loss only).
        polar_loss_min_weight = float(self.cfg.module.get("polar_loss_min_weight", 0.0))
        if polar_loss_min_weight > 0.0:
            lat_coeffs_equi = torch.clamp(lat_coeffs_equi, min=polar_loss_min_weight)
        lat_coeffs_equi = lat_coeffs_equi / lat_coeffs_equi.mean()
        self.loss_coeffs = TensorDict(
            surface=lat_coeffs_equi[None, None, None, :, None],
            level=lat_coeffs_equi[None, None, None, :, None],
            lev=lat_coeffs_equi[None, None, None, :, None],
        )

        # polar_loss_pf_min_weight (default 0.2 = old hardcoded behaviour) lets
        # the pushforward-step loss's own polar floor be raised independently
        # of polar_loss_min_weight above, which only affects the primary loss.
        polar_loss_pf_min_weight = float(self.cfg.module.get("polar_loss_pf_min_weight", 0.2))
        lat_coeffs_pf = torch.sqrt(
            torch.clamp(
                torch.cos(
                    torch.arange(-torch.pi / 2 + 1e-2, torch.pi / 2, torch.pi / self.lat_dim)
                ),
                min=polar_loss_pf_min_weight,
            )
        )
        lat_coeffs_pf = lat_coeffs_pf / lat_coeffs_pf.mean()
        self.loss_coeffs_pf = TensorDict(
            surface=lat_coeffs_pf[None, None, None, :, None],
            level=lat_coeffs_pf[None, None, None, :, None],
            lev=lat_coeffs_pf[None, None, None, :, None],
        )

        self.backbone = instantiate(self.cfg.backbone)  # necessary to put it on device
        if (self.cfg.module.get("scheduler", "") == "flow") or (
            self.cfg.module.get("scheduler", "") == "heun"
        ):
            surface_ch = (
                len(self.cfg.surface_variables)
                + len(self.cfg.level_variables) * len(self.cfg.pressure_levels)
                + len(self.cfg.depth_variables) * len(self.cfg.depth_levels)
            ) * 4  # det, prev, noise, current
        else:
            # +1 input-state slot when this (non-flow) model concatenates a
            # frozen deterministic model's own prediction as conditioning
            # (see shared_forward_logic's "det" in conditional_keys branch,
            # tensordict_cat([pred_state, input_state])) -- e.g.
            # EnergyScoreResidual (model/energy_residual.py). The flow/heun
            # branch above always includes a "det" slot already (det, prev,
            # noise, current), so this only matters here.
            conditional_keys = self.conditional.split("+")
            n_input_states = (1 + self.load_prev) + (1 if "det" in conditional_keys else 0)
            surface_ch = (
                len(self.cfg.surface_variables)
                + len(self.cfg.level_variables) * len(self.cfg.pressure_levels)
                + len(self.cfg.depth_variables) * len(self.cfg.depth_levels)
            ) * n_input_states
        num_surface_variables = len(self.cfg.surface_variables)
        num_level_variables = len(self.cfg.level_variables)

        # See n_sf below for why this isn't simply len(spatial_forcing_variables):
        # that list always has one trailing placeholder name beyond the real
        # channel count, always re-filled at runtime by a real orography channel.
        _n_configured_sf = len(self.cfg.get("spatial_forcing_variables", []))
        n_spatial_forcing = _n_configured_sf if _n_configured_sf else 0
        # How spatial forcings reach the model -- see the block below where
        # spatial_forcing_projector / spatial_forcing_1d_embedder are built:
        #  - "both" (default): concatenated as extra encoder input channels
        #    AND patch-embedded into per-token adaLN conditioning (SPADE).
        #    Matches every config that predates this flag.
        #  - "concat": spatial concatenation only, no SPADE token pathway.
        #  - "adaln_1d": no spatial structure at all -- each forcing's spatial
        #    mean is embedded as a scalar (same treatment as SSI) and summed,
        #    broadcast, into adaLN conditioning.
        #  - "spade": SPADE token pathway only, no channel concatenation.
        # Non-spatial forcings (SSI) always use the 1D solar_embedder path
        # below regardless of this flag -- that pathway isn't part of the
        # ablation.
        self.spatial_forcing_mode = self.cfg.module.get("spatial_forcing_mode", "both")
        assert self.spatial_forcing_mode in ("both", "concat", "adaln_1d", "spade"), (
            f"Unknown spatial_forcing_mode {self.spatial_forcing_mode!r}, "
            "expected 'both', 'concat', 'adaln_1d' or 'spade'"
        )
        encoder_spatial_forcing_ch = (
            n_spatial_forcing if self.spatial_forcing_mode in ("both", "concat") else 0
        )
        self.encoder = instantiate(
            self.cfg.embedder,
            surface_ch=surface_ch,
            level_ch=len(self.cfg.pressure_levels),
            img_size=[surface_ch, self.lat_dim, self.lon_dim],
            forcing_ch=n_spatial_forcing,
            ocean_depth_channels=len(self.cfg.depth_levels),
            surface_variables=num_surface_variables,
            level_variables=num_level_variables,
            load_prev=self.load_prev,
            spatial_forcing_ch=encoder_spatial_forcing_ch,
            is_flow=(
                (self.cfg.module.get("scheduler", 0) == "flow")
                or (self.cfg.module.get("scheduler", 0) == "heun")
            ),
        )
        self.val_metrics = [
            EnsembleMetrics(data_shape=(1, len(self.cfg.surface_variables)))
        ]  # only one timestep
        self.surface_mask, self.level_mask, self.lev_mask = self.get_cmip_masks(self.cfg)
        # Normalization stats needed for physical-unit rollout metrics.
        # Non-persistent: recomputed from config on load, not saved to checkpoint.
        _stats = self.get_cmip_stats(self.cfg)
        self.register_buffer("surface_mean_buf", _stats["surface_mean"].float(), persistent=False)
        self.register_buffer(
            "surface_std_buf",
            _stats["surface_std"].float().clamp(min=1e-8),
            persistent=False,
        )
        # cond_dim should be given as arg to the backbone
        self.forcing_embedder = TimestepEmbedder(self.cond_dim)
        if self.cfg.module.get("scheduler", "") == "flow" or (
            self.cfg.module.get("scheduler", "") == "heun"
        ):
            self.noise_timestep_embedder = TimestepEmbedder(self.cond_dim)

        # CO2 linear-gain pathway (see CO2LinearGain / CondBasicLayer.co2_pattern
        # and .co2_spade_proj): an isolated, unbounded-linear route for CO2's
        # forcing, kept out of the shared SpatialForcingProjector +
        # adaLN_modulation pathway so it can't inherit that pathway's OOD
        # saturation. Opt-in via module.co2_linear_gain, since it needs
        # "carbon" excluded from the main projector's input channels (shrinks
        # forcing_ch by 1) and existing checkpoints don't have these new
        # weights.
        #
        # module.co2_gain_mode picks how the bypass term is computed:
        #  - "scalar" (default): CO2LinearGain maps the spatial mean of the
        #    CO2 field to one scalar per batch element, broadcast through a
        #    fixed learned per-token pattern (CondBasicLayer.co2_pattern).
        #    Cheap, but can't represent real spatial structure in the CO2
        #    field.
        #  - "spade": a dedicated single-channel SpatialForcingProjector
        #    patch-embeds the actual CO2 field to a per-token embedding
        #    (CondBasicLayer.co2_spade_proj consumes it), so spatial
        #    variation in CO2 forcing can reach the bypass term. Kept
        #    alongside "scalar" (not replacing it) so the two can be compared
        #    directly on otherwise-identical configs.
        self.co2_linear_gain_enabled = bool(
            self.cfg.module.get("co2_linear_gain", False)
        ) and "carbon" in (self.cfg.get("spatial_forcing_variables", None) or [])
        self.co2_gain_mode = self.cfg.module.get("co2_gain_mode", "scalar")
        if self.co2_linear_gain_enabled:
            assert self.co2_gain_mode in ("scalar", "spade"), (
                f"Unknown co2_gain_mode {self.co2_gain_mode!r}, expected 'scalar' or 'spade'"
            )
            self.co2_forcing_idx = self.cfg.spatial_forcing_variables.index("carbon")
            if self.co2_gain_mode == "scalar":
                self.co2_linear_gain = CO2LinearGain()
            else:
                patch_size = self.cfg.embedder.patch_size[-1]
                self.co2_spade_projector = SpatialForcingProjector(
                    forcing_ch=1,
                    cond_dim=self.cond_dim,
                    patch_size=patch_size,
                )

        # SPADE: patch-embed spatial forcings to per-token conditioning
        if self.cfg.get("spatial_forcing_variables", None):
            patch_size = self.cfg.embedder.patch_size[-1]  # spatial patch size
            # n_spatial_forcing (computed above, see its comment) is the real
            # runtime channel count -- null_spatial_map/forcing_dropout must
            # match it, not len(spatial_forcing_variables) directly, or this
            # desyncs from the real tensor (RuntimeError in
            # shared_forward_logic's torch.where).
            n_sf = n_spatial_forcing
            if self.spatial_forcing_mode in ("both", "spade"):
                n_projector_ch = n_sf
                if self.co2_linear_gain_enabled:
                    n_projector_ch -= 1
                self.spatial_forcing_projector = SpatialForcingProjector(
                    forcing_ch=n_projector_ch,
                    cond_dim=self.cond_dim,
                    patch_size=patch_size,
                )
            if self.spatial_forcing_mode == "adaln_1d":
                # Shared scalar embedder across all spatial-forcing channels,
                # mirroring solar_embedder's treatment of SSI below.
                self.spatial_forcing_1d_embedder = TimestepEmbedder(self.cond_dim)
            mc = self.cfg.module
            # n_forcings = spatial forcings + 1 SSI group (last index)
            self.forcing_dropout = ForcingDropout(
                n_forcings=n_sf + 1,
                drop_p=float(mc.get("forcing_drop_p", 0.20)),
                uncond_prob=float(mc.get("uncond_proba", 0.10)),
                true_uncond_prob=float(mc.get("true_uncond_proba", 0.0)),
                all_present_prob=float(mc.get("forcing_all_present_prob", 0.40)),
                never_drop_indices=list(mc.get("forcing_never_drop_indices", None) or []),
                groups=[list(g) for g in (mc.get("forcing_dropout_groups", None) or [])],
            )
            # Learned null values substituted for dropped channels — avoids treating
            # zeros as a signal, since some forcings may legitimately be near-zero.
            self.null_spatial_map = nn.Parameter(torch.zeros(n_sf, 1, 1))

        # SSI and other non-spatial scalars get a shared embedder.
        if self.cfg.get("non_spatial_forcing_variables", None) is not None:
            self.solar_embedder = TimestepEmbedder(self.cond_dim)
            self.null_ssi_token = nn.Parameter(torch.zeros(self.cond_dim))

        # Assuming a reasonable max lead time (e.g., 24 months).
        # Adjust the embedding size (25) as needed for your max lead time.
        self.lead_time_embedding = TimestepEmbedder(self.cond_dim)

        # Energy-score ensemble noise: two interchangeable mechanisms, picked
        # by ForecastModule.energy_score_noise_mode --
        #  - "cond_token" (default, matches every checkpoint trained before
        #    this toggle existed): per-member Gaussian noise projected through
        #    a learned embedder and added to cond_tokens (token-wise
        #    conditioning). Needs energy_score_noise_embedder's weights.
        #  - "perturbed_ic": no learned params here -- Gaussian noise is added
        #    directly to the input state's surface/level/lev tensors by the
        #    caller (ForecastModule._perturbed_state_batch /
        #    _make_energy_score_members) before this method ever runs,
        #    matching ocean_model/ArchesClimate's _make_crps_members. Only
        #    meaningful for a model actually trained in this mode.
        if (
            getattr(self, "use_energy_score", False)
            and getattr(self, "energy_score_noise_mode", "cond_token") == "cond_token"
        ):
            self.energy_score_noise_embedder = nn.Sequential(
                nn.Linear(self.energy_score_noise_dim, self.cond_dim),
                nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim),
            )

    def shared_forward_logic(
        self,
        batch,
        noisy_next_state=None,
        use_condition=False,
        timesteps=None,
        forcing_mask=None,
        energy_score_noise=None,
    ) -> TensorDict:
        """Run shared encoder-backbone-decoder forward logic.

        Args:
            batch: Input batch containing state and forcings.
            noisy_next_state: Noisy next state for flow matching.
            use_condition: Whether to apply conditioning mask.
            timesteps: Noise timesteps for flow matching.
            forcing_mask: (B, F) keep-mask from ForcingGroupDropout.sample_mask
                (1 = keep, 0/-1 = substitute); passed through to the forcing
                projector/embedder to null out or zero dropped channels.
            energy_score_noise: (B, energy_score_noise_dim) explicit noise
                draw for the energy-score conditioning term, used at eval time
                (e.g. seeded ensemble rollouts) when
                energy_score_noise_mode == "cond_token". Ignored while
                self.training (which always draws its own fresh noise) and
                when energy_score_noise_mode == "perturbed_ic" (noise is
                already baked into batch["state"] by the caller in that mode,
                not passed here).

        Returns:
            Output TensorDict of predicted next state.
        """
        device = batch["state"].device
        conditional_keys = self.conditional.split("+")

        B = batch["state"]["non_spatial_forcings"].shape[0]
        # Token count from backbone tensor_size: Pl * Lat_p * Lon_p
        ts = self.cfg.backbone.tensor_size
        N = ts[0] * ts[1] * ts[2]

        # Token-wise conditioning (B, N, cond_dim) — starts at zero
        cond_tokens = torch.zeros(B, N, self.cond_dim, device=device)

        # add noisy next state as conditioning if flow matching
        if noisy_next_state is not None:
            input_state = noisy_next_state.clone()
            # Concatenate and clean up
            if self.learn_residual == "pred":
                prev_state = batch["prev_state"]
                input_state = tensordict_cat([prev_state, input_state], dim=1)
        else:
            input_state = batch["prev_state"]

        # Handle multiple previous states
        if self.load_prev > 1:
            prev_states = [batch["prev_state"][:, i] for i in range(self.load_prev)]
            input_state = tensordict_cat(prev_states, dim=1)
            del prev_states

        if "lead_time" not in batch:
            raise ValueError("Batch must contain 'lead_time'")
        lead_time_emb = self.lead_time_embedding(batch["lead_time"])  # (B, D)
        cond_tokens = cond_tokens + lead_time_emb.unsqueeze(1)  # broadcast to all tokens

        # Add deterministic prediction
        if "det" in conditional_keys:
            assert "pred_state" in batch
            pred_state = batch["pred_state"]
            input_state = tensordict_cat([pred_state, input_state], dim=1)

        # Handle conditioning mask
        if type(use_condition) is bool:
            torch.tensor(float(use_condition), device=device)
        else:
            # per-sample mask: broadcast over (N, D) dims
            torch.tensor(use_condition, dtype=torch.float32, device=device)[:, None, None]

        # SPADE: embed spatial and non-spatial forcings into token-wise conditioning.
        # A single unified mask handles all three regimes (uncond / all-present /
        # per-forcing dropout) so the probabilities are explicit and sum to 1.
        masked_sf = None  # spatial forcings after dropout masking, reused for encoder input
        co2_strength = None  # unbounded CO2 forcing-strength scalar, "scalar" mode
        co2_spatial_emb = None  # per-token CO2 projection, "spade" mode
        if "forcings" in conditional_keys and hasattr(self, "forcing_dropout"):
            # Use provided mask (e.g. held across a multi-step rollout) or sample fresh.
            if forcing_mask is None:
                forcing_mask = self.forcing_dropout.sample_mask(B, device)  # (B, F+1)
            sf_mask = forcing_mask[:, :-1]  # (B, F)
            ssi_mask = forcing_mask[:, -1:]  # (B, 1)
            # Per-row true_uncond flag: -1.0 only ever appears in the mask via
            # the true_uncond regime (ordinary dropout only ever produces 0.0),
            # and that regime sets every non-never-drop column of a row to
            # -1.0 at once, so "any column is -1.0" identifies the row.
            # Used below to zero every learned-projector *output* for these
            # rows (not just their input), since e.g. spatial_forcing_projector
            # has a bias term that would otherwise leak a small nonzero
            # contribution into cond_tokens even from an all-zero input.
            row_true_uncond = (forcing_mask == -1.0).any(dim=1)  # (B,)

            if self.cfg.get("spatial_forcing_variables", None):
                sf = batch["state"]["spatial_forcings"]  # (B, F, H, W)
                # Dropped channels get one of two substitutes, selected per
                # sample by the mask's sign/value (see ForcingDropout):
                # 0.0 -> learned null token (a distinct learnable "absent"
                # signal per channel, the pre-existing behaviour); -1.0 ->
                # literal zero (no learned contribution at all, the
                # true_uncond regime).
                keep = (sf_mask == 1.0)[:, :, None, None]  # (B, F, 1, 1)
                true_uncond_sf = (sf_mask == -1.0)[:, :, None, None]  # (B, F, 1, 1)
                null = self.null_spatial_map.expand(B, -1, sf.shape[-2], sf.shape[-1])
                zero = torch.zeros_like(null)
                substitute = torch.where(true_uncond_sf, zero, null)
                masked_sf = torch.where(keep, sf, substitute)  # (B, F, H, W)

                if self.spatial_forcing_mode in ("both", "spade"):
                    if self.co2_linear_gain_enabled:
                        # Computed from the dropout/null-substituted value, so a
                        # sample where CO2 is dropped sees the null contribution
                        # here too, not a backdoor to the true value.
                        if self.co2_gain_mode == "scalar":
                            co2_strength = self.co2_linear_gain(
                                masked_sf[:, self.co2_forcing_idx].mean(dim=(-2, -1))
                            )
                            # Zero downstream of the gain layer, not just its
                            # input -- a bias term would otherwise leak a
                            # nonzero gain even from an all-zero CO2 channel.
                            co2_strength = torch.where(
                                row_true_uncond, torch.zeros_like(co2_strength), co2_strength
                            )
                        else:
                            co2_field = masked_sf[
                                :, self.co2_forcing_idx : self.co2_forcing_idx + 1
                            ]
                            co2_spatial_emb = self.co2_spade_projector(
                                co2_field
                            )  # (B, N, cond_dim)
                            co2_spatial_emb = torch.where(
                                row_true_uncond[:, None, None],
                                torch.zeros_like(co2_spatial_emb),
                                co2_spatial_emb,
                            )
                        projector_sf = torch.cat(
                            [
                                masked_sf[:, : self.co2_forcing_idx],
                                masked_sf[:, self.co2_forcing_idx + 1 :],
                            ],
                            dim=1,
                        )
                    else:
                        projector_sf = masked_sf

                    c_spatial = self.spatial_forcing_projector(projector_sf)  # (B, N, cond_dim)
                    # Zero downstream of the projector, not just its input --
                    # see row_true_uncond's comment above.
                    c_spatial = torch.where(
                        row_true_uncond[:, None, None], torch.zeros_like(c_spatial), c_spatial
                    )
                    cond_tokens = cond_tokens + c_spatial

                if self.spatial_forcing_mode == "adaln_1d":
                    # Discard spatial structure entirely: each forcing becomes
                    # one scalar (its spatial mean) per batch element, embedded
                    # through a shared embedder and broadcast to every token --
                    # same global-conditioning treatment as SSI below.
                    sf_mean = masked_sf.mean(dim=(-2, -1))  # (B, F)
                    sf_1d_emb = torch.zeros(B, self.cond_dim, device=device)
                    for i in range(sf_mean.shape[1]):
                        sf_1d_emb = sf_1d_emb + self.spatial_forcing_1d_embedder(sf_mean[:, i])
                    # Zero downstream of the embedder, not just its input --
                    # see row_true_uncond's comment above.
                    sf_1d_emb = torch.where(
                        row_true_uncond[:, None], torch.zeros_like(sf_1d_emb), sf_1d_emb
                    )
                    cond_tokens = cond_tokens + sf_1d_emb.unsqueeze(1)

            if hasattr(self, "solar_embedder"):
                ssi_emb = torch.zeros(B, self.cond_dim, device=device)
                for i in range(batch["state"]["non_spatial_forcings"].shape[1]):
                    ssi_emb = ssi_emb + self.solar_embedder(
                        batch["state"]["non_spatial_forcings"][:, i]
                    )
                # When SSI is dropped, substitute either the learned null token
                # (0.0) or a literal zero (-1.0, true_uncond regime) -- see the
                # spatial-forcings block above for the same distinction.
                keep_ssi = (ssi_mask == 1.0)[:, :, None]  # (B, 1, 1)
                true_uncond_ssi = (ssi_mask == -1.0)[:, :, None]  # (B, 1, 1)
                null_ssi = self.null_ssi_token[None, None, :]  # (1, 1, cond_dim)
                zero_ssi = torch.zeros_like(null_ssi)
                substitute_ssi = torch.where(true_uncond_ssi, zero_ssi, null_ssi)
                cond_tokens = cond_tokens + torch.where(
                    keep_ssi, ssi_emb.unsqueeze(1), substitute_ssi
                )

        # Energy-score ensemble noise conditioning ("cond_token" mode only --
        # see energy_score_noise_mode in prep_model / ForecastModule.__init__.
        # In "perturbed_ic" mode this block is skipped entirely: noise is
        # already baked into batch["state"] by the caller). A fresh N(0, I)
        # draw on every forward() call while training, so calling forward() M
        # times from training_step yields M distinct stochastic members (see
        # ForecastModule._energy_score_loss). Skipped at eval time (val step,
        # rollouts) unless the caller explicitly supplies energy_score_noise
        # (see ForecastModule.forward_multistep's energy_score_seed) -- every
        # existing inference path leaves that None and is unaffected, still
        # returning a single deterministic-like prediction.
        if (
            getattr(self, "use_energy_score", False)
            and getattr(self, "energy_score_noise_mode", "cond_token") == "cond_token"
        ):
            if self.training:
                noise_z = torch.randn(B, self.energy_score_noise_dim, device=device)
                cond_tokens = cond_tokens + self.energy_score_noise_embedder(noise_z).unsqueeze(1)
            elif energy_score_noise is not None:
                cond_tokens = cond_tokens + self.energy_score_noise_embedder(
                    energy_score_noise
                ).unsqueeze(1)

        # Add noise timestep embedding (flow matching) — broadcast to all tokens
        if (self.cfg.module.get("scheduler", "") == "flow") or (
            self.cfg.module.get("scheduler", "") == "heun"
        ):
            timestep_emb = self.noise_timestep_embedder(timesteps)  # (B, D)
            cond_tokens = cond_tokens + timestep_emb.unsqueeze(1)

        # Forward pass — spatial forcings (already dropout-masked) are appended as
        # extra input channels only in "both"/"concat" modes; the encoder's
        # surface_proj conv was sized accordingly in prep_model (see
        # encoder_spatial_forcing_ch), so this must stay in sync with that flag.
        sf_for_encoder = masked_sf if self.spatial_forcing_mode in ("both", "concat") else None
        x = self.encoder.encode(batch["state"], input_state, spatial_forcings=sf_for_encoder)
        x = self.backbone(
            x, cond_tokens, co2_strength=co2_strength, co2_spatial_emb=co2_spatial_emb
        )
        out = self.encoder.decode(x)
        # Add residual connection
        if self.add_input_state == "default":
            # In-place addition if possible
            for key in out.keys():
                if key in batch["state"]:
                    out[key] = out[key] + batch["state"][key]

        # Clean up
        del x, cond_tokens, input_state

        return out
