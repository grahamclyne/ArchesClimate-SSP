import hashlib
import os
from pathlib import Path
from typing import Any

import lovely_tensors as lt
import numpy as np
import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

lt.monkey_patch()
from ArchesClimate.model.base_climate_module import (  # noqa: E402
    EMA,
    ClimateLightningModule,
)

# Clamp floors for the energy-score/variogram RMSE building blocks -- see
# _state_rmse / _patch_state_rmse / _variogram_score's docstrings. Both used
# to be 1e-12, which stops NaN/inf but leaves d(sqrt(x))/dx or d(x^0.5)/dx
# at ~5e5 right at the floor -- a spurious gradient spike many orders larger
# than a normal RMSE gradient, which the single shared gradient_clip_val=1
# norm clip (main_hydra.py) then lets dominate over every other loss term.
_RMSE_EPS = 1e-4  # squared-error scale (_state_rmse, _patch_state_rmse)
_VARIOGRAM_EPS = 1e-3  # increment scale (_variogram_score, before pow(power))


def _datetime_ymdh(t) -> tuple:
    """(year, month, day, hour) for either np.datetime64 or cftime.datetime.

    Lets timestamps from different calendars (e.g. CanESM5's noleap calendar
    vs. a standard-calendar dataset) be compared on a common basis.
    """
    if isinstance(t, np.datetime64):
        t = t.astype("datetime64[s]").item()
    return (t.year, t.month, t.day, t.hour)


def _select_spatial_forcings_channels(
    spatial: torch.Tensor,
    variables: list,
    ozone_bands: torch.Tensor | None = None,
    base_channels: int | None = None,
) -> torch.Tensor:
    """Reduce a raw (channel-first) spatial_forcings tensor to the channel subset/order.

    The channel subset/order a model with this `spatial_forcing_variables`
    list actually sees at runtime.

    Mirrors dataloaders/netcdf.py's XarrayDataset.__getitem__ exactly (the
    raw memmap tensor is a fixed master-list superset -- e.g. GHG + all
    aerosol species + full ozone-band range -- while any given model's
    declared spatial_forcing_variables is a subset/reordering of that).
    Order matters, per that function's own comment: ozone-tail truncation
    first, then the memmap_filled_in_full_ozone 10-band ozone.memmap append
    (ozone_bands, mirroring netcdf.py's `elif 'ozone' in data.keys()` branch
    -- skipped via base_channels dedup if this realization's raw tensor
    already has the bands baked in, same as netcdf.py's
    _already_has_ozone_bands check), then aerosol (middle 6 channels)
    removal, then GHG (leading 3) removal, each conditioned on whether the
    model's variable list still includes that group at all.
    """
    if "ozone_0" not in variables:
        spatial = spatial[:8]
    elif "ozone_1" not in variables:
        spatial = spatial[:9]
    elif "ozone_7" not in variables:
        spatial = spatial[:15]
    if ozone_bands is not None:
        already_has_ozone_bands = base_channels is not None and spatial.shape[0] > base_channels
        if not already_has_ozone_bands:
            spatial = torch.cat([spatial, ozone_bands], dim=0)
    if "load_ASNO3M" not in variables:
        spatial = torch.cat([spatial[:3], spatial[9:]], dim=0)
    if "methane" not in variables:
        spatial = spatial[3:]
    return spatial


def _load_pi_climatology_raw(
    memmap_dir: str,
    realization: str,
    experiment: str = "piControl",
    spatial_forcing_variables: list | None = None,
    ozone_band_indices: list | None = None,
    base_channels: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (un-normalized) pre-industrial forcing climatology, cached to disk.

    Defaults to averaging a real piControl run's first 120 months. Not every
    model has piControl processed (CanESM5 doesn't -- see
    sec:canesm5_application); pass experiment="historical" to instead average
    the first 12 months (1850) of the historical record as the
    pre-industrial-level proxy. This is not an approximation invented here --
    it's the exact same substitution preprocessing/common/forcings.py's DAMIP
    branches (hist-GHG/hist-aer/hist-stratO3) already use for "every other
    forcing held at piControl climatology" (they tile historical.pt's
    data[:12], never touching a real piControl file either), so using it here
    keeps 'pi'-ablation semantics consistent with how those training runs were
    actually built, and it works for any model that has a historical run --
    every model does, unlike piControl.

    The first call for a given (realization, experiment) reads a short slice
    of the memmap (~190 MB for 120 months, ~19 MB for 12) and caches the
    resulting per-channel climatology (~1.5 MB) to disk; every subsequent
    call, on this job or any other, just loads that tiny cache file instead
    of touching the memmap at all. This matters under concurrent cluster
    load: even a short slice is a real disk read that can stall for minutes
    when $SCRATCH is contended (observed directly -- a full 3000-month read
    once sat stuck over 8 minutes in kernel disk-wait), whereas the cache
    file is small enough to read instantly regardless.
    """
    n_months = 120 if experiment == "piControl" else 12
    # Cache key includes a hash of the variable list: the raw memmap tensor
    # is a fixed master-list superset shared by every model on this grid,
    # but _select_spatial_forcings_channels reduces it differently per
    # model's own spatial_forcing_variables -- two models must not collide
    # on the same cache file after selection.
    # (a plain hash() would do here logically, but str hashing is
    # per-process randomized by default in Python -- md5 keeps the cache
    # filename stable across processes/jobs so the cache actually hits.)
    # "|ozone_dedup_v4" bumps the key so any pre-existing cache file written
    # before the ozone.memmap-band append below is never read back, only
    # orphaned. v2 fixed a shape mismatch (pi_spatial_raw staying at the
    # memmap's native channel count, e.g. 9, vs the 19-channel runtime
    # stats). v3 attempted to fix a second bug introduced by v2 itself but
    # got the on-disk shape wrong: ds["ozone"] is NOT batch-first -- both
    # memmap_filled_in and memmap_filled_in_canesm5_native store it as plain
    # (T, 66, H, W), no leading size-1 batch dim (confirmed directly against
    # both memmaps' historical files). v3's `[0, :n_months]` therefore took
    # a single arbitrary timestep (index 0 of T) and sliced the *level* axis
    # down to :n_months, then `.mean(dim=0)` averaged over that truncated
    # level subset (not time), leaving a plain (H, W) map that
    # ozone_band_indices (values up to 65) then fancy-indexed on the lat
    # axis instead of levels. In-bounds for IPSL's H=144 so it ran without
    # error, silently producing garbage (a mis-averaged, wrong-axis-indexed
    # ozone climatology, not 10 real pressure-level bands) for every
    # no_ozone pi-ablation to date; on CanESM5-native's H=64 grid the same
    # bug raises IndexError instead (index 65 out of bounds for size 64),
    # which is how this was caught. v4 fixes it for real: slice time first,
    # mean over time, then index levels.
    variables_key = hashlib.md5(
        ("-".join(spatial_forcing_variables or ()) + "|ozone_dedup_v4").encode()
    ).hexdigest()[:8]
    cache_path = f"{memmap_dir}/pi_climatology_{experiment}_{realization}_{variables_key}.pt"
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, weights_only=True)
        return cached["spatial"], cached["non_spatial"]

    ds = TensorDict.load_memmap(f"{memmap_dir}/{realization}_{experiment}_interpolation.memmap")
    # piControl's forcings carry a seasonal cycle (and slight interannual
    # noise) but no long-term drift by construction (it's a control run), so
    # a 10-year climatology is already within ~1-2% of the full-record mean
    # (checked directly: max abs diff between single months 2000 apart is a
    # couple percent of scale for the GHG channels, ~1e-6 for aerosol/ozone).
    # For the historical fallback, n_months=12 matches the DAMIP proxy
    # exactly (a single year's climatology, not averaged across years).
    pi_spatial_raw = ds["spatial_forcings"][:, :n_months].mean(dim=1)  # (raw_F, 144, 144)
    pi_non_spatial_raw = ds["non_spatial_forcings"][:, :n_months].mean(dim=1)  # (F_ns,)

    ozone_bands = None
    if ozone_band_indices is not None and "ozone" in ds.keys():
        # Same 10-band selection netcdf.py's __getitem__ appends at runtime
        # for memmap_filled_in_full_ozone realizations that don't already
        # have it baked into spatial_forcings (dedup handled by
        # _select_spatial_forcings_channels via base_channels).
        ozone_bands = ds["ozone"][:n_months].mean(dim=0)[ozone_band_indices]  # (10, 144, 144)

    if spatial_forcing_variables is not None:
        pi_spatial_raw = _select_spatial_forcings_channels(
            pi_spatial_raw,
            spatial_forcing_variables,
            ozone_bands=ozone_bands,
            base_channels=base_channels,
        )

    # Write atomically (temp file + rename) so a second process racing to
    # populate the same cache can't observe a partially-written file.
    tmp_path = f"{cache_path}.tmp{os.getpid()}"
    torch.save({"spatial": pi_spatial_raw, "non_spatial": pi_non_spatial_raw}, tmp_path)
    os.replace(tmp_path, cache_path)
    return pi_spatial_raw, pi_non_spatial_raw


def load_pi_forcing_values(
    memmap_dir: str, dataset: Any, realization: str = "r1i1p1f1"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load pre-industrial-level forcing values, normalized, for use as a zeroing substitute.

    `zero_spatial_forcing_indices` / `zero_non_spatial_forcing_indices` normally
    substitute a learned "this channel is absent" token (see null_spatial_map /
    the zero-fill for non-spatial forcings below) -- a stand-in for "we don't
    know this forcing's value", not "this forcing's real-world value is zero".
    For a physically meaningful counterfactual (e.g. "CO2 held at its
    pre-industrial level for the whole rollout"), substitute the piControl
    experiment's own forcing values instead (or, if this model has no
    piControl run processed, its historical run's first year -- see
    _load_pi_climatology_raw), normalized the same way the dataset normalizes
    its inputs.

    Args:
        memmap_dir: Directory holding this model's *_interpolation.memmap
            subdirectories (e.g. derived from Path(dataset.files[0]).parent
            -- the same directory regardless of which domain filter the
            dataset itself was built with, since every experiment's memmap
            for one model lives side by side there).
        dataset: Dataset instance providing data_mean/data_std for
            "spatial_forcings" and "non_spatial_forcings".
        realization: Ensemble member to read forcings from.

    Returns:
        (pi_spatial, pi_non_spatial): normalized tensors of shape (F, 144, 144)
        and (F_ns,) respectively, indexable by the same channel indices as
        zero_spatial_forcing_indices / zero_non_spatial_forcing_indices.
    """
    experiment = (
        "piControl"
        if os.path.exists(f"{memmap_dir}/{realization}_piControl_interpolation.memmap")
        else "historical"
    )
    pi_spatial_raw, pi_non_spatial_raw = _load_pi_climatology_raw(
        memmap_dir,
        realization,
        experiment=experiment,
        spatial_forcing_variables=dataset.spatial_forcing_variables,
        ozone_band_indices=getattr(dataset, "_ozone_band_indices", None),
        base_channels=getattr(dataset, "_base_spatial_forcing_channels", None),
    )

    sf_mean = dataset.data_mean["spatial_forcings"]
    sf_std = dataset.data_std["spatial_forcings"]
    nsf_mean = dataset.data_mean["non_spatial_forcings"]
    nsf_std = dataset.data_std["non_spatial_forcings"]

    # pi_non_spatial_raw comes back at the raw memmap's fixed master-list
    # width (e.g. 6 ssi/exp_id channels), same as pi_spatial_raw before
    # _select_spatial_forcings_channels reduces it -- but unlike the spatial
    # case, nothing above has reduced it to this model's own
    # non_spatial_forcing_variables subset yet. nsf_mean/nsf_std, by
    # contrast, are already sized to that subset (dataset.__init__ builds
    # them by indexing master_list_non_spatial_forcing_variables per name in
    # dataset.non_spatial_forcing_variables). Mirror that same by-name
    # selection here so the two tensors' widths agree -- most models declare
    # non_spatial_forcing_variables: [], so this is usually an empty select.
    master_list_ns = getattr(dataset, "master_list_non_spatial_forcing_variables", None)
    if master_list_ns is not None:
        keep = [
            master_list_ns.index(v)
            for v in dataset.non_spatial_forcing_variables
            if v in master_list_ns
        ]
        pi_non_spatial_raw = pi_non_spatial_raw[keep]

    pi_spatial = (pi_spatial_raw - sf_mean) / sf_std
    pi_non_spatial = (pi_non_spatial_raw - nsf_mean) / nsf_std

    # data_mean/data_std["spatial_forcings"] are trimmed by one channel
    # relative to what the dataset actually serves per-batch (a trailing
    # placeholder channel with no real forcing data or normalization stats --
    # see the "[:-1]" trim in dataloaders/cmip_random_lead_time.py). That
    # placeholder is always re-filled at runtime with a real orography
    # channel concatenated back onto spatial_forcings (see
    # base_climate_module.py's n_sf, which null_spatial_map is sized off).
    # Pad with a zero row so pi_spatial is indexable by the same channel
    # indices as the runtime spatial_forcings tensor.
    n_runtime = len(dataset.spatial_forcing_variables)
    if pi_spatial.shape[0] < n_runtime:
        pad = torch.zeros(
            n_runtime - pi_spatial.shape[0], *pi_spatial.shape[1:], dtype=pi_spatial.dtype
        )
        pi_spatial = torch.cat([pi_spatial, pad], dim=0)

    return pi_spatial, pi_non_spatial


class ForecastModule(ClimateLightningModule):
    """Climate forecast module with spectral and gradient losses."""

    def __init__(
        self,
        cfg: Any | None = None,
        name: str = "forecast",
        dataset: Any | None = None,
        cond_dim: int = 512,
        pow: int = 2,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.98),
        weight_decay: float = 1e-5,
        num_warmup_steps: int = 1000,
        num_training_steps: int = 300000,
        num_cycles: float = 0.5,
        lat_dimension: int = 144,
        load_prev: int = 1,
        add_input_state: str = "",
        conditional: str = "",
        uncond_proba: float = 0.0,
        optimizer_name: str = "adamw",
        muon_lr: float | None = None,
        adamw_lr: float | None = None,
        lambda_pf: float = 0.0,
        pf_warmup_steps: int = 0,
        pf_n_steps: int = 4,
        ema_decay: float = 0.0,
        use_energy_score: bool = False,
        n_energy_score_members: int = 3,
        pf_n_energy_score_members: int | None = None,
        pf_temporal_energy_score: bool = False,
        energy_score_noise_mode: str = "cond_token",
        energy_score_noise_dim: int = 32,
        energy_score_noise_std: float = 0.05,
        energy_score_global_weight: float = 1.0,
        patch_energy_weight: float = 0.0,
        patch_energy_size: int = 16,
        per_variable_weight: float = 0.0,
        variogram_weight: float = 0.0,
        variogram_lags: tuple[int, ...] = (1, 2, 4),
        variogram_power: float = 0.5,
        variogram_auto_scale: bool = False,
        variogram_anneal_steps: int = 0,
        variogram_auto_scale_ema_decay: float = 0.0,
        noise_embedder_warmup_steps: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize ForecastModule.

        Args:
            cfg: Configuration object.
            name: Name of the model.
            dataset: Dataset object.
            cond_dim: Conditioning dimension.
            pow: Power for loss computation (2 for MSE).
            use_energy_score: If True, replace the MSE training loss with the
                energy score (multivariate RMSE over the joint state vector,
                i.e. sqrt of the lat-weighted mean squared error across all
                variables/levels). For a single deterministic prediction the
                energy score equals E[||X-y||], which normalises gradients by
                the current error magnitude and can improve calibration.
                Also applied to the pushforward loss when lambda_pf > 0.
            n_energy_score_members: Number of stochastic forward passes drawn
                per training example when use_energy_score is True, for the
                main one-step loss. Each member is sampled with independent
                noise (mechanism picked by energy_score_noise_mode below),
                and the loss becomes the proper (fair/unbiased) energy score
                over these M members -- mean member-to-target error minus
                half the mean pairwise member-to-member spread -- rather
                than the M=1 degenerate case (plain RMSE, no dispersion
                term). Ignored when use_energy_score is False. Multiplies
                training compute by roughly M relative to a single forward
                pass per step.
            pf_n_energy_score_members: Same, but for each pushforward step
                (lambda_pf > 0). Defaults to n_energy_score_members when not
                set. Worth lowering independently of n_energy_score_members:
                the pf loop already multiplies cost by pf_n_steps, so its
                member count compounds with that -- e.g. pf_n_steps=8 at the
                same M=3 as the main step is 8x the memory of the main step
                alone (confirmed to OOM a single 80GB A100 once pf_warmup_steps
                is reached, even though the main-step-only cost looked fine
                for the many steps before that point).
            pf_temporal_energy_score: False (default) scores the pushforward
                rollout the old way -- at each of the pf_n_steps frames,
                pf_n_energy_score_members members are drawn fresh (each an
                independent one-step sample conditioned on the SAME
                continuing state, not a continuation of that same member's
                own prior step), scored against that frame's target with
                _energy_score_loss, and the pf_n_steps per-frame scores are
                summed -- so skill/spread are pooled over vars+levels+lat+lon
                at each instant, but never across time; two members that
                agree at every frame individually but trace totally
                different trajectories through time score identically to
                two members whose trajectories actually match.
                True: pf_n_energy_score_members INDEPENDENT, self-consistent
                trajectories are rolled out instead (each member's own
                predicted state feeds into that same member's next step),
                stacked into one (B, pf_n_steps, vars, levels, lat, lon)
                tensor per member and per the target, and scored with a
                SINGLE _energy_score_loss call over the whole trajectory --
                _state_rmse (and patch/variogram, if also active) flatten
                every non-batch dim together already, so stacking a time
                axis right after batch makes them pool over time along with
                everything else with no other code changes needed. This is
                what actually measures whether an ensemble member's full
                temporal trajectory (not just its instantaneous state) looks
                like the true trajectory. Same total forward-pass compute as
                the per-step path (pf_n_energy_score_members members x
                pf_n_steps steps either way), but pf_n_energy_score_members
                trajectories' activations must all stay live simultaneously
                for the stacked call (vs. one frame's worth at a time), so
                peak memory is higher -- lower pf_n_energy_score_members
                first if this OOMs where the per-step path didn't.
            energy_score_noise_mode: Which mechanism injects the per-member
                stochasticity --
                  - "cond_token" (default, matches every checkpoint trained
                    before this toggle existed): a Gaussian noise vector is
                    projected through a learned embedder
                    (energy_score_noise_embedder) and added to the token-wise
                    conditioning (cond_tokens) inside shared_forward_logic.
                    Uses energy_score_noise_dim.
                  - "perturbed_ic": no learned params -- Gaussian noise is
                    added directly to the input state's surface/level/lev
                    tensors before the forward pass (_perturbed_state_batch /
                    _make_energy_score_members), matching
                    ocean_model/ArchesClimate's _make_crps_members / rollout
                    noise_std. Uses energy_score_noise_std. A model must
                    actually be trained in this mode for it to be meaningful
                    at eval/rollout time -- switching modes on an
                    already-trained checkpoint is an off-distribution
                    perturbation for whichever mode it wasn't trained with.
                  - "mc_dropout": no learned params either -- stochasticity
                    comes entirely from the backbone's existing MLP
                    nn.Dropout layers (backbone.dropout > 0 required; each of
                    the M training forward() calls already draws an
                    independent mask since dropout is active whenever
                    self.training is True). At eval/rollout time, dropout
                    would normally switch off with the rest of the model
                    (self.eval() in forward_multistep) -- _set_mc_dropout_active
                    keeps just the Dropout submodules in train() so seeded
                    ensemble rollouts stay stochastic. Since a checkpoint
                    trained with dropout=0 (e.g. a plain deterministic model)
                    has no dropout signal to draw on, this mode is only
                    meaningful once dropout has actually been > 0 during
                    training. Unlike "cond_token", adds zero new parameters,
                    so a use_energy_score module using this mode is
                    architecturally identical to the same module with
                    use_energy_score=False -- a strict (trainer.fit
                    ckpt_path=...) warmstart from a plain deterministic
                    checkpoint works directly, no missing/unexpected keys.
            energy_score_noise_dim: Dimensionality of the per-member Gaussian
                noise vector fed through energy_score_noise_embedder. Only
                used when use_energy_score is True and
                energy_score_noise_mode == "cond_token".
            energy_score_noise_std: Standard deviation of the Gaussian noise
                added to the input state's surface/level/lev tensors to build
                each perturbed-IC ensemble member, and the noise magnitude
                used for seeded ensemble rollouts in that mode
                (forward_multistep's energy_score_seed). Only used when
                use_energy_score is True and energy_score_noise_mode ==
                "perturbed_ic".
            energy_score_global_weight: Weight on the ordinary global (whole-grid
                RMSE) energy score term. Defaults to 1.0, matching every
                behavior before this option existed. Set to 0 to drop the
                global term entirely -- e.g. to train on patch_energy_weight
                and/or variogram_weight alone, isolating whether the
                structural terms work as loss signals on their own rather
                than as an addition to the global term. At least one of
                energy_score_global_weight/patch_energy_weight/
                variogram_weight must be nonzero.
            patch_energy_weight: 0 (default) leaves the energy score as a single
                global RMSE over the whole state, same as before this option
                existed. > 0 adds patch_energy_weight * (energy score computed on
                patch_energy_size x patch_energy_size lat/lon tiles) on top of the
                global term -- see _patch_state_rmse for why the global energy
                score alone rewards ensemble members for having the right total
                variance but is nearly blind to whether that variance is arranged
                the way the true field's spatial structure actually is.
            patch_energy_size: Lat/lon tile edge length (pixels) for the patch
                term above. Only used when patch_energy_weight > 0. Must evenly
                divide the grid's lat and lon extent (144 -> e.g. 8, 12, 16, 18,
                24; not validated here, an indivisible size silently truncates a
                remainder strip off the bottom/right edge instead of erroring).
            per_variable_weight: 0 (default) disables the per-variable energy
                score term (see _per_variable_state_rmse) entirely. > 0 adds
                per_variable_weight * (energy score computed as the SUM of
                each physical variable's own RMSE, pooled over that
                variable's own levels/lat/lon but never across variables)
                on top of whichever other terms are also active. Trades
                away this pooled global term's cross-variable coupling
                (one bad variable's error gets diluted/averaged in with
                every other coordinate) for stable, isolated per-variable
                gradients (one bad variable, e.g. wind, can't stall or
                dominate another variable's, e.g. temperature, learning
                signal, since each variable's contribution keeps its own
                natural scale rather than being blended into a shared mean)
                -- see _per_variable_state_rmse's docstring for the full
                tradeoff against _state_rmse/_patch_state_rmse.
            variogram_weight: 0 (default) disables the Variogram Score term (see
                _variogram_score) entirely. > 0 adds variogram_weight *
                _variogram_score(preds, gt) to the energy score -- a second,
                complementary way to penalize spatially-incoherent ensemble
                members, directly comparing spatial increments instead of
                tiling into patches. Computed from the exact same M members
                already drawn for the global energy score -- no extra forward
                passes.
            variogram_lags: Fixed lag set (grid cells) the variogram term is
                evaluated at, applied along each of lat and lon independently
                (exact strided differencing, e.g. y[..., h:] - y[..., :-h] --
                not a toroidal wrap, and not randomly sampled pairs). Only
                used when variogram_weight > 0. (1, 2, 4) x 2 axes = 6 terms
                per field, each an exact O(H*W) sum over every pixel at that
                lag rather than a stochastic subsample -- cheap enough not to
                need sampling, unlike the O(H^2 W^2) all-pairs score this
                approximates.
            variogram_power: Exponent p in the Scheuerer & Hamill formula
                (|pred_i - pred_j|^p - |gt_i - gt_j|^p)^2. Only used when
                variogram_weight > 0. 0.5 is Scheuerer & Hamill's recommendation
                for most atmospheric fields, not validated for this model.
                Deliberately not 2: p=2 is the Fourier dual of the power
                spectrum, so it would only re-say what the spectral
                diagnostics already show instead of adding new signal.
            variogram_auto_scale: If True, variogram_weight is no longer used
                directly as the loss multiplier -- instead, on the first
                training step that reaches _energy_score_loss with
                variogram_weight != 0, the multiplier is computed as
                variogram_weight * (global+patch energy-score contribution so
                far / raw variogram score), so the variogram term's actual
                contribution to the loss starts out equal to the energy
                score's (variogram_weight=1.0 => exact parity at init;
                0.3/3.0 => 0.3x/3x parity), then frozen for the rest of the
                run (stored in the `_variogram_auto_weight` buffer, so it
                survives checkpoint resume). Needed because the two terms
                live on unrelated natural scales -- global energy score is an
                L2 RMSE over the whole state (order ~1-100 depending on
                normalization), the variogram term is a mean of squared
                power-p increment differences (order ~0.1) -- so a
                variogram_weight that "looks like" a normal loss weight (e.g.
                0.1-1.0) can silently be 1-2 orders of magnitude off from
                giving the term any real influence, or from drowning out the
                energy score entirely. Under DDP, the computed weight is
                broadcast from rank 0 to every replica the first time it's
                set, so all ranks train with the identical frozen weight
                despite computing it from different local mini-batches.
                False (default): variogram_weight is used as-is, unscaled.
            variogram_anneal_steps: Curriculum for the variogram term's
                strength, separate from (and applied on top of)
                variogram_auto_scale's calibration above. 0 (default): the
                term is active at its full (auto-scaled or raw) weight from
                step 0, matching every behaviour before this option existed.
                > 0: the calibration in variogram_auto_scale still fires and
                freezes on the true first active step exactly as documented
                above (so it still targets full parity), but the weight
                actually applied to the loss is that frozen value times
                min(self.global_step / variogram_anneal_steps, 1.0) -- i.e.
                the term ramps linearly from 0 up to full strength over the
                first variogram_anneal_steps, instead of hitting a
                random-init model at full strength immediately. Motivated by
                a genuinely fresh (non-resumed) from-scratch run showing zero
                loss improvement over its first 900 steps with the term at
                full auto-scaled strength from step 0, versus otherwise
                identical non-variogram baselines that drop ~3x over a
                comparable window -- the structural term's gradient landscape
                (pairwise increments through pow(variogram_power), steep even
                after clamping) plausibly fights the easy "learn the mean
                state" direction a random-init model would otherwise find
                quickly. Also used with variogram_auto_scale=False (ramps the
                raw variogram_weight instead of a calibrated one).
            variogram_auto_scale_ema_decay: Only used when variogram_auto_scale
                is True. 0 (default): the step-0 calibration freezes forever,
                matching every behaviour before this option existed. In
                (0, 1): after the initial hard calibration, every subsequent
                active step nudges _variogram_auto_weight toward the
                *current* (global+patch energy-score contribution / raw
                variogram score) * variogram_weight target via an EMA with
                this decay (higher = slower/smoother tracking, e.g. 0.999 ~
                averages over the last ~1000 steps). Needed because the raw
                variogram score is not stationary -- it grows as ensemble
                members genuinely start to diverge over training, so a
                target calibrated once against an early (typically small)
                snapshot becomes a stale, oversized multiplier later on and
                can no longer track the term's true relative scale -- a
                plausible driver of the late-training blowups seen in
                gradfix_p13_w050/w025 (variogram loss and the overall
                train_loss spike into the tens/hundreds/thousands well
                before pf_warmup_steps, worse and earlier the higher
                variogram_weight is, consistent with a frozen multiplier
                whose staleness scales with variogram_weight itself). Under
                DDP the target is averaged across ranks (all_reduce) before
                each EMA update, so replicas stay in sync without a fresh
                broadcast-from-rank-0 every step.
            noise_embedder_warmup_steps: While self.global_step is below this
                (absolute step, e.g. matching a warmstart's seeded
                global_step), on_before_optimizer_step drops the gradient of
                every parameter except energy_score_noise_embedder before
                optimizer.step() -- everything else's weights and optimizer
                momentum/state stay untouched (both Muon and AdamW skip
                params with grad is None entirely, including weight decay),
                so it's a true freeze rather than a zeroed-gradient update.
                Meant for warm-starting a use_energy_score module from a
                checkpoint that predates energy_score_noise_embedder: the new
                head trains from a random init while the rest of the model
                holds at its warm-started weights, then everything trains
                together once global_step reaches this threshold. 0 (default)
                disables this -- every param trains from step 0 as normal.
            lr: Learning rate.
            betas: Adam optimizer betas.
            weight_decay: Weight decay coefficient.
            num_warmup_steps: Number of warmup steps.
            num_training_steps: Total number of training steps.
            num_cycles: Number of cosine scheduler cycles.
            lat_dimension: Latitude dimension size.
            load_prev: Number of previous states to load.
            add_input_state: How to add input state.
            conditional: Conditioning configuration.
            uncond_proba: Unconditional training probability.
            optimizer_name: "adamw" or "muon" -- see model/optimizers.py.
            muon_lr: Learning rate for Muon-optimized parameters, when
                optimizer_name == "muon".
            adamw_lr: Learning rate for AdamW-optimized parameters (falls
                back to `lr` when not set).
            lambda_pf: Weight of the pushforward-rollout loss term added to
                the main one-step loss; 0 (default) disables pushforward
                training entirely.
            pf_warmup_steps: Global step at which pushforward training starts
                being applied (0 = from the start).
            pf_n_steps: Number of pushforward rollout steps per training batch
                (capped by however many pf_future_states targets the dataloader
                prepares for this config's pf_n_steps -- see
                cmip_random_lead_time.py). Only the last
                pf_grad_steps of these receive gradient; earlier steps are
                detached, so memory grows slower than steps but still grows.
            ema_decay: Exponential-moving-average decay for shadow weights,
                0 (default) disables EMA.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.__dict__.update(locals())
        assert energy_score_noise_mode in ("cond_token", "perturbed_ic", "mc_dropout"), (
            f"Unknown energy_score_noise_mode {energy_score_noise_mode!r}, "
            "expected 'cond_token', 'perturbed_ic', or 'mc_dropout'"
        )
        # Sentinel < 0 means "not yet computed" -- see variogram_auto_scale.
        # A buffer (not a plain attribute) so the frozen value survives
        # checkpoint save/resume instead of being recomputed on the resumed
        # run's first step (which would use a different, later-in-training
        # base energy-score magnitude).
        self.register_buffer("_variogram_auto_weight", torch.tensor(-1.0))
        self.prep_model()

    def forward(
        self,
        batch: dict[str, TensorDict],
        use_condition,
        *args: Any,
        **kwargs: Any,
    ) -> TensorDict:
        """Forward pass of the model.

        Args:
            batch: Input batch containing state and previous state.
            use_condition: Whether to apply conditioning mask.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            TensorDict of output predictions.
        """
        forcing_mask = kwargs.get("forcing_mask", None)
        energy_score_noise = kwargs.get("energy_score_noise", None)
        return self.shared_forward_logic(
            batch,
            use_condition=use_condition,
            forcing_mask=forcing_mask,
            energy_score_noise=energy_score_noise,
        )

    def _perturbed_state_batch(
        self,
        batch: dict[str, Any],
        std: float,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Clone `batch` with independent Gaussian noise added to the current state.

        This is perturbed-IC ensembling -- the only source of stochasticity for an
        energy-score-trained model, matching ocean_model/ArchesClimate's
        _make_crps_members / forward_multistep noise_std. `generator` lets seeded
        rollouts (forward_multistep's energy_score_seed) reproduce a specific draw;
        left None (default) during training, where each call just needs independent
        noise, not reproducibility.
        """
        if std <= 0:
            return batch
        state = batch["state"].clone()
        for key in ("surface", "level", "lev"):
            t = state[key]
            noise = (
                torch.randn(t.shape, dtype=t.dtype, device=t.device, generator=generator)
                if generator is not None
                else torch.randn_like(t)
            )
            state[key] = t + noise * std
        new_batch = dict(batch)
        new_batch["state"] = state
        return new_batch

    def _make_energy_score_members(
        self,
        batch: dict[str, Any],
        num_members: int,
    ) -> list[dict[str, Any]]:
        """Build `num_members` independently perturbed-IC copies of `batch`.

        For energy-score ensemble training -- see _perturbed_state_batch.
        """
        return [
            self._perturbed_state_batch(batch, self.energy_score_noise_std)
            for _ in range(num_members)
        ]

    def _set_mc_dropout_active(self, active: bool) -> None:
        """Toggle just the Dropout submodules' train()/eval() state.

        Independent of the rest of the model. self.eval() before a rollout normally turns dropout
        off along with everything else; for energy_score_noise_mode == "mc_dropout"
        that would collapse every ensemble member to the same prediction, since
        dropout masks are the only source of per-member variation in that mode.
        """
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(active)

    def _energy_score_member_inputs(
        self,
        loop_batch: dict[str, Any],
        generator: torch.Generator,
        noise_scale: float,
    ) -> tuple[dict[str, Any], torch.Tensor | None]:
        """Per-member (forward_batch, energy_score_noise) for one seeded rollout step.

        Dispatches on self.energy_score_noise_mode. Shared by
        forward_multistep's avg and non-avg branches.
        """
        if self.energy_score_noise_mode == "perturbed_ic":
            return (
                self._perturbed_state_batch(
                    loop_batch,
                    self.energy_score_noise_std,
                    generator=generator,
                ),
                None,
            )
        if self.energy_score_noise_mode == "mc_dropout":
            # nn.Dropout draws from the global RNG, not a passed-in generator --
            # reseed it from our seeded generator so each member stays
            # reproducible under a fixed energy_score_seed.
            seed_val = int(
                torch.randint(
                    0, 2**31 - 1, (1,), generator=generator, device=generator.device
                ).item()
            )
            torch.manual_seed(seed_val)
            return loop_batch, None
        # cond_token
        B = loop_batch["state"]["non_spatial_forcings"].shape[0]
        noise = (
            torch.randn(
                B,
                self.energy_score_noise_dim,
                device=self.device,
                generator=generator,
            )
            * noise_scale
        )
        return loop_batch, noise

    def _state_rmse(
        self,
        a: TensorDict,
        b: TensorDict,
        loss_coeffs: TensorDict,
    ) -> torch.Tensor:
        """Per-batch-element lat-weighted RMSE between two states.

        Treats the full multi-variable state as one vector:
        sqrt(mean_{vars,lev,lat,lon}(w_lat*(a-b)^2)). Masks (surface/level/lev)
        must already have been applied to both a and b before calling this.

        Clamped to a small epsilon before the sqrt: d(sqrt(x))/dx -> inf as
        x -> 0, so an exactly (or near-)identical pair of states -- e.g. two
        energy-score members that haven't yet diverged early in training --
        would otherwise inject an inf/NaN gradient here. The clamp floor is
        _RMSE_EPS (1e-4), not the 1e-12 this used to use: 1e-12 stops the
        NaN but d(sqrt(x))/dx at x=1e-12 is still ~5e5 -- a spurious gradient
        many orders of magnitude larger than a normal RMSE gradient (squared
        errors here are physically O(1e-2)-O(1e2)), which dominates the
        single shared gradient_clip_val=1 norm clip (main_hydra.py) and
        starves every other loss term's gradient whenever members happen to
        be close (common for the spread term, especially early in training
        or when energy_score_noise_embedder's output is small). Confirmed
        directly: pure energy-score runs (no variogram) plateaued on a noisy,
        physically-inconsistent seasonal cycle, and any run adding
        variogram_weight > 0 (many more near-zero increments per step, see
        _variogram_score) went completely flat -- zero improvement in
        es_global_raw for the entire run, vs. a clean ~3x drop with this
        term absent. 1e-4 keeps the same NaN-guard purpose while keeping
        d(sqrt(x))/dx bounded to ~50, comparable to a normal gradient.
        """
        B = a["surface"].shape[0]
        sf_sq = (a["surface"] - b["surface"]).pow(2).mul(loss_coeffs["surface"]).view(B, -1)
        lv_sq = (a["level"] - b["level"]).pow(2).mul(loss_coeffs["level"]).view(B, -1)
        le_sq = (a["lev"] - b["lev"]).pow(2).mul(loss_coeffs["lev"]).view(B, -1)
        all_sq = torch.cat([sf_sq, lv_sq, le_sq], dim=-1)  # (B, N_total)
        return all_sq.mean(-1).clamp_min(_RMSE_EPS).sqrt()  # (B,)

    def _per_variable_state_rmse(
        self,
        a: TensorDict,
        b: TensorDict,
        loss_coeffs: TensorDict,
    ) -> torch.Tensor:
        """Sum of per-variable RMSEs.

        Each physical variable gets its own RMSE (pooled over its own
        levels/lat/lon, same lat-weighting as _state_rmse), then these are
        SUMMED -- not averaged -- across
        variables into one number per batch element. "Variable" here means
        each surface channel independently; each level-variable's full
        pressure-level stack (e.g. all 17 "ta" levels pooled into one RMSE,
        separate from all 17 "ua" levels); the lev/depth variable's full
        depth stack, same way.

        Tradeoff vs the other rmse_fn building blocks _energy_score_loss can
        use: _state_rmse pools every variable/level/lat/lon coordinate into
        one number, so a bad prediction in one variable (e.g. wind) is
        diluted by every other coordinate and can't dominate or stall that
        step's gradient -- but the resulting single scalar also can't
        express "this specific variable is wrong", only "the whole state's
        aggregate error is such-and-such". This function instead keeps each
        variable's own gradient contribution isolated (summed, not blended
        into a shared mean) at the cost of never comparing across
        variables -- the reverse tradeoff again from _patch_state_rmse,
        which keeps lat/lon isolated instead. Cross-variable correlation
        (e.g. does a temperature error co-occur with the right pressure
        error) is not scored by any of the three; only _state_rmse pools
        variables together at all, and even it only pools them into a
        single joint scalar rather than actually measuring their
        covariance.

        Summing (not averaging) means adding more variables raises this
        term's overall magnitude -- deliberate: each variable's own RMSE
        keeps the same natural scale/gradient magnitude regardless of how
        many other variables are being scored alongside it, matching how a
        single _state_rmse-style term behaves for a fixed variable set.
        Averaging would instead shrink every variable's effective gradient
        as more variables are added.
        """

        def per_var_rmse(
            a_t: torch.Tensor, b_t: torch.Tensor, coeffs_t: torch.Tensor
        ) -> torch.Tensor:
            # a_t/b_t: (B, C, L, lat, lon) -- C independent variables, L their
            # own levels (1 for surface). coeffs_t broadcasts over (C, L).
            sq = (a_t - b_t).pow(2).mul(coeffs_t)  # (B, C, L, lat, lon)
            mse = sq.mean(dim=(2, 3, 4))  # (B, C) -- pooled over level/lat/lon only
            return mse.clamp_min(_RMSE_EPS).sqrt()  # (B, C), one RMSE per variable

        sf = per_var_rmse(a["surface"], b["surface"], loss_coeffs["surface"])
        lv = per_var_rmse(a["level"], b["level"], loss_coeffs["level"])
        le = per_var_rmse(a["lev"], b["lev"], loss_coeffs["lev"])
        return torch.cat([sf, lv, le], dim=1).sum(dim=1)  # (B,) -- summed across all variables

    def _patch_state_rmse(
        self,
        a: TensorDict,
        b: TensorDict,
        loss_coeffs: TensorDict,
        patch_size: int,
    ) -> torch.Tensor:
        """Like _state_rmse, but the lat/lon grid is first cut into non-overlapping tiles.

        patch_size x patch_size tiles, and each tile gets its own RMSE,
        instead of one RMSE over the whole ~43M-coordinate state.

        Why: a single global RMSE (energy score's rmse building block) aggregates
        every lat/lon/level/variable coordinate into one number per batch element.
        In that regime the aggregate concentrates around the total variance of the
        difference and is almost blind to how that difference is arranged in space
        -- independent per-pixel noise and a spatially coherent error pattern of the
        same total power score nearly identically, so the energy score's dispersion
        term (_energy_score_loss's `spread`) rewards members for having the right
        total variance but not for having it correlated the way the true field is.
        Splitting into small patches keeps each aggregate low-dimensional enough
        that concentration doesn't wash out structure: two members that agree
        patch-by-patch (spatially coherent, like the true field) score a lower
        patch RMSE against each other than two members with the same total
        variance spread independently pixel-by-pixel. patch_size=1 is the limit
        of this -- every pixel scored on its own (pooling only over
        variables/levels, never over lat/lon at all) -- see
        deterministic_damip_pf4_energy_score_per_pixel.yaml.

        Returns one RMSE per (batch element, patch) instead of one per batch
        element, i.e. shape (B * n_patches,) instead of (B,) -- feed straight into
        the same skill/spread energy-score arithmetic as _state_rmse.
        """

        def patchify(t: torch.Tensor) -> torch.Tensor:
            B = t.shape[0]
            mid_shape = t.shape[1:-2]  # everything between batch and (lat, lon)
            H, W = t.shape[-2], t.shape[-1]
            ph, pw = H // patch_size, W // patch_size
            t = t[..., : ph * patch_size, : pw * patch_size]
            t = t.unfold(-2, patch_size, patch_size).unfold(-2, patch_size, patch_size)
            # -> (B, *mid_shape, ph, pw, patch_size, patch_size)
            t = t.reshape(B, *mid_shape, ph * pw, patch_size, patch_size)
            t = t.movedim(1 + len(mid_shape), 1)  # -> (B, ph*pw, *mid_shape, patch, patch)
            return t.reshape(B * ph * pw, *mid_shape, patch_size, patch_size)

        sf_a, sf_b = patchify(a["surface"]), patchify(b["surface"])
        lv_a, lv_b = patchify(a["level"]), patchify(b["level"])
        le_a, le_b = patchify(a["lev"]), patchify(b["lev"])
        # loss_coeffs is broadcast (e.g. (1,1,1,lat,1), varying only over lat) --
        # expand to the real per-field shape first so unfold sees a real lon extent
        # (its broadcast size-1 lon dim would otherwise be smaller than patch_size).
        coeffs_sf = patchify(loss_coeffs["surface"].expand_as(a["surface"]))
        coeffs_lv = patchify(loss_coeffs["level"].expand_as(a["level"]))
        coeffs_le = patchify(loss_coeffs["lev"].expand_as(a["lev"]))

        Bp = sf_a.shape[0]
        sf_sq = (sf_a - sf_b).pow(2).mul(coeffs_sf).reshape(Bp, -1)
        lv_sq = (lv_a - lv_b).pow(2).mul(coeffs_lv).reshape(Bp, -1)
        le_sq = (le_a - le_b).pow(2).mul(coeffs_le).reshape(Bp, -1)
        all_sq = torch.cat([sf_sq, lv_sq, le_sq], dim=-1)
        return all_sq.mean(-1).clamp_min(_RMSE_EPS).sqrt()  # (B * n_patches,)

    def _variogram_score(
        self,
        preds: list[TensorDict],
        gt: TensorDict,
        lags: tuple[int, ...],
        power: float,
    ) -> torch.Tensor:
        """Fair (bias-corrected) Variogram Score (Scheuerer & Hamill 2015).

        Evaluated at a small fixed set of lat/lon lags rather than sampled
        pairs:

            VS = mean_{field,axis,lag}[ fair_sq(|pred_i - pred_j|^power,
                                                 |gt_i - gt_j|^power) ]

        where (i, j) range over every pixel pair separated by `lag` cells
        along `axis` (exact strided differencing, e.g.
        y[..., h:] - y[..., :-h] -- not a toroidal wrap, and not a random
        subsample: at these small lags the full sum is only O(H*W) per term,
        cheap enough not to need sampling).

        fair_sq is the M-member-unbiased estimator of (E_m[pred_diff] -
        gt_diff)^2 -- the cross-member average (1/(M(M-1))) * sum_{i!=j}
        (pred_diff_i - gt_diff)(pred_diff_j - gt_diff), which at M=2 is just
        (pred_diff_1 - gt_diff)(pred_diff_2 - gt_diff). This matters because
        the naive single-member alternative, mean_m[(pred_diff_m -
        gt_diff)^2], decomposes as Var_m(pred_diff) + (E_m[pred_diff] -
        gt_diff)^2 -- i.e. it silently adds a term that's minimized by
        shrinking pred_diff's spread across members, rewarding an
        artificially tight ensemble regardless of whether that's the right
        spread. The cross-member form has expectation exactly
        (E_m[pred_diff] - gt_diff)^2, no spread leakage, mirroring the same
        fix _energy_score_core already applies to the global/patch RMSE terms
        (skill - 0.5*spread, built from i!=j cross terms) -- variogram just
        never had the analogous correction. M=1 (energy score off/plain
        deterministic) falls back to the naive (biased) form since there's no
        second member to cross against.

        preds: list of M member TensorDicts, each with "surface"/"level"/"lev"
        keys, (lat, lon) as the last two dims (same convention as
        _state_rmse). gt: the corresponding ground-truth TensorDict.
        """
        M = len(preds)
        device = gt["surface"].device
        total = torch.zeros((), device=device)
        n_terms = 0
        for key in ("surface", "level", "lev"):
            g = gt[key]
            if g.numel() == 0:
                continue
            for axis in (-2, -1):
                H_axis = g.shape[axis]
                for lag in lags:
                    if H_axis <= lag:
                        continue

                    def inc(t: torch.Tensor, axis: int = axis, lag: int = lag) -> torch.Tensor:
                        if axis == -2:
                            hi, lo = t[..., lag:, :], t[..., :-lag, :]
                        else:
                            hi, lo = t[..., lag:], t[..., :-lag]
                        # Clamped before pow(power): d(x^power)/dx -> inf as
                        # x -> 0 for power < 1 (e.g. the default power=0.5),
                        # and an exactly-zero increment is common early in
                        # training -- same fix as _state_rmse's clamp before
                        # sqrt. Confirmed via a GPU smoke test: unclamped,
                        # this NaNs literally every parameter's gradient the
                        # first time variogram_weight > 0. Floor is
                        # _VARIOGRAM_EPS (1e-3), not the 1e-12 this used to
                        # use -- 1e-12 stops the NaN but d(x^0.5)/dx at
                        # x=1e-12 is still ~5e5, a spurious gradient spike
                        # that (combined with the single shared
                        # gradient_clip_val=1 norm clip in main_hydra.py)
                        # drowned out every other loss term's gradient.
                        # Confirmed directly: every run with
                        # variogram_weight > 0 showed zero improvement in
                        # es_global_raw for its entire training run (many
                        # near-zero increments per step here vs. only the
                        # spread term in plain energy-score training). 1e-3
                        # matches the natural scale of a real (normalized)
                        # increment while keeping d(x^0.5)/dx bounded to
                        # ~16, comparable to a normal gradient.
                        return (hi - lo).abs().clamp_min(_VARIOGRAM_EPS).pow(power)

                    gt_diff = inc(g)
                    pred_diffs = [inc(preds[m][key]) - gt_diff for m in range(M)]
                    if M == 1:
                        term = pred_diffs[0].pow(2).mean()
                    else:
                        cross_sum = None
                        for i in range(M):
                            for j in range(M):
                                if i == j:
                                    continue
                                t = pred_diffs[i] * pred_diffs[j]
                                cross_sum = t if cross_sum is None else cross_sum + t
                        term = (cross_sum / (M * (M - 1))).mean()
                    total = total + term
                    n_terms += 1
        return total / max(n_terms, 1)

    def _energy_score_core(
        self,
        preds: list[TensorDict],
        gt: TensorDict,
        rmse_fn: Any,
        log_prefix: str | None = None,
    ) -> torch.Tensor:
        """Shared (fair/unbiased) energy-score arithmetic -- skill minus half the member spread.

        Parametrised by which rmse function scores a pair of states.
        _energy_score_loss calls this once with _state_rmse (global) and,
        when patch_energy_weight > 0, again with _patch_state_rmse (local).

        skill (mean member-to-truth rmse) and spread (mean member-to-member
        rmse, the dispersion/calibration term) are logged separately when
        log_prefix is given -- the combined (skill - 0.5*spread) return value
        can sit flat while skill and spread move in opposite directions (e.g.
        during a noise_embedder_warmup_steps freeze, where only spread can
        move at all since skill's underlying weights are frozen), which the
        combined number alone can't distinguish from genuinely no progress.
        M=1 has no spread term (no second member to compare against), so only
        skill is logged in that case.
        """
        M = len(preds)
        skill = sum(rmse_fn(p, gt) for p in preds) / M
        if M == 1:
            if log_prefix:
                self.mylog({f"{log_prefix}_skill": skill.mean().detach()})
            return skill.mean()
        spread_sum = None
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                term = rmse_fn(preds[i], preds[j])
                spread_sum = term if spread_sum is None else spread_sum + term
        spread = spread_sum / (M * (M - 1))
        if log_prefix:
            self.mylog(
                {
                    f"{log_prefix}_skill": skill.mean().detach(),
                    f"{log_prefix}_spread": spread.mean().detach(),
                }
            )
        return (skill - 0.5 * spread).mean()

    def _energy_score_loss(
        self,
        pred: TensorDict | list[TensorDict],
        gt: TensorDict,
        loss_coeffs: TensorDict,
    ) -> torch.Tensor:
        """(Fair/unbiased) energy score over an M-member ensemble.

        ES = mean_batch[ (1/M) sum_m rmse(pred_m, gt)
                          - (1/(2*M*(M-1))) sum_{m != n} rmse(pred_m, pred_n) ]

        where rmse is the lat-weighted full-state-vector RMSE from
        _state_rmse (not a true Euclidean norm of the concatenated vector --
        kept consistent with the rest of this loss). The M*(M-1) (rather than
        M^2) normalisation excludes the zero m==n terms, giving the standard
        unbiased finite-ensemble estimator (as used by e.g. GenCast/NeuralGCM)
        instead of a biased one that shrinks towards zero spread as M grows.

        pred: a single TensorDict (M=1, dispersion term is zero -> plain RMSE,
        the original deterministic behaviour) or a list of M TensorDicts, one
        per independently-sampled stochastic member (see
        ForecastModule.n_energy_score_members / shared_forward_logic's noise
        conditioning). Each must already have the same masking/key-filtering
        applied as gt.

        If patch_energy_weight > 0, adds patch_energy_weight * (the same energy
        score computed on patch_energy_size x patch_energy_size lat/lon tiles
        instead of the whole grid) -- see _patch_state_rmse for why the global
        term alone is blind to spatial/structural correctness.

        If variogram_weight > 0, adds variogram_weight * _variogram_score(preds,
        gt) -- a complementary structural-correctness term using a
        cross-member bias correction (see _variogram_score) so it doesn't
        double as a spread penalty the way a naive per-member average would.
        Can be used together with or instead of patch_energy_weight. Ramped
        in linearly over variogram_anneal_steps rather than applied at full
        strength from step 0 -- see variogram_anneal_steps' docstring.
        """
        preds = [pred] if isinstance(pred, TensorDict) else list(pred)

        # Logged raw (unweighted) so you can compare each term's natural
        # magnitude, and *_weighted so you can see each term's actual
        # contribution to `total` -- the sum of the _weighted values below is
        # exactly the returned loss. mylog prefixes with train_/val_
        # automatically (see BaseLightningModule.mylog) and shows in both the
        # progress bar and the logger (wandb), so this is visible without
        # digging through code -- meant to answer "is patch/variogram getting
        # drowned out by the global term" directly from the numbers, not by
        # guessing from the weights alone (the global term's natural
        # magnitude and the patch/variogram terms' natural magnitudes aren't
        # the same scale, so equal-looking weights don't mean equal
        # contribution).
        global_weight = getattr(self, "energy_score_global_weight", 1.0)
        total = None
        if global_weight:
            global_score = self._energy_score_core(
                preds,
                gt,
                lambda p, q: self._state_rmse(p, q, loss_coeffs),
                log_prefix="es_global",
            )
            global_contrib = global_weight * global_score
            total = global_contrib
            self.mylog(
                {
                    "es_global_raw": global_score.detach(),
                    "es_global_weighted": global_contrib.detach(),
                }
            )
        patch_weight = getattr(self, "patch_energy_weight", 0.0)
        if patch_weight:
            patch_size = getattr(self, "patch_energy_size", 16)
            patch_score = self._energy_score_core(
                preds,
                gt,
                lambda p, q: self._patch_state_rmse(p, q, loss_coeffs, patch_size),
                log_prefix="es_patch",
            )
            patch_contrib = patch_weight * patch_score
            total = patch_contrib if total is None else total + patch_contrib
            self.mylog(
                {
                    "es_patch_raw": patch_score.detach(),
                    "es_patch_weighted": patch_contrib.detach(),
                }
            )
        per_variable_weight = getattr(self, "per_variable_weight", 0.0)
        if per_variable_weight:
            per_variable_score = self._energy_score_core(
                preds,
                gt,
                lambda p, q: self._per_variable_state_rmse(p, q, loss_coeffs),
                log_prefix="es_per_variable",
            )
            per_variable_contrib = per_variable_weight * per_variable_score
            total = per_variable_contrib if total is None else total + per_variable_contrib
            self.mylog(
                {
                    "es_per_variable_raw": per_variable_score.detach(),
                    "es_per_variable_weighted": per_variable_contrib.detach(),
                }
            )
        variogram_weight = getattr(self, "variogram_weight", 0.0)
        if variogram_weight:
            lags = getattr(self, "variogram_lags", (1, 2, 4))
            power = getattr(self, "variogram_power", 0.5)
            variogram = self._variogram_score(preds, gt, lags, power)
            # Floor at 0 rather than leaving this negative (which happens
            # whenever spread > 2*skill, i.e. the ensemble is currently
            # over-dispersed relative to its own error -- a fair/proper
            # scoring rule allows and even rewards this). clamp_min has zero
            # gradient in the clamped region rather than flipping its sign
            # (unlike abs()), so it can only ever stop this term from
            # pushing further in an already-good direction, never actively
            # push the wrong way -- but it does give up the strict
            # properness of the estimator (no penalty for excessive
            # over-dispersion) in exchange for removing negative excursions
            # from the loss.
            variogram = variogram.clamp_min(0.0)
            effective_weight = variogram_weight
            if getattr(self, "variogram_auto_scale", False):
                is_distributed = (
                    torch.distributed.is_available() and torch.distributed.is_initialized()
                )
                if self._variogram_auto_weight.item() < 0:
                    # First step variogram_weight is active: hard-initialize
                    # the multiplier so this term's contribution starts equal
                    # to (variogram_weight x) the energy-score contribution
                    # accumulated so far (global + patch) -- see
                    # variogram_auto_scale's docstring for why the two terms'
                    # natural magnitudes aren't comparable. With
                    # variogram_auto_scale_ema_decay == 0 this value is then
                    # never touched again (the original, pre-EMA behaviour);
                    # otherwise it's just the EMA's starting point below.
                    # variogram is a fair/energy-score-style estimator (mean
                    # member-to-target error minus half the mean pairwise
                    # member spread), so it routinely goes negative when
                    # spread dominates -- clamp_min alone (no abs()) would
                    # then floor a negative denom at ~0, exploding this
                    # ratio. abs() first, and a much less extreme floor than
                    # 1e-12 (raw variogram is O(0.01-1) in practice, so 1e-12
                    # only ever matters as a division-by-exact-zero guard,
                    # not a realistic denominator).
                    base = total if total is not None else torch.zeros((), device=variogram.device)
                    denom = variogram.detach().abs().clamp_min(1e-2)
                    self._variogram_auto_weight.fill_(
                        (base.detach() / denom * variogram_weight).item()
                    )
                    if is_distributed:
                        torch.distributed.broadcast(self._variogram_auto_weight, src=0)
                    self.mylog({"es_variogram_auto_weight": self._variogram_auto_weight.clone()})
                else:
                    ema_decay = getattr(self, "variogram_auto_scale_ema_decay", 0.0)
                    if ema_decay > 0:
                        # Re-track the natural-scale ratio instead of staying
                        # frozen at the step-0 snapshot -- see
                        # variogram_auto_scale_ema_decay's docstring for why
                        # that snapshot goes stale as the raw variogram
                        # score's magnitude drifts over training. Same
                        # abs()+floor fix as the initial calibration above,
                        # for the same reason -- and since this now runs
                        # every active step instead of once, a single
                        # near-zero-denominator step is no longer a rare
                        # unlucky draw, it's a near-certainty over enough
                        # steps, so on top of that we also hard-clamp the
                        # per-step target to a bounded multiple of the
                        # current weight -- a single noisy step can nudge
                        # the EMA but never detonate it.
                        base = (
                            total if total is not None else torch.zeros((), device=variogram.device)
                        )
                        denom = variogram.detach().abs().clamp_min(1e-2)
                        target = base.detach() / denom * variogram_weight
                        if is_distributed:
                            torch.distributed.all_reduce(target, op=torch.distributed.ReduceOp.SUM)
                            target = target / torch.distributed.get_world_size()
                        current = self._variogram_auto_weight.clamp_min(1e-8)
                        target = target.clamp(min=0.1 * current, max=10.0 * current)
                        updated = ema_decay * self._variogram_auto_weight + (1 - ema_decay) * target
                        self._variogram_auto_weight.fill_(updated.item())
                        self.mylog(
                            {"es_variogram_auto_weight": self._variogram_auto_weight.clone()}
                        )
                effective_weight = self._variogram_auto_weight.item()
            anneal_steps = getattr(self, "variogram_anneal_steps", 0)
            ramp = min(self.global_step / anneal_steps, 1.0) if anneal_steps > 0 else 1.0
            effective_weight = effective_weight * ramp
            variogram_contrib = effective_weight * variogram
            total = variogram_contrib if total is None else total + variogram_contrib
            self.mylog(
                {
                    "es_variogram_raw": variogram.detach(),
                    "es_variogram_weighted": variogram_contrib.detach(),
                    "es_variogram_ramp": torch.tensor(ramp, device=variogram.device),
                }
            )
        assert total is not None, (
            "At least one of energy_score_global_weight, patch_energy_weight, "
            "variogram_weight must be > 0 -- otherwise there's no loss term at all."
        )
        self.mylog({"es_total": total.detach()})
        return total

    def loss(
        self,
        pred: TensorDict | list[TensorDict],
        gt: TensorDict,
        multistep: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Calculate the loss between predictions and ground truth.

        Args:
            pred: Predicted TensorDict values, or a list of M TensorDicts (one
                per stochastic member) when training an energy-score ensemble
                -- see ForecastModule.n_energy_score_members. Masking below is
                applied to every member; only the energy-score path actually
                uses more than the first.
            gt: Ground truth TensorDict values.
            multistep: Whether to compute multistep loss.
            **kwargs: Additional keyword arguments.

        Returns:
            Computed loss tensor.
        """
        loss_coeffs = self.loss_coeffs.to(self.device)

        is_ensemble = isinstance(pred, list)
        preds = pred if is_ensemble else [pred]

        for p in preds:
            p["surface"] = p["surface"] * self.surface_mask.to(self.device)
            p["level"] = p["level"] * self.level_mask.to(self.device)
            p["lev"] = p["lev"] * self.lev_mask.to(self.device)
        gt["surface"] = gt["surface"] * self.surface_mask.to(self.device)
        gt["level"] = gt["level"] * self.level_mask.to(self.device)
        gt["lev"] = gt["lev"] * self.lev_mask.to(self.device)
        gt.pop("spatial_forcings", None)
        gt.pop("non_spatial_forcings", None)
        for p in preds:
            p.pop("spatial_forcings", None)
            p.pop("non_spatial_forcings", None)
        if multistep:  # means we have to compute multistep loss
            # discount for multistep loss
            lead_iter = next(iter(gt.values())).shape[1]
            future_coeffs = (
                torch.tensor([1 / (1 + i) ** 2 for i in range(lead_iter)])
                .to(self.device)
                .reshape(-1, 1, 1, 1, 1)
            )

            loss_coeffs.apply(lambda x: x * future_coeffs)

        if self.use_energy_score:
            return self._energy_score_loss(preds if is_ensemble else preds[0], gt, loss_coeffs)

        pred = preds[0]
        weighted_error = (pred - gt).abs().pow(self.pow).mul(loss_coeffs)

        return sum(weighted_error.mean().values())

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Fires after backward(), before optimizer.step() (Lightning automatic optimization).

        Freezes every parameter except energy_score_noise_embedder while
        self.global_step is below noise_embedder_warmup_steps.
        """
        noise_embedder_warmup_steps = getattr(self, "noise_embedder_warmup_steps", 0)
        if noise_embedder_warmup_steps and self.global_step < noise_embedder_warmup_steps:
            for name, p in self.named_parameters():
                if p.grad is not None and "energy_score_noise_embedder" not in name:
                    p.grad = None

    def training_step(
        self, batch: dict[str, Any], batch_nb: int
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Perform a training step.

        Args:
            batch: The input batch.
            batch_nb: The batch number.

        Returns:
            Loss value or prediction dictionary.
        """
        step = self.global_step

        use_condition = (
            torch.rand((batch["state"].shape[0],), device=batch["state"].device) > self.uncond_proba
        )
        # add multistep after x amount of steps
        switch_step = 0  # or however many steps you want
        if step < switch_step or "future_states" not in batch:
            # standard prediction. When training an energy-score ensemble,
            # draw n_energy_score_members independent stochastic forward
            # passes (each with its own noise draw, see
            # shared_forward_logic) instead of one -- `pred` stays bound to
            # member 0, which is what seeds the pushforward rollout below (one
            # concrete sampled trajectory continues, matching how a
            # diffusion/flow rollout would pick a single sample).
            if self.use_energy_score and self.n_energy_score_members > 1:
                if self.energy_score_noise_mode == "perturbed_ic":
                    member_batches = self._make_energy_score_members(
                        batch, self.n_energy_score_members
                    )
                    preds = [self.forward(mb, use_condition=use_condition) for mb in member_batches]
                else:
                    preds = [
                        self.forward(batch, use_condition=use_condition)
                        for _ in range(self.n_energy_score_members)
                    ]
                pred = preds[0]
            else:
                pred = self.forward(batch, use_condition=use_condition)
                preds = pred
            # Save t+1 forcings before loss() pops them from next_state in-place.
            _next_sf = batch["next_state"].get("spatial_forcings", None)
            _next_nsf = batch["next_state"].get("non_spatial_forcings", None)
            if _next_sf is not None:
                _next_sf = _next_sf.clone()
            if _next_nsf is not None:
                _next_nsf = _next_nsf.clone()
            loss = self.loss(preds, batch["next_state"])

            # Pushforward regularization on lead-1 samples.
            # Feeds the detached one-step prediction back as input and supervises
            # the t+2 target with stability-weighted loss (polar cells upweighted
            # vs. area weighting so polar compounding gets penalised).
            has_pf = batch.get("has_pf", None)
            pf_active = (
                self.lambda_pf > 0.0
                and step >= self.pf_warmup_steps
                and has_pf is not None
                and bool(has_pf.all())
            )
            if pf_active:
                # Pushforward rollout, self.pf_n_steps long (capped by however many
                # pf_future_states targets the dataloader actually prepared for this
                # config's module.module.pf_n_steps -- see cmip_random_lead_time.py).
                # Gradient flows only through the last 2 steps; all earlier steps
                # are detached so memory stays bounded.  The forcing dropout mask is
                # drawn once here and reused across all steps (same consistency
                # rule as the 1-step forward pass, now held over the full rollout).
                # pf_future_states: (B, N, ...) — targets at t+2 … t+(N+1)
                pf_targets = batch["pf_future_states"]  # TensorDict with batch dim (B, N, ...)
                pf_n_steps = min(self.pf_n_steps, pf_targets.shape[1])
                pf_grad_steps = 2  # last N steps receive gradient
                loss_coeffs_pf = self.loss_coeffs_pf.to(self.device)
                surface_mask = self.surface_mask.to(self.device)
                level_mask = self.level_mask.to(self.device)
                lev_mask = self.lev_mask.to(self.device)

                # Draw forcing dropout mask once for the entire rollout.
                # Sample via the dropout module's interface so probabilities match training.
                B_pf = batch["state"]["non_spatial_forcings"].shape[0]
                if hasattr(self, "forcing_dropout"):
                    pf_forcing_mask = self.forcing_dropout.sample_mask(
                        B_pf, self.device
                    )  # (B, F+1)
                else:
                    pf_forcing_mask = None

                if self.use_energy_score and self.pf_temporal_energy_score:
                    # See pf_temporal_energy_score's docstring. pf_members
                    # INDEPENDENT, self-consistent trajectories, each rolled
                    # forward from a fresh single-step sample of `batch`
                    # (not reused from `preds` above, to keep this path's
                    # bookkeeping fully separate from the per-step path's).
                    pf_members = self.pf_n_energy_score_members or self.n_energy_score_members
                    if self.energy_score_noise_mode == "perturbed_ic":
                        seed_batches = self._make_energy_score_members(batch, pf_members)
                    else:
                        seed_batches = [batch] * pf_members
                    cur_states = [
                        self.forward(sb, use_condition=use_condition) for sb in seed_batches
                    ]
                    for cs in cur_states:
                        if _next_sf is not None:
                            cs["spatial_forcings"] = _next_sf
                        if _next_nsf is not None:
                            cs["non_spatial_forcings"] = _next_nsf
                    prev_states = [batch["state"] for _ in range(pf_members)]
                    cur_timestamp = batch["timestamp"] + batch["lead_time"]
                    lead_time = batch["lead_time"]

                    member_traj: list[list[TensorDict]] = [[] for _ in range(pf_members)]
                    gt_traj: list[TensorDict] = []
                    for k in range(pf_n_steps):
                        apply_grad = k >= (pf_n_steps - pf_grad_steps)
                        gt_k = pf_targets[:, k].exclude("spatial_forcings", "non_spatial_forcings")
                        gt_traj.append(gt_k)

                        next_states = []
                        for m in range(pf_members):
                            state_in = cur_states[m] if apply_grad else cur_states[m].detach()
                            prev_in = prev_states[m] if apply_grad else prev_states[m].detach()
                            pf_batch_km = {
                                "state": state_in,
                                "prev_state": prev_in,
                                "lead_time": lead_time,
                                "timestamp": cur_timestamp,
                            }
                            pred_km = self.forward(
                                pf_batch_km,
                                use_condition=True,
                                forcing_mask=pf_forcing_mask,
                            )
                            pred_km["surface"] = pred_km["surface"] * surface_mask
                            pred_km["level"] = pred_km["level"] * level_mask
                            pred_km["lev"] = pred_km["lev"] * lev_mask
                            pred_km_for_loss = pred_km.exclude(
                                "spatial_forcings", "non_spatial_forcings"
                            )
                            member_traj[m].append(pred_km_for_loss)

                            next_target = pf_targets[:, k]
                            next_state_m = pred_km.clone()
                            next_state_m["spatial_forcings"] = next_target["spatial_forcings"]
                            next_state_m["non_spatial_forcings"] = next_target[
                                "non_spatial_forcings"
                            ]
                            next_states.append(next_state_m)

                        prev_states = [cs.detach() for cs in cur_states]
                        cur_states = next_states
                        cur_timestamp = cur_timestamp + lead_time

                    # Stack each member's own per-step predictions along a new
                    # time axis (dim=1, right after batch) -> (B, pf_n_steps,
                    # vars, levels, lat, lon). _energy_score_loss's rmse
                    # building blocks flatten every non-batch dim together
                    # already (loss_coeffs_pf's broadcast still lines up: torch
                    # inserts the missing leading dim), so this pools skill/
                    # spread over TIME along with vars/levels/lat/lon with no
                    # further code changes.
                    preds_traj = [torch.stack(member_traj[m], dim=1) for m in range(pf_members)]
                    gt_traj_td = torch.stack(gt_traj, dim=1)
                    loss_pf = self._energy_score_loss(preds_traj, gt_traj_td, loss_coeffs_pf)
                    self.mylog(loss_pf=loss_pf)
                    loss = loss + self.lambda_pf * loss_pf
                    self.mylog(loss=loss)
                    return loss

                # Rollout: state at t+1 is `pred` (already computed above).
                # pf_future_states[:,0] = t+2, pf_future_states[:,1] = t+3, …
                cur_state = pred.clone()
                if _next_sf is not None:
                    cur_state["spatial_forcings"] = _next_sf
                if _next_nsf is not None:
                    cur_state["non_spatial_forcings"] = _next_nsf
                prev_state = batch["state"]
                cur_timestamp = batch["timestamp"] + batch["lead_time"]
                lead_time = batch["lead_time"]

                loss_pf = torch.tensor(0.0, device=self.device)
                for k in range(pf_n_steps):
                    apply_grad = k >= (pf_n_steps - pf_grad_steps)
                    state_in = cur_state if apply_grad else cur_state.detach()

                    pf_batch_k = {
                        "state": state_in,
                        "prev_state": prev_state.detach() if not apply_grad else prev_state,
                        "lead_time": lead_time,
                        "timestamp": cur_timestamp,
                    }
                    # Energy-score ensemble: M independent stochastic forward
                    # passes at this pf step too (same rationale as the main
                    # step above). preds_k[0] (== pred_k) is the member that
                    # continues the rollout below; all M score the pf loss.
                    # Uses pf_n_energy_score_members (falls back to
                    # n_energy_score_members) -- kept independently tunable
                    # since this cost multiplies by pf_n_steps on top.
                    pf_members = self.pf_n_energy_score_members or self.n_energy_score_members
                    n_members = pf_members if self.use_energy_score else 1
                    if self.use_energy_score and self.energy_score_noise_mode == "perturbed_ic":
                        pf_member_batches = self._make_energy_score_members(pf_batch_k, n_members)
                    else:
                        # "cond_token" mode (or no energy score at all): same batch
                        # object forwarded n_members times -- each forward() call
                        # independently draws its own cond-token noise internally
                        # (shared_forward_logic) when self.training, so this still
                        # yields n_members distinct stochastic members.
                        pf_member_batches = [pf_batch_k] * n_members
                    preds_k = [
                        self.forward(mb, use_condition=True, forcing_mask=pf_forcing_mask)
                        for mb in pf_member_batches
                    ]
                    for p_k in preds_k:
                        p_k["surface"] = p_k["surface"] * surface_mask
                        p_k["level"] = p_k["level"] * level_mask
                        p_k["lev"] = p_k["lev"] * lev_mask
                    pred_k = preds_k[0]

                    gt_k = pf_targets[:, k]
                    # Forcing keys live on the target state but not on the model's
                    # prediction; drop them so the two TensorDicts have matching keys.
                    gt_k = gt_k.exclude("spatial_forcings", "non_spatial_forcings")
                    preds_k_for_loss = [
                        p_k.exclude("spatial_forcings", "non_spatial_forcings") for p_k in preds_k
                    ]
                    # PF loss on the full field, so the forced global-mean
                    # trend is also supervised across the multi-step rollout.
                    preds_k_for_pf = preds_k_for_loss
                    gt_k_for_pf = gt_k
                    if self.use_energy_score:
                        loss_pf = loss_pf + self._energy_score_loss(
                            preds_k_for_pf if n_members > 1 else preds_k_for_pf[0],
                            gt_k_for_pf,
                            loss_coeffs_pf,
                        )
                    else:
                        pf_err_k = (
                            (preds_k_for_pf[0] - gt_k_for_pf)
                            .abs()
                            .pow(self.pow)
                            .mul(loss_coeffs_pf)
                        )
                        loss_pf = loss_pf + sum(pf_err_k.mean().values())

                    # Advance for next step: use next target's forcings, detach state.
                    next_target = pf_targets[:, k]
                    next_state = pred_k.clone()
                    next_state["spatial_forcings"] = next_target["spatial_forcings"]
                    next_state["non_spatial_forcings"] = next_target["non_spatial_forcings"]
                    prev_state = cur_state.detach()
                    cur_state = next_state
                    cur_timestamp = cur_timestamp + lead_time

                loss_pf = loss_pf / pf_n_steps
                self.mylog(loss_pf=loss_pf)
                loss = loss + self.lambda_pf * loss_pf

            self.mylog(loss=loss)
            return loss
        else:
            # multistep prediction
            lead_iter = batch["future_states"].shape[1]
            pred_future_states = self.forward_multistep(
                batch,
                iters=lead_iter,
                test_dataset=self.trainer.train_dataloader.dataset,
                index=0,
                ensemble_member_index=0,
            )
            loss = self.loss(pred_future_states, batch["future_states"], multistep=True)

            self.mylog(lead_iter=lead_iter)
            self.mylog(loss=loss)
            return loss

    def on_fit_start(self) -> None:
        """Initialise EMA shadow weights once the model is on-device."""
        if self.ema_decay > 0:
            self._ema = EMA(list(self.named_parameters()), decay=self.ema_decay)
            if hasattr(self, "_pending_ema_state"):
                device = next(self.parameters()).device
                sd = {k: v.to(device) for k, v in self._pending_ema_state.items()}
                self._ema.load_state_dict(sd)
                del self._pending_ema_state

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update EMA shadow weights after every optimizer step."""
        if hasattr(self, "_ema"):
            self._ema.update(list(self.named_parameters()), step=self.global_step)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Persist EMA shadow weights alongside the regular checkpoint.

        Saves two extra keys:
        - ``ema_shadow``: raw shadow dict (legacy / backward compat).
        - ``ema_state_dict``: full state_dict with EMA weights substituted for
          training weights (buffers kept as-is).  Load directly via
          ``model.load_state_dict(ckpt["ema_state_dict"])``.
        """
        if hasattr(self, "_ema"):
            shadow = {k: v.cpu() for k, v in self._ema.state_dict().items()}
            checkpoint["ema_shadow"] = shadow
            ema_sd = {
                k: (shadow[k] if k in shadow else v) for k, v in checkpoint["state_dict"].items()
            }
            checkpoint["ema_state_dict"] = ema_sd

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Stash loaded EMA state; applied to the EMA object in on_fit_start."""
        if "ema_shadow" in checkpoint:
            self._pending_ema_state = checkpoint["ema_shadow"]

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics; swap in EMA weights for validation."""
        for metric in self.val_metrics:
            metric.reset()
        if hasattr(self, "_ema"):
            # backs training weights up to CPU, loads EMA weights into params
            self._ema.swap_with(list(self.named_parameters()))

    @staticmethod
    def _save_rollout_chunk(
        out: TensorDict,
        out_dir: str,
        i: int,
        ensemble_member_index: int,
        seed_suffix: str,
        seed_suffix_list: list[str] | None,
        save_all_vars: bool,
    ) -> None:
        """Save one rollout chunk (`out["surface"]`, shape (B, T_chunk, ...)) to disk.

        One file per batch row when seed_suffix_list is given
        (generate_rollouts' batch_seeds path, B == len(seed_suffix_list)),
        matching exactly the files a sequential per-seed loop would have
        produced; otherwise (B == 1, the pre-existing default) one file
        for the whole (singleton-batch) tensor, unchanged from before.
        """
        suffixes = seed_suffix_list if seed_suffix_list is not None else [seed_suffix]
        for b, suf in enumerate(suffixes):
            # .clone() is required here: a [b:b+1] slice of a batched (B>1)
            # tensor is a view sharing the full B-row storage, so
            # torch.save on the unclosed view serializes all B rows' worth
            # of bytes into every single-seed file (silently, no shape
            # error -- torch.load still reports the sliced (1, ...) shape).
            torch.save(
                out["surface"][b : b + 1].clone(),
                f"{out_dir}/rollout_surface_{i}_{ensemble_member_index}{suf}.pt",
            )
            if save_all_vars:
                torch.save(
                    out["level"][b : b + 1].clone(),
                    f"{out_dir}/rollout_level_{i}_{ensemble_member_index}{suf}.pt",
                )
                torch.save(
                    out["lev"][b : b + 1].clone(),
                    f"{out_dir}/rollout_lev_{i}_{ensemble_member_index}{suf}.pt",
                )

    def forward_multistep(
        self,
        batch: dict[str, Any],
        iters: int | None = None,
        return_format: str = "tensordict",
        test_dataset: Any | None = None,
        is_sampling: bool = False,
        out_dir: str | None = None,
        index: int | None = None,
        ensemble_member_index: int | None = None,
        zero_spatial_forcing_indices: list[int] | None = None,
        zero_non_spatial_forcing_indices: list[int] | None = None,
        pi_spatial_values: torch.Tensor | None = None,
        pi_non_spatial_values: torch.Tensor | None = None,
        clamp_spatial_forcing_indices: list[int] | None = None,
        clamp_spatial_forcing_std: float | None = None,
        clamp_polar_tas_std: float | None = None,
        clamp_polar_n_rows: int = 5,
        save_all_vars: bool = False,
        energy_score_seed: int | None = None,
        energy_score_noise_scale: float = 1.0,
        seed_suffix: str = "",
        seed_suffix_list: list[str] | None = None,
        teacher_force: bool = False,
    ) -> list[TensorDict] | TensorDict:
        """Roll out the model autoregressively for multiple steps.

        Args:
            seed_suffix_list: If given, `batch`'s batch dimension holds
                len(seed_suffix_list) independently-seeded energy-score
                members stacked together (one batched forward pass per
                rollout step instead of one sequential call per seed --
                see generate_rollouts' batch_seeds). Each save point then
                writes one file per batch row, suffixed
                f"_{seed_suffix_list[b]}" instead of the single shared
                seed_suffix, so on-disk output is identical to running each
                seed as its own sequential call.
            teacher_force: If True, condition every step on the real
                ground-truth state from test_dataset instead of the model's
                own previous prediction -- isolates whether artifacts seen in
                free-running rollouts come from accumulated autoregressive
                drift/exposure bias vs. the model's single-pass output itself
                (ported from ocean_model/ArchesClimate/model/forecast.py's
                identical teacher_force path, not previously present in this
                repo). `pred` is still saved/returned for evaluation as
                normal; only what feeds back into loop_batch changes.
            batch: Initial batch with state, forcings, and timestamp.
            iters: Number of rollout iterations.
            return_format: Output format, 'tensordict' or 'list'.
            test_dataset: Dataset providing forcings and timestamps.
            is_sampling: Whether in inference/sampling mode.
            out_dir: Directory for intermediate saves.
            index: Dataset index for sampling mode.
            ensemble_member_index: Ensemble member index for saving.
            zero_spatial_forcing_indices: Channel indices to zero in spatial forcings.
            zero_non_spatial_forcing_indices: Channel indices to zero in non-spatial forcings.
            pi_spatial_values: If given, substitute these normalized
                pre-industrial-level values for zero_spatial_forcing_indices
                instead of the learned "channel absent" token.
            pi_non_spatial_values: Same, for zero_non_spatial_forcing_indices.
            clamp_spatial_forcing_indices: Channel indices in spatial forcings
                to clip to +/- clamp_spatial_forcing_std, applied each rollout
                step (see black-carbon/AIBCM SSP4-3.4 spike note in CLAUDE.md).
            clamp_spatial_forcing_std: Symmetric clamp bound, in standard
                deviations, for clamp_spatial_forcing_indices.
            clamp_polar_tas_std: If set, clip the model's own predicted
                surface tas (channel 0) to +/- this many normalized standard
                deviations at the poles, applied each rollout step to the
                model's own output before it's fed back in as the next
                step's input.
            clamp_polar_n_rows: Number of grid rows from each pole (South:
                rows 0..n-1, North: rows -n..-1) to clamp. Only used when
                clamp_polar_tas_std is set.
            save_all_vars: If True, also save level/lev rollouts, not just
                surface (overrides the `debug` surface-only shortcut).
            energy_score_seed: If set (only meaningful when self.use_energy_score
                is True), seeds a dedicated torch.Generator and draws a fresh
                noise vector from it on every autoregressive step -- via
                cond_tokens (energy_score_noise_dim) or direct state
                perturbation (energy_score_noise_std), depending on
                self.energy_score_noise_mode, mirroring the fresh-draw-per-
                forward-call behaviour used during training for that same
                mode -- instead of the default eval-time behaviour of no
                noise at all (a single deterministic-like trajectory). None
                (default) preserves that old deterministic-at-eval behaviour
                for every existing call site/model.
            energy_score_noise_scale: Multiplies the drawn noise_z before it
                reaches energy_score_noise_embedder (cond_token mode only --
                no effect in perturbed_ic mode, which already has its own
                magnitude knob via energy_score_noise_std). Since that
                embedder is a single nn.Linear with no nonlinearity, this
                scales the resulting conditioning perturbation exactly
                linearly -- equivalent to having trained with a
                proportionally larger noise std. Default 1.0 matches what
                every existing checkpoint was actually trained with (unit
                variance); values > 1 push the conditioning input off the
                training distribution for the rest of the network (which
                only ever saw unit-variance perturbations), so treat this as
                an experimental knob and sanity-check rollout quality before
                trusting a large scale.
            seed_suffix: Appended to saved rollout filenames -- always a
                plain digit (e.g. "_2") for a real energy-score seed, or
                "_0" for a deterministic/no-noise rollout (also see
                ROLLOUT_RE in analysis/compare_runs.py, which additionally
                still accepts the older "_det"/"_tf" sentinels found in
                already-generated files predating this convention).
                Standardized rollout filename is
                rollout_{domain}_{chunk}_{ensemble_member_index}
                {seed_suffix}.pt, 4 fields always -- never omitted, since
                omitting it let a deterministic comparison rollout collide
                with and silently overwrite a real seeded ensemble member's
                "_0.pt" file.
            pi_spatial_values: If given, substitute these normalized piControl
                values (shape (F, 144, 144), see load_pi_forcing_values) for
                zero_spatial_forcing_indices instead of the learned
                null_spatial_map token -- a physical "held at pre-industrial
                level" counterfactual rather than a "channel absent" one.
            pi_non_spatial_values: Same, for zero_non_spatial_forcing_indices
                (shape (F_ns,)), substituted instead of zero.
            clamp_spatial_forcing_indices: Channel indices in spatial forcings
                to clip to +/- clamp_spatial_forcing_std, applied each rollout
                step after the raw forcing is loaded from test_dataset. Since
                forcings are z-score normalized ((x - mean) / std, see
                CMIPBaseDataset.normalize), clamp_spatial_forcing_std is
                literally a number of standard deviations. Unlike
                zero_spatial_forcing_indices, this doesn't remove the channel's
                signal -- it caps outlier spikes while leaving the rest of the
                trajectory untouched.
            clamp_spatial_forcing_std: Symmetric clamp bound, in standard
                deviations, for clamp_spatial_forcing_indices. Required if
                clamp_spatial_forcing_indices is given.
            save_all_vars: If True, also save "level" and "lev" alongside
                "surface" for each chunk (needed to reconstruct a full
                prev_state for downstream analysis). Off by default since
                "level" is ~10x the size of "surface" per chunk.

        Returns:
            Stacked TensorDict of future states, or list if
            return_format='list'.
        """
        self.eval()
        if (
            getattr(self, "use_energy_score", False)
            and self.energy_score_noise_mode == "mc_dropout"
        ):
            self._set_mc_dropout_active(True)
        energy_score_generator = None
        if energy_score_seed is not None and getattr(self, "use_energy_score", False):
            energy_score_generator = torch.Generator(device=self.device)
            energy_score_generator.manual_seed(energy_score_seed)
        # sequential_date_indices = list(  # noqa: E501
        #     filter_unique_third_tuple_values(test_dataset.id2pt))
        preds_future = []
        loop_batch = {k: v for k, v in batch.items()}
        loop_batch["lead_time"] = torch.tensor([1], dtype=torch.int64).to(self.device)
        # Batch size of `batch` -- 1 for every existing call site, or
        # len(seed_suffix_list) under generate_rollouts' batch_seeds path
        # (multiple independently-seeded members stacked in one forward
        # pass). Forcings pulled from test_dataset below are per-timestep,
        # not per-batch-row, so they need broadcasting to this size rather
        # than the old hardcoded singleton-batch assignment.
        B = batch["state"]["surface"].shape[0]

        # if is_sampling:  # append first state as sanity check for sampling
        #     preds_future.append(loop_batch["state"].cpu())
        for i in range(iters):
            if i % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            # with torch.set_grad_enabled(not is_sampling):
            #     # for multistep training
            #     if torch.is_grad_enabled():
            #         pred = gradient_checkpoint.checkpoint(
            #             self.forward, loop_batch,
            #             use_condition=True, use_reentrant=False,
            #         )
            #     else:
            forward_batch, energy_score_noise = loop_batch, None
            if energy_score_generator is not None:
                forward_batch, energy_score_noise = self._energy_score_member_inputs(
                    loop_batch,
                    energy_score_generator,
                    energy_score_noise_scale,
                )
            pred = self.forward(
                forward_batch,
                use_condition=True,
                energy_score_noise=energy_score_noise,
            ).detach()
            # else:
            #     pred = self.forward(loop_batch,use_condition=True)
            preds_future.append(pred.cpu())
            if "future_states" not in batch.keys():
                i = i + 1
                # state_only=True: only next_batch's forcings are used below --
                # skips ~11x redundant memmap reads/clones per rollout step.
                next_batch = test_dataset.__getitem__(i, state_only=True)
                next_timestamp = next_batch["timestamp"]
                next_non_spatial_forcings = next_batch["state"]["non_spatial_forcings"][None]
                next_spatial_forcings = next_batch["state"]["spatial_forcings"][None]
            else:  # multstep training
                next_batch = batch["future_states"][:, i]
                next_timestamp = batch["future_timestamps"][:, i]
                next_non_spatial_forcings = next_batch["non_spatial_forcings"]
                next_spatial_forcings = next_batch["spatial_forcings"]

            # next_timestamp = torch.stack([  # noqa: E501
            #     torch.tensor(
            #         test_dataset.next_timestamp_map[x.item()],
            #         device=loop_batch['state'].device)
            #     for x in loop_batch['timestamp']])
            if pred["level"].shape[1] > 4:
                pred["level"][:, -1, :4, :, :] = torch.clamp(
                    pred["level"][:, -1, :4, :, :], max=6, min=-4
                )

            # print('surface',pred['surface'][0,0])
            # print(pred)
            # print(loop_batch['state']['surface'])
            # print('spatial_forcings',loop_batch['state']['spatial_forcings'])
            # print(index,i)
            # print(next_timestamp)
            # print('lead_time',batch['lead_time'])
            # print('lead_time_loop',loop_batch['lead_time'])

            # print(next_non_spatial_forcings)
            # print(next_spatial_forcings)
            # pred["surface"][:, 2, :, :, :] = torch.clamp(
            #     pred["surface"][:, 2, :, :, :], max=10, min=-2
            # )
            # pred["level"][:, 1:3, :, :, :] = torch.clamp(
            #     pred["level"][:, 1:3, :, :, :], max=1.5, min=-1.5
            # )
            # pred['level'][:,-1,:4,:,:] = zg_logged_normed[None,i]
            # pred['lev'][:,:,:10] = zg_data[:,i+1,:,:10]
            # m = torch.max(torch.where(  # noqa: E501
            #     torch.isnan(pred['level'][:,-1,0:3,:,:].max()),
            #     torch.tensor(-float('inf')),
            #     pred['level'][:,-1,0:3,:,:].max()))
            loop_batch = dict(
                prev_state=loop_batch[
                    "prev_state"
                ],  # copy data, update after adding new spatial forcings
                state=loop_batch["state"],
                timestamp=next_timestamp,
            )
            if self.load_prev > 1:
                loop_batch["prev_state"] = torch.stack(
                    [loop_batch["state"], loop_batch["prev_state"][:, 0]], dim=1
                )
            else:
                loop_batch["prev_state"] = loop_batch["state"].copy()
            if teacher_force and "future_states" not in batch.keys():
                # Real ground-truth state at this timestep instead of the model's
                # own (possibly drifting/artifacted) prediction -- next_batch here
                # is test_dataset's state_only=True item at index i, i.e. the
                # actual data, not a next_state/target. pred is left untouched
                # above/below (still appended to preds_future for evaluation).
                loop_batch["state"] = next_batch["state"].clone().to(self.device)
            else:
                loop_batch["state"] = pred
            loop_batch["state"]["non_spatial_forcings"] = next_non_spatial_forcings.expand(B, -1)
            loop_batch["state"]["spatial_forcings"] = next_spatial_forcings.expand(B, -1, -1, -1)
            if clamp_spatial_forcing_indices:
                loop_batch["state"]["spatial_forcings"][:, clamp_spatial_forcing_indices] = (
                    torch.clamp(
                        loop_batch["state"]["spatial_forcings"][:, clamp_spatial_forcing_indices],
                        min=-clamp_spatial_forcing_std,
                        max=clamp_spatial_forcing_std,
                    )
                )
            if zero_spatial_forcing_indices:
                if pi_spatial_values is not None:
                    # Physical "held at pre-industrial level" counterfactual
                    # instead of "channel absent" -- see load_pi_forcing_values.
                    loop_batch["state"]["spatial_forcings"][:, zero_spatial_forcing_indices] = (
                        pi_spatial_values[zero_spatial_forcing_indices].to(
                            loop_batch["state"]["spatial_forcings"].device
                        )
                    )
                else:
                    # Substitute the model's learned "absent" value, not zero --
                    # zero (in normalized space) reads as "at the channel's
                    # mean", not "absent", which understates the true ablation
                    # effect. See null_spatial_map in base_climate_module.py.
                    loop_batch["state"]["spatial_forcings"][:, zero_spatial_forcing_indices] = (
                        self.null_spatial_map[zero_spatial_forcing_indices]
                    )
            if zero_non_spatial_forcing_indices:
                if pi_non_spatial_values is not None:
                    loop_batch["state"]["non_spatial_forcings"][
                        :, zero_non_spatial_forcing_indices
                    ] = pi_non_spatial_values[zero_non_spatial_forcing_indices].to(
                        loop_batch["state"]["non_spatial_forcings"].device
                    )
                else:
                    # No equivalent fix here: null_ssi_token lives in embedding
                    # space (substituted after solar_embedder, not on the raw
                    # non_spatial_forcings value), and ForcingDropout only ever
                    # masks the whole non-spatial group as one unit during
                    # training -- there's no learned "this one specific
                    # non-spatial channel is absent" state to substitute for.
                    # Zeroing here remains an acknowledged-imperfect proxy.
                    loop_batch["state"]["non_spatial_forcings"][
                        :, zero_non_spatial_forcing_indices
                    ] = 0
            loop_batch["state"]["surface"] *= self.surface_mask.to(self.device)
            loop_batch["prev_state"]["surface"] *= self.surface_mask.to(self.device)
            loop_batch["state"]["level"] *= self.level_mask.to(self.device)
            loop_batch["prev_state"]["level"] *= self.level_mask.to(self.device)
            loop_batch["state"]["lev"] *= self.lev_mask.to(self.device)
            loop_batch["prev_state"]["lev"] *= self.lev_mask.to(self.device)
            loop_batch["lead_time"] = torch.tensor([1], dtype=torch.int64).to(self.device)
            if ((len(preds_future) % 120) == 0) and (i > 0):
                if is_sampling:
                    out = torch.stack(preds_future, dim=1)
                    self._save_rollout_chunk(
                        out,
                        out_dir,
                        i,
                        ensemble_member_index,
                        seed_suffix,
                        seed_suffix_list,
                        save_all_vars,
                    )
                    preds_future = []

        if is_sampling and preds_future:
            # Flush the final partial chunk (< 120 steps) -- without this,
            # any rollout whose length isn't a multiple of 120 silently
            # loses its last steps, since the save trigger above only fires
            # on exact multiples of 120.
            out = torch.stack(preds_future, dim=1)
            self._save_rollout_chunk(
                out,
                out_dir,
                i,
                ensemble_member_index,
                seed_suffix,
                seed_suffix_list,
                save_all_vars,
            )
            preds_future = []

        if not preds_future:
            # Everything was already flushed to disk above (sampling mode);
            # the caller doesn't use this return value in that case. Avoid
            # crashing on stack([]).
            return [] if return_format == "list" else TensorDict({}, batch_size=[])
        if return_format == "list":
            return preds_future
        preds_future = torch.stack(preds_future, dim=1)
        return preds_future

    @staticmethod
    def _rollout_skill(
        pred_surface: torch.Tensor,
        gt_surface: torch.Tensor,
        lat_coeffs: torch.Tensor,
        surface_mean: torch.Tensor,
        surface_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rollout skill score (lower = better) against ground truth.

        Works in physical units so the forced-response signal (a few K of
        warming over decades) is not swamped by the normalisation std (~13 K).
        Steps:
          1. Denormalise to physical units using per-pixel stats.
          2. Area-weighted global mean → (T, V).
          3. Annual means → removes seasonal cycle.
          4. 10-year running mean → isolates decadal forced response.
          5. RMSE of the smoothed trajectories, each variable standardised
             by its own GT trajectory amplitude so all V contribute equally.

        Args:
            pred_surface: (T, V, 1, H, W) predicted surface in normalised space.
            gt_surface:   (T, V, 1, H, W) ground truth surface in normalised space.
            lat_coeffs:   (H,) cosine-latitude area weights (unnormalised).
            surface_mean: (V, 1, H, W) per-pixel normalisation mean.
            surface_std:  (V, 1, H, W) per-pixel normalisation std.

        Returns:
            Tuple of (scalar skill score, area-weighted global-mean physical-unit
            predicted trajectory of shape (T, V)).
        """
        T, V, _, H, W = pred_surface.shape
        dev = pred_surface.device
        w = (lat_coeffs / lat_coeffs.mean()).to(dev)  # (H,)

        # Denormalise: (T, V, H, W) in physical units
        mu = surface_mean[:, 0].to(dev)  # (V, H, W)
        sigma = surface_std[:, 0].to(dev)  # (V, H, W)
        pred_phys = pred_surface[:, :, 0] * sigma[None] + mu[None]
        gt_phys = gt_surface[:, :, 0] * sigma[None] + mu[None]

        def area_mean(x: torch.Tensor) -> torch.Tensor:
            # x: (T, V, H, W) → (T, V)
            return (x * w[None, None, :, None]).mean(dim=(-2, -1))

        pred_gm = area_mean(pred_phys)  # (T, V)
        gt_gm = area_mean(gt_phys)

        # Annual means — collapses 12 monthly steps into 1 year
        T12 = (T // 12) * 12
        n_years = T12 // 12
        pred_annual = pred_gm[:T12].view(n_years, 12, V).mean(1)  # (n_years, V)
        gt_annual = gt_gm[:T12].view(n_years, 12, V).mean(1)

        # 10-year running mean — retains only the decadal forced response
        W10 = min(10, n_years)
        n_s = n_years - W10 + 1
        pred_smooth = torch.stack(
            [pred_annual[i : i + W10].mean(0) for i in range(n_s)]
        )  # (n_s, V)
        gt_smooth = torch.stack([gt_annual[i : i + W10].mean(0) for i in range(n_s)])

        # Standardise each variable by the amplitude of the GT forced response so
        # high-variance variables (psl in Pa) don't dominate low-variance ones (tas in K).
        gt_amp = gt_smooth.std(0, correction=0).clamp(min=1e-6)  # (V,)
        skill = ((pred_smooth - gt_smooth) / gt_amp[None]).pow(2).mean().sqrt()

        return skill, pred_gm

    def _rollout_from(
        self,
        dataset: Any,
        target_time: np.datetime64,
        rollout_length: int,
        metric: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressively roll out from a fixed start date and return the rollout skill.

        Args:
            dataset: Dataset to roll out over (must support __getitem__ and id2pt).
            target_time: Start date to locate the initial condition in the dataset.
            rollout_length: Number of steps to roll out.
            metric: Optional ensemble metric to update at each step (skipped if None).

        Returns:
            Tuple of (scalar rollout skill tensor, area-weighted global-mean surface
            tas trajectory in physical units, shape (T,); see `_rollout_skill`).
        """
        # dataset.id2pt timestamps may be cftime.datetime (e.g. CanESM5's noleap
        # calendar) or np.datetime64 (standard-calendar datasets) depending on
        # the source data; comparing the two types directly with `==` silently
        # returns False rather than raising, so `target_time` (always
        # np.datetime64) would never match a cftime-typed dataset. Compare on
        # (year, month, day, hour) components instead, which both types expose.
        target_ymdh = _datetime_ymdh(target_time)
        matching_indices = [
            i for i, t in enumerate(dataset.id2pt.values()) if _datetime_ymdh(t[2]) == target_ymdh
        ]

        init = dataset.__getitem__(matching_indices[0])
        loop_batch = {k: init[k].unsqueeze(0).to(self.device) for k in init.keys()}
        loop_batch["lead_time"] = torch.tensor([1], dtype=torch.int64, device=self.device)

        # Keep only surface predictions on GPU (~5 MB × 240 ≈ 1.2 GB)
        # instead of all tensors on CPU (~80 MB × 240 ≈ 19 GB system RAM).
        pred_surfaces: list[torch.Tensor] = []
        gt_surface_list: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(rollout_length):
                pred = self.forward(loop_batch, use_condition=True)

                # Clamp zg to prevent runaway (mirrors forward_multistep).
                if pred["level"].shape[1] > 4:
                    pred["level"][:, -1, :4] = torch.clamp(pred["level"][:, -1, :4], max=6, min=-4)

                # Clone surface before in-place masking below corrupts it.
                pred_surface = pred["surface"].clone()  # (1, V, D, H, W)
                pred_surfaces.append(pred_surface)

                gt_item = dataset.__getitem__(i)
                if metric is not None:
                    metric.update(
                        gt_item["state"]["surface"][None, :, 0].to(self.device),
                        pred_surface[:, :, 0],
                    )
                gt_surface_list.append(gt_item["state"]["surface"].cpu())

                # Next forcings from dataset[i+1] (matches forward_multistep).
                next_item = dataset.__getitem__(i + 1) if i + 1 < len(dataset) else gt_item
                next_non_spatial = next_item["state"]["non_spatial_forcings"][None].to(self.device)
                next_spatial = next_item["state"]["spatial_forcings"][None].to(self.device)

                if self.load_prev > 1:
                    loop_batch["prev_state"] = torch.stack(
                        [loop_batch["state"], loop_batch["prev_state"][:, 0]], dim=1
                    )
                else:
                    loop_batch["prev_state"] = loop_batch["state"].copy()
                loop_batch["state"] = pred
                loop_batch["state"]["non_spatial_forcings"] = next_non_spatial
                loop_batch["state"]["spatial_forcings"] = next_spatial
                loop_batch["state"]["surface"] *= self.surface_mask.to(self.device)
                loop_batch["state"]["level"] *= self.level_mask.to(self.device)
                loop_batch["state"]["lev"] *= self.lev_mask.to(self.device)
                loop_batch["prev_state"]["surface"] *= self.surface_mask.to(self.device)
                loop_batch["prev_state"]["level"] *= self.level_mask.to(self.device)
                loop_batch["prev_state"]["lev"] *= self.lev_mask.to(self.device)
                loop_batch["lead_time"] = torch.tensor([1], dtype=torch.int64, device=self.device)

        # Rollout skill on CPU. Do NOT call empty_cache() here — releasing
        # the CUDA allocator pool forces training to re-request memory in
        # small chunks, spiking system RAM and triggering the OOM.
        gt_surface = torch.stack(gt_surface_list).float()  # (T, V, D, H, W) CPU
        pred_surface_cpu = torch.stack(pred_surfaces).squeeze(1).float().cpu()  # (T, V, D, H, W)
        del pred_surfaces, gt_surface_list

        lat_coeffs = torch.tensor(
            [
                torch.cos(x)
                for x in torch.arange(-torch.pi / 2 + 1e-2, torch.pi / 2, torch.pi / self.lat_dim)
            ]
        )
        skill, pred_gm = self._rollout_skill(
            pred_surface_cpu,
            gt_surface,
            lat_coeffs,
            self.surface_mean_buf.cpu(),
            self.surface_std_buf.cpu(),
        )
        tas_idx = self.cfg.surface_variables.index("tas")
        global_mean_tas = pred_gm[:, tas_idx]
        del pred_surface_cpu, gt_surface
        return skill, global_mean_tas

    def validation_step(self, batch: dict[str, Any], batch_nb: int) -> None:
        """Run validation rollouts from a fixed start date and log metrics.

        Runs the standard rollout over the validation dataset (ssp434), plus a
        separate rollout over the ssp585 test-split dataset, logged as
        `ssp585_rollout_skill`. Also logs the raw global-mean surface tas
        trajectory for each rollout as `global_mean_temp_ssp434` /
        `global_mean_temp_ssp585` W&B line plots.

        Args:
            batch: Validation batch (unused; uses fixed start date).
            batch_nb: Batch index (unused).

        Returns:
            None
        """
        if self.global_rank != 0:
            return  # Only run on GPU 0
        val_rollout_length = 1000
        target_time = np.datetime64("2015-01-16T12:00:00.000000000")

        self.val_metrics = [x.to(self.device) for x in self.val_metrics]
        dataset = self.trainer.val_dataloaders.dataset
        metric = self.val_metrics[0]

        skill, global_mean_tas = self._rollout_from(
            dataset, target_time, val_rollout_length, metric=metric
        )
        self.mylog(rollout_skill=skill.to(self.device), mode="val_")
        self._log_global_mean_temp(global_mean_tas, "global_mean_temp_ssp434")

        if hasattr(self, "_ssp585_test_dataset"):
            ssp585_skill, ssp585_global_mean_tas = self._rollout_from(
                self._ssp585_test_dataset, target_time, val_rollout_length
            )
            self.mylog(ssp585_rollout_skill=ssp585_skill.to(self.device), mode="val_")
            self._log_global_mean_temp(ssp585_global_mean_tas, "global_mean_temp_ssp585")

        return None

    def _log_global_mean_temp(self, trajectory: torch.Tensor, key: str) -> None:
        """Log a monthly global-mean surface tas trajectory to W&B as a line plot.

        `self.log` only accepts scalars, so the full (T,) rollout trajectory is
        logged directly through the underlying wandb run instead.

        Args:
            trajectory: (T,) global-mean surface tas in physical units (K).
            key: W&B panel name, e.g. "global_mean_temp_ssp585".
        """
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        import wandb

        values = trajectory.tolist()
        table = wandb.Table(data=[[i, v] for i, v in enumerate(values)], columns=["month", "tas"])
        experiment.log(
            {key: wandb.plot.line(table, "month", "tas", title=key)},
            step=self.global_step,
        )

    def on_validation_epoch_end(self) -> None:
        """Restore training weights after validation, then log metrics."""
        if hasattr(self, "_ema"):
            # restores training weights from the CPU backup made above
            self._ema.swap_with(list(self.named_parameters()))
        for metric in self.val_metrics:
            outputs = metric.compute()
            for k, v in outputs.items():
                outputs[k] = v.mean()
            self.mylog(**outputs, mode="val_")
            metric.reset()

    def generate_rollouts(
        self,
        cfg: Any,
        batch_index: int,
        test_dataset: Any,
        target_name: str,
        out_dir: str,
        start_member: int,
        end_member: int,
        flat_forcings: bool,
        num_rollout_steps: int,
        num_perturbations_per_member: int = 1,
        debug: bool = False,
        seed_index: int = 0,
        zero_spatial_forcing_indices: list[int] | None = None,
        zero_non_spatial_forcing_indices: list[int] | None = None,
        zero_forcing_source: str = "null",
        clamp_spatial_forcing_indices: list[int] | None = None,
        clamp_spatial_forcing_std: float | None = None,
        clamp_polar_tas_std: float | None = None,
        clamp_polar_n_rows: int = 5,
        save_all_vars: bool = False,
        teacher_force: bool = False,
        energy_score_noise_scale: float = 1.0,
        batch_seeds: bool = False,
    ) -> None:
        """Generate model rollouts and save them to disk.

        Args:
            cfg: Configuration object.
            batch_index: Index of the batch.
            test_dataset: Dataset for testing.
            target_name: Name of the target.
            out_dir: Output directory path.
            start_member: Start index for ensemble members.
            end_member: End index for ensemble members.
            flat_forcings: Whether to use flat forcings.
            num_rollout_steps: Length of the rollout sequence.
            seed_index: If set, run only this seed (0-indexed) instead of
                all num_perturbations_per_member seeds batched/looped
                together. Use this to parallelise seeds across separate jobs.
            energy_score_noise_scale: Multiplies the drawn noise_z before it
                reaches energy_score_noise_embedder (see forward_multistep's
                energy_score_noise_scale).
            clamp_polar_tas_std: If set, clip the model's own predicted
                surface tas (channel 0) to +/- this many normalized standard
                deviations at the poles, applied each rollout step (see
                forward_multistep's clamp_polar_tas_std).
            clamp_polar_n_rows: Number of grid rows from each pole to clamp.
                Only used when clamp_polar_tas_std is set.
            batch_seeds: Only meaningful when this is a seeded energy-score
                ensemble (use_energy_score and num_perturbations_per_member
                > 1) and seed_index is None (running every seed, not one
                job-per-seed). If True, run every seed of a given
                ensemble_member as ONE batched forward pass (batch size =
                num_perturbations_per_member) instead of num_perturbations_per_member
                sequential single-sample calls -- a real speedup on one GPU
                when the rollout is compute-bound, since energy_score_noise
                is drawn once per step as (B, noise_dim) from a single
                seeded generator, giving every batch row its own
                independent draw already (no separate per-seed generator
                needed). Output on disk is byte-identical in naming/shape to
                the sequential path (see _save_rollout_chunk) -- purely a
                performance toggle, off by default to keep existing
                behaviour/memory footprint unchanged.
            teacher_force: See forward_multistep's teacher_force -- condition
                every step on real ground truth instead of the model's own
                prediction, to isolate autoregressive-accumulation artifacts
                from single-pass ones.
            num_perturbations_per_member: Number of perturbations
                per ensemble member.
            debug: Whether to enable debug mode.
            zero_spatial_forcing_indices: Channel indices to zero in spatial forcings.
            zero_non_spatial_forcing_indices: Channel indices to zero in non-spatial forcings.
            zero_forcing_source: "null" (default) substitutes the learned
                null_spatial_map / null_ssi_token "channel absent" value, as
                before. "pi" substitutes the piControl experiment's own
                (normalized) forcing values instead -- a physical "held at
                pre-industrial level" counterfactual. See
                load_pi_forcing_values.
            clamp_spatial_forcing_indices: Channel indices to clip to
                +/- clamp_spatial_forcing_std standard deviations, see
                forward_multistep.
            clamp_spatial_forcing_std: Symmetric clamp bound, in standard
                deviations, for clamp_spatial_forcing_indices.
            save_all_vars: If True, also save "level"/"lev" per chunk, not just "surface".
        """
        os.makedirs(out_dir, exist_ok=True)

        assert zero_forcing_source in ("null", "pi"), (
            f"Unknown zero_forcing_source {zero_forcing_source!r}, expected 'null' or 'pi'"
        )
        pi_spatial_values = None
        pi_non_spatial_values = None
        if zero_forcing_source == "pi" and (
            zero_spatial_forcing_indices or zero_non_spatial_forcing_indices
        ):
            # Derived from an actual dataset file rather than
            # cfg.cluster.work_path + a hardcoded "memmap_filled_in" -- works
            # for any model's memmap directory (e.g. CanESM5 native's
            # memmap_filled_in_canesm5_native), not just IPSL's.
            memmap_dir = str(Path(test_dataset.files[0]).parent)
            pi_spatial_values, pi_non_spatial_values = load_pi_forcing_values(
                memmap_dir, test_dataset
            )
            pi_spatial_values = pi_spatial_values.to(self.device)
            pi_non_spatial_values = pi_non_spatial_values.to(self.device)

        # starting_file_indexes = [  # noqa: E501
        #     i for i, s in enumerate(test_dataset.files)
        #     if 'ssp534-over' in s.lower()]
        starting_indexes = [i for i, t in enumerate(test_dataset.id2pt.values()) if t[1] == 0]
        # if (
        #     len(
        #         [
        #             i
        #             for i, s in enumerate(test_dataset.files)
        #             if "ssp534-over" in s.lower()
        #         ]
        #     )
        #     > 0
        # ):  # useful if more than one ensemble member for ssp534-over
        #     # need to replace ssp534 dataset with ssp585 dataset
        #     rollout_length = 731
        #     batch_gen_dataset = hydra.utils.instantiate(
        #         cfg.dataloader.dataset, domain="val"
        #     )

        #     starting_indexes = [
        #         i for i, t in enumerate(test_dataset.id2pt.values())
        #         if t[1] == 300
        #     ]
        # else:
        # Noise-seeded ensembling: only meaningful for use_energy_score models
        # (see shared_forward_logic). Previously only activated when
        # num_perturbations_per_member > 1, which meant a plain single-member
        # rollout (num_seeds=1) silently skipped energy-score noise entirely
        # -- energy_score_noise_scale had no effect since energy_score_seed
        # stayed None and forward_multistep never built a generator. Now any
        # use_energy_score model gets a seeded (and thus noise-scaled) pass
        # even for a single member, defaulting to seed 0 -- every other
        # model/call keeps the single seeds=[None] pass below, with
        # energy_score_seed=None.
        run_seeded_ensemble = (
            getattr(self, "use_energy_score", False) and num_perturbations_per_member >= 1
        )
        if run_seeded_ensemble:
            seeds = (
                [seed_index]
                if seed_index is not None
                else list(range(num_perturbations_per_member))
            )
        else:
            seeds = [None]

        for ensemble_member in range(start_member, end_member):
            # only works for ssps, not for train/multi-dataset targets
            starting_indexes[ensemble_member]
            if run_seeded_ensemble and batch_seeds and len(seeds) > 1:
                # One batched forward pass, batch size == len(seeds), instead
                # of len(seeds) sequential single-sample calls -- see
                # generate_rollouts' batch_seeds docstring. Every batch row
                # starts from the identical initial state; per-row diversity
                # comes entirely from energy_score_noise being drawn as
                # (B, noise_dim) from one seeded generator inside
                # forward_multistep (each row gets an independent slice of
                # that generator's stream), same as the sequential path's
                # per-seed reseeding, just in one call instead of many.
                batch = test_dataset.__getitem__(0)
                # unsqueeze(0) first (identical to the sequential path below,
                # which is known-good), then tile via torch.cat rather than
                # .expand -- nested TensorDict values (e.g. "state") report
                # ndim as their own declared batch-dim count, which doesn't
                # reliably match the tensor shape after unsqueeze, so an
                # expand target built from that ndim can mismatch actual
                # shape length. torch.cat sidesteps that: it just
                # concatenates len(seeds) identical, already-correctly-shaped
                # copies along dim 0.
                batch = {k: v.unsqueeze(0) for k, v in batch.items()}
                batch = {k: torch.cat([v] * len(seeds), dim=0) for k, v in batch.items()}
                batch = {k: v.to(self.device) for k, v in batch.items()}
                energy_score_seed = (batch_index * 10) + ensemble_member
                seed_suffix_list = [f"_{s}" for s in seeds]
                self.forward_multistep(
                    batch,
                    iters=num_rollout_steps,
                    test_dataset=test_dataset,
                    is_sampling=True,
                    out_dir=out_dir,
                    index=0,
                    ensemble_member_index=ensemble_member,
                    zero_spatial_forcing_indices=zero_spatial_forcing_indices,
                    zero_non_spatial_forcing_indices=zero_non_spatial_forcing_indices,
                    pi_spatial_values=pi_spatial_values,
                    pi_non_spatial_values=pi_non_spatial_values,
                    clamp_spatial_forcing_indices=clamp_spatial_forcing_indices,
                    clamp_spatial_forcing_std=clamp_spatial_forcing_std,
                    clamp_polar_tas_std=clamp_polar_tas_std,
                    clamp_polar_n_rows=clamp_polar_n_rows,
                    save_all_vars=save_all_vars,
                    energy_score_seed=energy_score_seed,
                    energy_score_noise_scale=energy_score_noise_scale,
                    seed_suffix_list=seed_suffix_list,
                    teacher_force=teacher_force,
                )
                continue
            for seed in seeds:
                batch = test_dataset.__getitem__(0)
                # batch = self.trainer.val_dataloaders.dataset.__getitem__(b_index)
                batch = {k: batch[k].unsqueeze(0) for k in batch.keys()}
                batch = {k: batch[k].to(self.device) for k in batch.keys()}
                energy_score_seed = None
                # Standardized rollout filename:
                # rollout_{domain}_{chunk}_{ensemble_member}_{seed}.pt,
                # always 4 fields, seed always numeric (0 for a
                # deterministic/no-noise rollout) so every downstream
                # consumer that parses/globs this filename can rely on a
                # plain digit in this position.
                seed_suffix = "_0"
                if run_seeded_ensemble:
                    # Matches diffusion.py's DCPPDiffusion.generate_rollouts
                    # base_seed convention (batch_index * 10 + ensemble_member),
                    # offset by the perturbation seed so each member/seed pair
                    # gets an independent, reproducible noise draw.
                    energy_score_seed = (batch_index * 10) + ensemble_member + seed
                    seed_suffix = f"_{seed}"
                self.forward_multistep(
                    batch,
                    iters=num_rollout_steps,
                    test_dataset=test_dataset,
                    is_sampling=True,
                    out_dir=out_dir,
                    index=0,
                    ensemble_member_index=ensemble_member,
                    zero_spatial_forcing_indices=zero_spatial_forcing_indices,
                    zero_non_spatial_forcing_indices=zero_non_spatial_forcing_indices,
                    pi_spatial_values=pi_spatial_values,
                    pi_non_spatial_values=pi_non_spatial_values,
                    clamp_spatial_forcing_indices=clamp_spatial_forcing_indices,
                    clamp_spatial_forcing_std=clamp_spatial_forcing_std,
                    clamp_polar_tas_std=clamp_polar_tas_std,
                    clamp_polar_n_rows=clamp_polar_n_rows,
                    save_all_vars=save_all_vars,
                    energy_score_seed=energy_score_seed,
                    energy_score_noise_scale=energy_score_noise_scale,
                    seed_suffix=seed_suffix,
                    teacher_force=teacher_force,
                )

            # torch.save(
            #     rollout["surface"],
            #     f"{out_dir}/rollout_{ensemble_member}_surface.pt"
            # )
            # torch.save(
            #     rollout["level"],
            #     f"{out_dir}/rollout_{ensemble_member}_level.pt"
            # )
            # torch.save(  # noqa: E501
            #     rollout["lev"],
            #     f"{out_dir}/rollout_{ensemble_member}_lev.pt")


class ForecastModuleWithCond(ForecastModule):
    """A forecast module that can handle conditional inputs.

    This module can take additional information:
    - month and hour
    - previous state
    - pred state (e.g. prediction of other weather model)

    Attributes:
        cond_dim (int): Dimension of the conditional embedding
        use_prev (bool): Whether to use previous state
    """

    def __init__(
        self,
        *args: Any,
        cond_dim: int = 32,
        use_prev: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize ForecastModuleAvg.

        Args:
            *args: Positional arguments for parent class.
            cond_dim: Conditioning dimension.
            use_prev: Whether to use previous state.
            **kwargs: Keyword arguments for parent class.
        """
        super().__init__(*args, **kwargs)
        # cond_dim should be given as arg to the backbone

    def forward(self, batch: dict[str, Any], use_condition: bool, **kwargs) -> TensorDict:
        """Forward pass with conditional inputs.

        Args:
            batch: Input batch.
            use_condition: Whether to apply conditioning mask.
            **kwargs: Additional keyword arguments forwarded to the parent forward().

        Returns:
            TensorDict of output predictions.
        """
        return super().forward(batch, use_condition=use_condition, **kwargs)
