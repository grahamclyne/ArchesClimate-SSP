import importlib
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import lovely_tensors as lt
import torch
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
)
from lightning.pytorch.utilities import rank_zero_warn
from tensordict.tensordict import TensorDict
from tqdm import tqdm

from ArchesClimate.utils.tensordict_utils import tensordict_apply

lt.monkey_patch()
import ArchesClimate.stats as ArchesClimate_stats  # noqa: E402
from ArchesClimate.model.base_climate_module import (  # noqa: E402
    EMA,
    ClimateLightningModule,
)

ArchesClimate_stats_path = importlib.resources.files(ArchesClimate_stats)


class DCPPDiffusion(ClimateLightningModule):
    """DCPP Diffusion model for climate forecasting."""

    def __init__(
        self,
        cfg: Any | None = None,
        name: str = "diffusion",
        add_input_state: str = "default",
        cond_dim: int = 256,
        num_train_timesteps: int = 1000,
        scheduler: str = "ddpm",
        prediction_type: str = "v_prediction",
        beta_schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.012,
        snr_gamma: float | None = None,
        flow_matching: bool = False,
        conditional: str = "",
        uncond_proba: float = 0.0,
        forcing_drop_p: float = 0.20,
        forcing_all_present_prob: float = 0.40,
        forcing_dropout_groups: list[list[int]] | None = None,
        polar_loss_min_weight: float = 0.0,
        ckpt_path: str | None = None,
        ft_lr: float = 1e-5,
        snr_reweighting: bool = False,
        selected_vars: bool = False,
        num_inference_steps: int = 50,
        cf_guidance: float = 1.0,
        num_members: int = 1,
        debug: bool = False,
        load_deterministic_model: bool | str | list[str] = False,
        det_ckpt_fname: str = "epoch-epoch=06.ckpt",
        loss_delta_normalization: bool = False,
        pow: int = 2,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.98),
        weight_decay: float = 1e-5,
        num_warmup_steps: int = 1000,
        num_training_steps: int = 300000,
        num_cycles: float = 0.5,
        num_rollout_steps: int = 10,
        num_batch_examples: int = 1,
        dataset: Any | None = None,
        s_churn: int = 0,
        stochastic_sampling: bool = False,
        amplitude_loss_weight: float = 0.0,
        scale_input_noise: float = 1.0,
        boundary_padding: bool = False,
        prior: str = "normal",
        load_prev: int = 0,
        total_boundary_calc: bool = False,
        save_v_fields: bool = False,
        learn_residual: bool | str = False,
        state_normalization: bool = False,
        sd3_timestep_sampling: bool = True,
        delta_std_name: str | None = None,
        add_stochastic_noise: bool = False,
        state_scaler_multiplier: float = 1.0,
        optimizer_name: str = "adamw",
        muon_lr: float | None = None,
        adamw_lr: float | None = None,
        co2_linear_gain: bool = False,
        co2_gain_mode: str = "scalar",
        ema_decay: float = 0.0,
        pf_conditioning: bool = False,
        pf_n_steps: int = 8,
        pf_warmup_steps: int = 0,
    ):
        """Initializes the DCPPDiffusion model.

        Args:
            cfg: Configuration object.
            name: Name of the model.
            add_input_state: How to add input state to the output.
            cond_dim: Dimension of the conditioning embedding.
            num_train_timesteps: Number of training timesteps for the scheduler.
            scheduler: Type of scheduler to use ('ddpm', 'heun', 'flow').
            prediction_type: The prediction type of the scheduler.
            beta_schedule: The beta schedule for the scheduler.
            beta_start: The starting beta value.
            beta_end: The ending beta value.
            snr_gamma: Signal-to-noise ratio gamma value.
            flow_matching: Whether to use flow matching.
            conditional: String defining the conditioning keys.
            uncond_proba: Probability of unconditional training.
            forcing_drop_p: Per-forcing dropout probability (read via
                cfg.module.get in base_climate_module's ForcingDropout setup).
            forcing_all_present_prob: Probability that all forcings are kept
                (read via cfg.module.get in base_climate_module).
            forcing_dropout_groups: Channel-index groups dropped as a single
                unit (read via cfg.module.get in base_climate_module's
                ForcingDropout setup, same as forcing_drop_p above).
            polar_loss_min_weight: Floor applied to the per-row cos(lat)
                training loss weight (read via cfg.module.get in
                base_climate_module's prep_model(), same pattern as
                forcing_drop_p above).
            ckpt_path: Path to a checkpoint to load from.
            ft_lr: Fine-tuning learning rate.
            snr_reweighting: Whether to use SNR reweighting.
            selected_vars: Whether to use selected variables.
            num_inference_steps: Number of inference steps.
            cf_guidance: Classifier-free guidance scale.
            num_members: Number of ensemble members.
            debug: Whether to run in debug mode.
            load_deterministic_model: Path to a deterministic model to load.
            det_ckpt_fname: Filename of the deterministic model checkpoint.
            loss_delta_normalization: Whether to use loss delta normalization.
            pow: Power for the loss function.
            lr: Learning rate.
            betas: Adam optimizer betas.
            weight_decay: Weight decay for the optimizer.
            num_warmup_steps: Number of warmup steps for the scheduler.
            num_training_steps: Total number of training steps.
            num_cycles: Number of cycles for the cosine scheduler.
            num_rollout_steps: Number of steps for rollout generation.
            num_batch_examples: Number of batch examples for inference.
            dataset: The dataset object.
            s_churn: Churning parameter for the scheduler.
            stochastic_sampling: Must be True for s_churn (and s_tmin/s_tmax/
                s_noise) to have any effect -- FlowMatchEulerDiscreteScheduler.step()
                takes a purely deterministic ODE branch when this is False.
            amplitude_loss_weight: Weight of the auxiliary amplitude-matching
                loss term added to the base flow-matching loss, 0 disables it.
            scale_input_noise: Scale for the input noise.
            boundary_padding: Whether to use boundary padding.
            prior: Type of prior to use ('normal', 'student').
            load_prev: Number of previous states to load.
            total_boundary_calc: Whether to use total boundary calculation.
            save_v_fields: Whether to save velocity fields during sampling.
            learn_residual: Whether to learn the residual.
            state_normalization: Whether to use state normalization.
            sd3_timestep_sampling: Whether to use SD3-style timestep sampling.
            delta_std_name: Name of the delta standard deviation file.
            add_stochastic_noise: Whether to add stochastic noise in
                sampling.
            state_scaler_multiplier: Scalar multiplier applied to the loaded
                delta_std state scaler. Values > 1 make the normalized residuals
                smaller (easier to learn but under-dispersed); values < 1 make
                them larger. Default 1.0 (no change).
            optimizer_name: Which optimizer to build ("adamw" or "muon"),
                see OptimizerMixin.configure_optimizers.
            muon_lr: Learning rate for Muon-optimized parameters, when
                optimizer_name == "muon".
            adamw_lr: Learning rate for AdamW-optimized parameters (falls
                back to `lr` when not set).
            co2_linear_gain: Whether to apply CO2LinearGain's learned
                affine scaling to the CO2 forcing scalar before conditioning.
            co2_gain_mode: CO2LinearGain's output mode ("scalar" or
                per-channel), when co2_linear_gain is True.
            ema_decay: Exponential-moving-average decay for shadow weights,
                0 (default) disables EMA. Same mechanism as
                ForecastModule.ema_decay (model/forecast.py) -- see the
                on_fit_start/on_train_batch_end/on_save_checkpoint/
                on_load_checkpoint/on_validation_epoch_start hooks below.
            pf_conditioning: Toggles pushforward-style conditioning exposure
                (see _pf_drifted_batch docstring). False (default) preserves
                exact prior single-step behaviour.
            pf_n_steps: Number of frozen det_model steps to roll forward
                (no gradient) before drawing the single trainable
                flow-matching step, when pf_conditioning is active. Capped by
                however many pf_future_states targets the dataloader
                actually prepared (module.pf_n_steps in the dataloader
                config), same convention as ForecastModule.pf_n_steps.
            pf_warmup_steps: Global step at which pf_conditioning starts
                being applied (0 = from the start). Lets training begin on
                clean ground-truth conditioning before switching to
                pf-drifted conditioning once the model's single-step
                predictions are no longer near-random.
        """
        super().__init__()
        params = {k: v for k, v in locals().items() if k != "self"}
        self.__dict__.update(params)

        self.prep_model()
        self.det_model = None
        if load_deterministic_model:
            from ArchesClimate.model.base_module import (
                AvgModule,
                load_module,
            )

            if isinstance(load_deterministic_model, str):
                self.det_model, _ = load_module(load_deterministic_model, ckpt_fname=det_ckpt_fname)
            else:
                # load averaging model
                self.det_model = AvgModule(load_deterministic_model, ckpt_fname=det_ckpt_fname)

            for param in self.det_model.parameters():
                param.requires_grad = False
        self.scheduler_step = "default"
        if ckpt_path is not None:
            print("init from ckpt", ckpt_path)
            self.init_from_ckpt(ckpt_path)

        if scheduler == "heun":
            self.noise_scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=num_train_timesteps
            )

        elif scheduler == "flow":
            # stochastic_sampling (default False = old behaviour) must be set
            # here, at construction, for s_churn/s_tmin/s_tmax/s_noise passed
            # to scheduler.step() to have any effect at all -- with it False,
            # FlowMatchEulerDiscreteScheduler.step() takes the deterministic
            # ODE branch and s_churn is accepted but never referenced,
            # confirmed via two runs differing only in s_churn producing
            # bit-identical output.
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=num_train_timesteps,
                stochastic_sampling=stochastic_sampling,
            )

        # TODO(gclyne): fix this logic
        if (
            (len(cfg.depth_levels) > 40)
            or (self.delta_std_name == "cmip_delta_std_oro_new_masking_zg_log_5")
            or (self.delta_std_name == "cmip_delta_std_full_ozone_0")
            or (self.delta_std_name == "cmip_delta_std_oro_new_masking_zg_log_alt_loss_0")
            or (self.delta_std_name == "cmip_delta_std_filled_in_1")
        ):
            unsqueeze_dim = 1
        else:
            unsqueeze_dim = 1
        delta_path = ArchesClimate_stats_path.joinpath(self.delta_std_name)

        if self.load_deterministic_model:
            raw = torch.load(delta_path)
            if "per_lead_time" in raw:
                # Per-lead-time-indexed stats (see generate_delta_stats.py's
                # lead-time bucketing): different lead times (e.g. 1 vs 12
                # months) have genuinely different residual scales -- a
                # single pooled/shared scaler either compresses the
                # normalized target for short lead times or blows past
                # unit variance for long ones, both outside the regime the
                # flow-matching noise schedule assumes. Keyed by int lead
                # time (in months); _get_state_scaler below selects/stacks
                # per-sample at use time.
                self.state_scaler_by_lead_time = {
                    int(lt): self._build_state_scaler(stats, unsqueeze_dim, state_scaler_multiplier)
                    for lt, stats in raw["per_lead_time"].items()
                }
                self.state_scaler = None
            else:
                self.state_scaler = self._build_state_scaler(
                    raw, unsqueeze_dim, state_scaler_multiplier
                )
                self.state_scaler_by_lead_time = None

        self.inference_scheduler = deepcopy(self.noise_scheduler)

        self.test_outputs = []
        self.validation_samples = {}

    @staticmethod
    def _build_state_scaler(
        raw: dict, unsqueeze_dim: int, state_scaler_multiplier: float
    ) -> TensorDict:
        """Build a state_scaler TensorDict from one delta-std stats bucket.

        raw: {"surface": tensor, "level": tensor, "lev": tensor}, the same
        shape whether it's the whole (pooled) file or one lead-time's entry
        under a "per_lead_time" file.
        """
        scaler = TensorDict(
            {
                "surface": raw["surface"]
                .nanmean(axis=(-1, -2), keepdims=True)
                .unsqueeze(unsqueeze_dim),
                "level": raw["level"].nanmean(axis=(-1, -2), keepdims=True).unsqueeze(0),
                "lev": raw["lev"].nanmean(axis=(-1, -2), keepdims=True).unsqueeze(0),
            }
        )
        if state_scaler_multiplier != 1.0:
            scaler = scaler.apply(lambda x: x * state_scaler_multiplier)
        return scaler

    def _get_state_scaler(self, lead_time: torch.Tensor) -> TensorDict:
        """Return the state_scaler for normalizing/denormalizing residuals.

        With per-lead-time stats loaded (state_scaler_by_lead_time is not
        None), builds a batched TensorDict by picking each sample's own
        lead time's scaler and stacking along a new leading batch dim, so
        tensordict_apply's per-key op divides/multiplies each sample by
        the scaler matching its own lead time -- a batch can mix lead
        times (e.g. lead_times=[1, 12] both drawn within one batch).
        Falls back to the single shared scaler otherwise (prior
        behaviour, unchanged).
        """
        if self.state_scaler_by_lead_time is None:
            return self.state_scaler.to(self.device)
        fallback = next(iter(self.state_scaler_by_lead_time.values()))
        per_sample = [
            self.state_scaler_by_lead_time.get(int(lt), fallback)
            for lt in lead_time.reshape(-1).tolist()
        ]
        return torch.stack(per_sample, dim=0).to(self.device)

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
        """Swap in EMA weights for validation."""
        if hasattr(self, "_ema"):
            # backs training weights up to CPU, loads EMA weights into params
            self._ema.swap_with(list(self.named_parameters()))

    def forward(
        self,
        batch: TensorDict,
        noisy_next_state: TensorDict,
        timesteps: torch.Tensor,
        use_condition: bool | torch.Tensor = True,
        is_sampling: bool = False,
        forcing_mask: torch.Tensor | None = None,
    ) -> TensorDict:
        """Forward pass of the diffusion model.

        Args:
            batch: The input batch, containing the current state.
            noisy_next_state: The noisy version of the next state.
            timesteps: The diffusion timesteps.
            use_condition: Whether to use conditioning. Can be a boolean or a
              tensor for per-sample selection.
            is_sampling: Whether the model is in sampling mode.
            forcing_mask: Precomputed forcing keep-mask (B, F+1), e.g. drawn
                once by the caller and shared with det_model.forward so both
                models condition on the same set of visible forcing channels
                when learn_residual=="pred" (otherwise each forward call
                would independently sample its own mask via
                self.forcing_dropout, decoupling the residual target from
                what this model actually sees). None samples a fresh mask.

        Returns:
            The predicted output from the model.
        """
        del is_sampling  # Unused
        out = self.shared_forward_logic(
            batch,
            noisy_next_state,
            use_condition,
            timesteps,
            forcing_mask=forcing_mask,
        )
        return out

    def validation_step(self, batch: TensorDict, batch_idx: int) -> None:
        """Performs a validation step, generating samples and updating metrics.

        Args:
            batch: The validation batch.
            batch_idx: The index of the batch.
        """
        val_num_members = 5
        samples = [
            self.sample(batch, seed=j, disable_tqdm=True)["surface"]
            for j in tqdm(range(val_num_members))
        ]
        for metric in self.val_metrics:
            metric.to(self.device)
            # The prediction shape is (N_ensemble, B, ...), so we add a
            # dimension to the stacked samples.
            metric.update(
                batch["next_state"]["surface"][:, :, 0],
                torch.stack(samples)[None, :, :, 0],
            )
        self.validation_samples[batch_idx] = samples

    def on_validation_epoch_end(self) -> None:
        """Restore training weights after validation, then log metrics."""
        if hasattr(self, "_ema"):
            # restores training weights from the CPU backup made above
            self._ema.swap_with(list(self.named_parameters()))
        for metric in self.val_metrics:
            scores = metric.compute()
            for k, v in scores.items():
                scores[k] = v.mean()
            self.log_dict(scores, sync_dist=True)  # dont put on_epoch = True here
            metric.reset()
        self.validation_samples.clear()

    def multistep_loss(self, pred: TensorDict, gt: TensorDict) -> torch.Tensor:
        """Calculate loss for multi-step predictions.

        Args:
            pred: The predicted multi-step states.
            gt: The ground truth multi-step states.

        Returns:
            The calculated loss.
        """
        loss_coeffs = self.loss_coeffs.to(self.device)

        # Apply masks
        pred["surface"] = pred["surface"] * self.surface_mask.to(self.device)
        pred["level"] = pred["level"] * self.level_mask.to(self.device)
        pred["lev"] = pred["lev"] * self.lev_mask.to(self.device)
        gt["surface"] = gt["surface"] * self.surface_mask.to(self.device)
        gt["level"] = gt["level"] * self.level_mask.to(self.device)
        gt["lev"] = gt["lev"] * self.lev_mask.to(self.device)

        # Discount for multistep loss
        lead_iter = next(iter(gt.values())).shape[1]
        future_coeffs = (
            torch.tensor([1 / (1 + i) ** 2 for i in range(lead_iter)])
            .to(self.device)
            .view(1, -1, 1, 1, 1, 1)
        )
        discounted_coeffs = loss_coeffs.apply(lambda x: x * future_coeffs)

        weighted_error = (pred - gt).abs().pow(self.pow).mul(discounted_coeffs)
        return sum(weighted_error.mean().values())

    def loss(
        self,
        pred_v: TensorDict,
        target_v: TensorDict,
        implied_x1: TensorDict | None = None,
        gt_x1: TensorDict | None = None,
    ) -> torch.Tensor:
        """Computes the loss for the diffusion model.

        Args:
            pred_v: The predicted velocity field.
            target_v: The target velocity field.
            implied_x1: The implied reconstructed state x1 from the prediction.
            gt_x1: The ground truth state x1.

        Returns:
            The computed loss.
        """
        eps = 1e-8
        pred_v["surface"] = pred_v["surface"] * self.surface_mask.to(self.device)
        pred_v["level"] = pred_v["level"] * self.level_mask.to(self.device)
        pred_v["lev"] = pred_v["lev"] * self.lev_mask.to(self.device)
        target_v["surface"] = target_v["surface"] * self.surface_mask.to(self.device)
        target_v["level"] = target_v["level"] * self.level_mask.to(self.device)
        target_v["lev"] = target_v["lev"] * self.lev_mask.to(self.device)

        implied_x1["surface"] = implied_x1["surface"] * self.surface_mask.to(self.device)
        implied_x1["level"] = implied_x1["level"] * self.level_mask.to(self.device)
        implied_x1["lev"] = implied_x1["lev"] * self.lev_mask.to(self.device)
        gt_x1["surface"] = gt_x1["surface"] * self.surface_mask.to(self.device)
        gt_x1["level"] = gt_x1["level"] * self.level_mask.to(self.device)
        gt_x1["lev"] = gt_x1["lev"] * self.lev_mask.to(self.device)

        # A. Standard Flow Matching Loss (on velocity field)
        # Use epsilon to prevent NaN if self.pow < 1
        mse_v = sum(((pred_v - target_v).abs() + eps).pow(self.pow).mean().values())

        # A2. Amplitude-matching loss (on RECONSTRUCTED residual magnitude).
        # Targets the polar amplitude-overshoot pattern found in rollouts
        # (predicted seasonal swings larger than target's, in both directions
        # -- overshooting summer peaks and winter troughs alike) independently
        # of the main per-pixel MSE terms above: those penalise being at the
        # wrong VALUE, this penalises the predicted residual being the wrong
        # SIZE, which per-pixel MSE can miss if errors are spatially
        # scattered. implied_x1/gt_x1 are the reconstructed (denoised)
        # residual around the deterministic model's prediction, so this term
        # is independently toggleable via amplitude_loss_weight (default 0.0
        # = no effect, preserves prior behaviour exactly).
        amplitude_loss_weight = getattr(self, "amplitude_loss_weight", 0.0)
        loss_amp = torch.tensor(0.0, device=mse_v.device)
        if amplitude_loss_weight > 0.0:
            amp_diff = implied_x1.apply(torch.abs) - gt_x1.apply(torch.abs)
            loss_amp = sum(amp_diff.pow(2).mean().values())

        return mse_v + amplitude_loss_weight * loss_amp

    def sample_rollout(
        self,
        batch: TensorDict,
        index: int,
        test_dataset: Any,
        *args,
        lead_time_months: int = 1,
        disable_tqdm: bool = False,
        debug: bool = True,
        return_format: str = "tensordict",
        seed: int,
        out_dir: str,
        ensemble_member_index: int,
        flat_forcings: bool = False,
        num_seeds: int = 1,
        zero_spatial_forcing_indices: list[int] | None = None,
        zero_non_spatial_forcing_indices: list[int] | None = None,
        pi_spatial_values: torch.Tensor | None = None,
        pi_non_spatial_values: torch.Tensor | None = None,
        clamp_spatial_forcing_indices: list[int] | None = None,
        clamp_spatial_forcing_std: float | None = None,
        clamp_polar_tas_std: float | None = None,
        clamp_polar_n_rows: int = 5,
        teacher_force: bool = False,
        **kwargs,
    ) -> list[TensorDict] | TensorDict:
        """Generates a multi-step rollout prediction.

        Args:
            batch: The initial state batch.
            index: The starting index in the dataset.
            test_dataset: The dataset to use for future forcings.
            *args: Additional arguments for the sample method.
            lead_time_months: The number of months to roll out.
            disable_tqdm: Whether to disable the progress bar.
            debug: If True, saves only surface data.
            return_format: 'tensordict' or 'list'.
            seed: Random seed for sampling.
            out_dir: Directory to save rollout files.
            ensemble_member_index: Index of the ensemble member.
            flat_forcings: Whether to use flat forcings.
            num_seeds: Number of independent noise seeds to sample per
                ensemble member (see energy_score_seed/batch_seeds path).
            zero_spatial_forcing_indices: Spatial-forcing channel indices to
                zero out (or substitute via pi_spatial_values) at every
                rollout step, for forcing-ablation experiments.
            zero_non_spatial_forcing_indices: Non-spatial-forcing channel
                indices to zero out (or substitute via pi_non_spatial_values)
                at every rollout step, for forcing-ablation experiments.
            pi_spatial_values: If given, substitute these normalized
                pre-industrial-level values (shape (F, 144, 144), see
                model/forecast.py's load_pi_forcing_values) for
                zero_spatial_forcing_indices instead of the learned
                null_spatial_map token -- a physical "held at pre-industrial
                level" counterfactual rather than a "channel absent" one.
                Mirrors ForecastModule.forward_multistep's pi_spatial_values.
            pi_non_spatial_values: Same, for zero_non_spatial_forcing_indices
                (shape (F_ns,)), substituted instead of zero.
            clamp_spatial_forcing_indices: Channel indices in spatial
                forcings to clip to +/- clamp_spatial_forcing_std, applied
                each rollout step (see ForecastModule's rollout loop).
            clamp_spatial_forcing_std: Symmetric clamp bound, in standard
                deviations, for clamp_spatial_forcing_indices.
            clamp_polar_tas_std: If set, clip the model's own predicted
                surface tas (channel 0) to +/- this many normalized standard
                deviations, applied each rollout step to the model's own
                output before it's fed back in as the next step's input --
                same mechanism as clamp_spatial_forcing_indices, but on the
                predicted STATE at the poles rather than an input forcing.
                Targets the amplitude-overshoot pattern documented for this
                model at the South/North Pole (predicted seasonal swings
                larger than target's, in both directions).
            clamp_polar_n_rows: Number of grid rows from each pole (South:
                rows 0..n-1, North: rows -n..-1) to clamp. Default 5, matching
                the "South Pole"/"North Pole" band used in the polar-bias
                analysis. Only used when clamp_polar_tas_std is set.
            teacher_force: If True, feed the real ground-truth state
                (surface/level/lev) back in as next step's conditioning
                input instead of this step's own sample -- isolates a
                single-step sample's quality from compounding autoregressive
                exposure bias. The sample itself is unaffected and is still
                what gets appended to preds_future/saved to disk; only what
                loop_batch["state"] holds going into the *next* iteration
                changes. Forcings (spatial/non_spatial) are unaffected by
                this flag -- they are always read from test_dataset already,
                teacher-forced or not.
            **kwargs: Additional keyword arguments for the sample method.

        Returns:
            The generated rollout as a stacked TensorDict or a list of
            TensorDicts.
        """
        torch.set_grad_enabled(False)
        # preds_future shape per entry: [num_seeds, ...] after tiling
        preds_future = []
        loop_batch = {k: v.to(self.device) for k, v in batch.items()}

        # Tile initial batch so all seeds run in parallel
        if num_seeds > 1:
            loop_batch = {
                k: v.repeat(num_seeds, *([1] * (v.ndim - 1))) for k, v in loop_batch.items()
            }

        # put in t0 for sanity check that we can use to compare
        # against test/persistence, i.e. do we start from the same
        preds_future.append(loop_batch["state"])
        # list(filter_unique_third_tuple_values(deepcopy(test_dataset.id2pt)))

        for i in tqdm(range(lead_time_months), disable=disable_tqdm):
            loop_batch = {k: v.to(self.device) for k, v in loop_batch.items()}
            # num_seeds=1 here because the batch dim already has the seeds tiled
            sample = self.sample(
                loop_batch,
                *args,
                **kwargs,
                disable_tqdm=True,
                seed=seed * 100000 + i,
                cf_guidance=self.cf_guidance,
                num_seeds=1,
            )
            sample.batch_size = [num_seeds]
            if self.load_prev > 1:
                loop_batch["prev_state"] = torch.stack(
                    [loop_batch["state"], loop_batch["prev_state"][:, 0]], dim=1
                )
            else:
                loop_batch["prev_state"] = loop_batch["state"].copy()

            sample["level"][:, -1, 0, :, :] = torch.clamp(
                sample["level"][:, -1, 0, :, :], max=8, min=-8
            )
            if clamp_polar_tas_std is not None:
                n = clamp_polar_n_rows
                sample["surface"][:, 0, ..., :n, :] = torch.clamp(
                    sample["surface"][:, 0, ..., :n, :],
                    min=-clamp_polar_tas_std,
                    max=clamp_polar_tas_std,
                )
                sample["surface"][:, 0, ..., -n:, :] = torch.clamp(
                    sample["surface"][:, 0, ..., -n:, :],
                    min=-clamp_polar_tas_std,
                    max=clamp_polar_tas_std,
                )
            loop_batch["state"] = TensorDict(sample)
            loop_batch["state"].batch_size = [num_seeds]
            print(sample["surface"])
            index = index + 1
            # state_only=True: only next_batch's forcings are used below --
            # skips building next_state, which would otherwise require a
            # dataset lookahead of up to max(lead_times) months (see
            # forward_multistep's state_only usage in model/forecast.py).
            next_batch = test_dataset.__getitem__(index, state_only=True)

            next_batch = {k: v.to(self.device) for k, v in next_batch.items()}

            loop_batch["timestamp"] = next_batch["timestamp"].expand(num_seeds)
            # Broadcast forcings to all seeds in the batch
            loop_batch["state"]["non_spatial_forcings"] = next_batch["state"][
                "non_spatial_forcings"
            ][None].expand(num_seeds, -1)
            loop_batch["state"]["spatial_forcings"] = next_batch["state"]["spatial_forcings"][
                None
            ].expand(num_seeds, -1, -1, -1)
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
            # Snapshot the model's own (masked) sample for saving/evaluation
            # *before* any teacher_force substitution below -- what gets
            # scored/saved is always this step's actual prediction, never
            # the ground truth fed back in for the next step's conditioning.
            preds_future.append(loop_batch["state"].clone())
            if teacher_force:
                # Real ground-truth surface/level/lev at this step instead
                # of this step's own sample, for what feeds into the *next*
                # sample() call as conditioning -- next_batch["state"]
                # already holds it (state_only=True still populates "state"
                # fully, see CMIPBaseDataset.__getitem__). Isolates a single
                # step's sample quality from compounding autoregressive
                # exposure bias; the snapshot just appended above is
                # unaffected, so per-step prediction quality is still what
                # gets evaluated.
                for _k in ("surface", "level", "lev"):
                    _gt = next_batch["state"][_k]
                    loop_batch["state"][_k] = (
                        _gt[None].expand(num_seeds, *([-1] * _gt.ndim)).clone()
                    )
                loop_batch["state"]["surface"] *= self.surface_mask.to(self.device)
                loop_batch["state"]["level"] *= self.level_mask.to(self.device)
                loop_batch["state"]["lev"] *= self.lev_mask.to(self.device)
            if "pred_state" in loop_batch:
                del loop_batch["pred_state"]
            # reset for longer rollouts
            if (
                (((len(preds_future) % 120) == 0) and i > 0)
                or (i == 1017)
                or (i == lead_time_months - 1)
            ):
                # out["surface"] shape: [num_seeds, T, ...]
                out = torch.stack(preds_future, dim=1)

                for s in range(num_seeds):
                    s_seed = seed + s
                    torch.save(
                        out["surface"][s],
                        f"{out_dir}/rollout_surface_{i}_{ensemble_member_index}_{s_seed}.pt",
                    )
                    if not debug:
                        torch.save(
                            out["level"][s],
                            f"{out_dir}/rollout_level_{i}_{ensemble_member_index}_{s_seed}.pt",
                        )
                        torch.save(
                            out["lev"][s],
                            f"{out_dir}/rollout_lev_{i}_{ensemble_member_index}_{s_seed}.pt",
                        )
                preds_future = []

        if return_format == "list":
            return preds_future
        if len(preds_future) > 0:
            preds_future = torch.stack(preds_future, dim=1)
        return preds_future

    def forward_multistep(self, batch: TensorDict, iters: int) -> TensorDict:
        """Performs a multi-step forward pass for training.

        This method is differentiable and used for multi-step finetuning.

        Args:
            batch: The input batch.
            iters: The number of steps to unroll.

        Returns:
            A TensorDict of stacked future predictions.
        """
        torch.set_grad_enabled(True)
        preds_future = []
        loop_batch = {k: v for k, v in batch.items()}
        # Forcing mask for step 0 is sampled inside sample_trainable itself
        # (None here); for step i>0 it's set below, right after computing
        # that step's pred_state, so both stay consistent -- see the
        # forcing_mask docstring on sample_trainable.
        forcing_mask = None

        for i in range(iters):
            # Use a trainable version of the sample method
            sample = self.sample_trainable(
                loop_batch,
                disable_tqdm=True,
                seed=None,  # No fixed seed during training
                cf_guidance=self.cf_guidance,
                forcing_mask=forcing_mask,
            )

            preds_future.append(sample.copy())
            # --- Prepare for the next step ---
            # The predicted sample becomes the new state
            if self.load_prev > 1:
                loop_batch["prev_state"] = torch.stack(
                    [loop_batch["state"], loop_batch["prev_state"][:, 0]], dim=1
                )
            else:
                # Make sure not to use .copy() which detaches gradients
                loop_batch["prev_state"] = loop_batch["state"]

            loop_batch["state"] = sample

            # Get future forcings and timestamp from the input batch
            next_state_data = batch["future_states"][:, i]
            next_timestamp = batch["future_timestamps"][:, i]

            loop_batch["timestamp"] = next_timestamp
            # Add forcings to the new state for the next prediction step
            loop_batch["state"]["non_spatial_forcings"] = next_state_data["non_spatial_forcings"]
            loop_batch["state"]["spatial_forcings"] = next_state_data["spatial_forcings"]

            # Apply masks
            loop_batch["state"]["surface"] *= self.surface_mask.to(self.device)
            loop_batch["prev_state"]["surface"] *= self.surface_mask.to(self.device)
            loop_batch["state"]["level"] *= self.level_mask.to(self.device)
            loop_batch["prev_state"]["level"] *= self.level_mask.to(self.device)
            loop_batch["state"]["lev"] *= self.lev_mask.to(self.device)
            loop_batch["prev_state"]["lev"] *= self.lev_mask.to(self.device)

            # Recompute deterministic prediction if needed for the next step
            if (self.learn_residual == "pred") or (self.learn_residual == "partial"):
                if hasattr(self, "forcing_dropout"):
                    bs = loop_batch["state"]["non_spatial_forcings"].shape[0]
                    forcing_mask = self.forcing_dropout.sample_mask(bs, loop_batch["state"].device)
                with torch.no_grad():
                    loop_batch["pred_state"] = self.det_model.forward(
                        loop_batch, use_condition=True, forcing_mask=forcing_mask
                    ).detach()

        # Stack predictions along the time dimension
        preds_future = torch.stack(preds_future, dim=1)
        return preds_future

    def _pf_drifted_batch(self, batch: TensorDict) -> TensorDict | None:
        """Roll the frozen det_model forward pf_n_steps (no grad) from this batch's true state.

        Returns a batch with "state"/"prev_state" swapped for that drifted
        estimate and "next_state"/"timestamp"
        advanced to match -- so the one trainable flow-matching step this
        batch takes afterward conditions on a realistically-compounded-error
        state instead of clean ground truth.

        This is pushforward-style *conditioning exposure*, not multi-step
        finetuning of the flow model itself: nothing in the pf_n_steps det
        rollout has gradient (det_model is frozen and already called with
        torch.no_grad() everywhere else in this file), so cost is
        pf_n_steps extra cheap forward passes through the frozen det model,
        not pf_n_steps repetitions of the expensive flow-matching sampler.
        The idea is to close the gap between the clean states the flow
        model sees during ordinary single-step training and the
        autoregressively-drifted states it actually has to condition on a
        few hundred steps into a real rollout -- see the pf_conditioning
        discussion this was designed against.

        Returns None if this batch doesn't carry enough pf_future_states
        (dataloader's module.pf_n_steps must be >= self.pf_n_steps) for the
        caller to fall back to plain single-step training on it.
        """
        has_pf = batch.get("has_pf", None)
        if self.det_model is None or has_pf is None or not bool(has_pf.all()):
            return None
        pf_targets = batch.get("pf_future_states", None)
        if pf_targets is None:
            return None
        pf_n_steps = min(self.pf_n_steps, pf_targets.shape[1])
        if pf_n_steps < 1:
            return None

        surface_mask = self.surface_mask.to(self.device)
        level_mask = self.level_mask.to(self.device)
        lev_mask = self.lev_mask.to(self.device)

        with torch.no_grad():
            cur_state = batch["state"]
            prev_state = batch["prev_state"]
            cur_timestamp = batch["timestamp"]
            lead_time = batch["lead_time"]
            next_true = batch["next_state"]  # t+1, forcings still attached

            for k in range(pf_n_steps):
                det_batch = {
                    "state": cur_state,
                    "prev_state": prev_state,
                    "lead_time": lead_time,
                    "timestamp": cur_timestamp,
                }
                pred_k = self.det_model.forward(det_batch, use_condition=True).detach()
                pred_k["surface"] = pred_k["surface"] * surface_mask
                pred_k["level"] = pred_k["level"] * level_mask
                pred_k["lev"] = pred_k["lev"] * lev_mask

                # Forcings for the timestep pred_k represents are known
                # externally (not predicted): step 0's live on next_true
                # (t+1), later steps read them off pf_future_states -- same
                # convention as ForecastModule's own pushforward loop.
                target_k = next_true if k == 0 else pf_targets[:, k - 1]
                pred_k["spatial_forcings"] = target_k["spatial_forcings"]
                pred_k["non_spatial_forcings"] = target_k["non_spatial_forcings"]

                prev_state = cur_state
                cur_state = pred_k
                cur_timestamp = cur_timestamp + lead_time

        # pf_future_states[:,0] is t+2, so index pf_n_steps-1 is
        # t+(pf_n_steps+1) -- exactly one lead_time past cur_state's
        # (drifted) t+pf_n_steps.
        true_next = pf_targets[:, pf_n_steps - 1]

        drifted = dict(batch)
        drifted["state"] = cur_state
        drifted["prev_state"] = prev_state
        drifted["next_state"] = true_next
        drifted["timestamp"] = cur_timestamp
        return drifted

    def training_step(self, batch: TensorDict, batch_idx: int) -> torch.Tensor:
        """Perform a single training step.

        Can be either a single-step diffusion loss or a multi-step
        sequence loss, depending on the presence of 'future_states'
        in the batch.

        Args:
            batch: The training batch.
            batch_idx: The index of the batch.
        """
        if (
            self.pf_conditioning
            and "future_states" not in batch
            and self.global_step >= self.pf_warmup_steps
        ):
            drifted = self._pf_drifted_batch(batch)
            if drifted is not None:
                batch = drifted

        device, bs = batch["state"].device, batch["state"].shape[0]
        del batch["next_state"]["non_spatial_forcings"]
        del batch["next_state"]["spatial_forcings"]
        del batch["prev_state"]["non_spatial_forcings"]
        del batch["prev_state"]["spatial_forcings"]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=device,
        ).long()

        if "future_states" not in batch:
            # --- Standard Single-Step Diffusion Training ---
            noise = torch.randn_like(batch["next_state"])
            next_state = batch["next_state"]
            use_condition = torch.rand((bs,), device=device) > self.uncond_proba

            # Shared forcing keep-mask: sampled once here (from this model's own
            # ForcingDropout) and reused for both det_model.forward below and
            # self.forward further down, so the residual target and this
            # model's own conditioning agree on which forcings were visible.
            forcing_mask = None
            if self.learn_residual == "default":
                next_state = batch["next_state"] - batch["state"]

            elif self.learn_residual == "pred":
                if "pred_state" not in batch:
                    if hasattr(self, "forcing_dropout"):
                        forcing_mask = self.forcing_dropout.sample_mask(bs, device)
                    with torch.no_grad():
                        batch["pred_state"] = self.det_model.forward(
                            batch, use_condition=True, forcing_mask=forcing_mask
                        ).detach()
                next_state = batch["next_state"] - batch["pred_state"]

            if self.state_normalization:
                next_state = tensordict_apply(
                    torch.div, next_state, self._get_state_scaler(batch["lead_time"])
                )

            if self.scheduler == "flow":
                # weighting scheme: logit normal
                u = torch.normal(mean=0, std=1, size=(bs,), device="cpu").sigmoid()
                indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
                if self.sd3_timestep_sampling:
                    u = torch.normal(mean=0, std=1, size=(bs,), device="cpu").sigmoid()
                    indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
                else:
                    indices = timesteps.cpu()
                timesteps = self.noise_scheduler.timesteps[indices].to(device)

                schedule_timesteps = self.noise_scheduler.timesteps.to(device)
                sigmas = self.noise_scheduler.sigmas.to(
                    device=device,
                    dtype=next(iter(batch["state"].values())).dtype,
                )

                step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
                sigma = sigmas[step_indices].flatten()[:, None, None, None, None]  # shape (bs,)

                # sigma=0: all data, sigma=1: all noise
                noisy_next_state = noise.apply(lambda x: x * sigma) + next_state.apply(
                    lambda x: x * (1.0 - sigma)
                )
                target_state = noise - next_state
                if self.prediction_type == "sample":
                    target_state = next_state

            # create uncond
            pred = self.forward(
                batch,
                noisy_next_state,
                timesteps,
                use_condition.float(),
                forcing_mask=forcing_mask,
            )
            # compute loss
            implied_next_state = noisy_next_state - (sigma[:, 0, 0, 0, 0] * pred)
            loss = self.loss(
                pred,
                target_state,
                implied_x1=implied_next_state,
                gt_x1=next_state,
            )
            self.mylog(loss=loss)
            return loss
        else:
            # --- Multi-Step Finetuning ---
            lead_iter = batch["future_states"].shape[1]

            # Generate a sequence of predictions
            pred_future_states = self.forward_multistep(
                batch,
                iters=lead_iter,
            )

            # Compute loss over the whole sequence
            del batch["future_states"]["non_spatial_forcings"]
            del batch["future_states"]["spatial_forcings"]
            loss = self.multistep_loss(pred_future_states, batch["future_states"])

            self.mylog(multistep_loss=loss, lead_iter=lead_iter)
            return loss

    def on_after_backward(self) -> None:
        """Hook to check for unused parameters."""
        for name, param in self.named_parameters():
            if param.grad is None and param.requires_grad:
                rank_zero_warn(f"Parameter '{name}' did not receive a gradient.")

    def sample(
        self,
        batch: TensorDict,
        seed: int | None = None,
        num_steps: int | None = None,
        cf_guidance: float | None = None,
        disable_tqdm: bool = False,
        rescale: float = 1.0,
        num_seeds: int = 1,
        **kwargs: Any,
    ) -> TensorDict:
        """Generates a sample from the diffusion model.

        Args:
            batch: The input batch containing the initial state.
            seed: Random seed for reproducibility.
            num_steps: Number of inference steps. Overrides the default.
            cf_guidance: Classifier-free guidance scale. Overrides the default.
            disable_tqdm: Whether to disable the progress bar.
            rescale: Rescaling factor (unused).
            num_seeds: Number of independent noise seeds to sample.
            **kwargs: Additional arguments for the scheduler step.

        Returns:
            The final denoised state as a TensorDict.
        """
        if cf_guidance is None:
            cf_guidance = self.cf_guidance
        scheduler = self.inference_scheduler
        num_steps = num_steps or self.num_inference_steps
        scheduler.set_timesteps(num_steps)
        if seed is not None:
            torch.manual_seed(seed)

        # get args to scheduler
        if self.scheduler == "flow":
            scheduler_kwargs = dict(s_churn=self.s_churn)
        else:
            # TODO: put some churning here
            scheduler_kwargs = {}

        loop_batch = {
            k: v.clone().detach() for k, v in batch.items() if "next" not in k
        }  # ensures no data leakage

        if ("pred_state" not in loop_batch) and (self.learn_residual == "pred"):
            with torch.no_grad():
                loop_batch["pred_state"] = self.det_model.forward(
                    loop_batch, use_condition=True
                ).detach()

        # Tile batch along dim 0 so each seed gets its own independent noise
        # from torch.randn_like, giving num_seeds parallel denoising trajectories.
        if num_seeds > 1:
            loop_batch = {
                k: v.repeat(num_seeds, *([1] * (v.ndim - 1))) for k, v in loop_batch.items()
            }

        noisy_state = loop_batch["state"].apply(torch.randn_like)
        scale_input_noise = self.scale_input_noise
        if scale_input_noise is not None:
            noisy_state = noisy_state * scale_input_noise
        v_fields = []
        with torch.no_grad():
            for t in tqdm(scheduler.timesteps, disable=disable_tqdm):
                pred = self.forward(
                    loop_batch,
                    noisy_state,
                    timesteps=torch.tensor([t]).to(self.device),
                    use_condition=True,
                    is_sampling=True,
                )
                # A cf_guidance of 1 means use the forcings as given, and we
                # dont need to make another forward pass.
                if cf_guidance != 1.0:
                    uncond_pred = self.forward(
                        loop_batch,
                        noisy_state,
                        timesteps=torch.tensor([t]).to(self.device),
                        use_condition=False,
                        is_sampling=True,
                    )
                    epsilon = pred - uncond_pred
                    pred = epsilon.apply(lambda x: cf_guidance * x) + uncond_pred
                if self.save_v_fields:
                    v_fields.append(pred)
                step_index = getattr(scheduler, "_step_index", None)

                def scheduler_step(*args, **kwargs) -> Any:
                    out = scheduler.step(*args, **kwargs)
                    if hasattr(scheduler, "_step_index"):
                        scheduler._step_index = step_index
                    return out.prev_sample

                # The scheduler expects a flat tensor, so we concatenate the
                # variables in the TensorDict.
                pred_stacked = torch.concatenate(
                    [
                        pred["surface"].squeeze(2),
                        pred["level"].reshape(
                            -1,
                            len(self.cfg.level_variables) * len(self.cfg.pressure_levels),
                            self.lat_dim,
                            self.lon_dim,
                        ),
                        pred["lev"].squeeze(1),
                    ],
                    dim=1,
                )
                noisy_state_stacked = torch.concatenate(
                    [
                        noisy_state["surface"].squeeze(2),
                        noisy_state["level"].reshape(
                            -1,
                            len(self.cfg.level_variables) * len(self.cfg.pressure_levels),
                            self.lat_dim,
                            self.lon_dim,
                        ),
                        noisy_state["lev"].squeeze(1),
                    ],
                    dim=1,
                )
                out = scheduler_step(pred_stacked, t, noisy_state_stacked, **scheduler_kwargs)

                # After getting 'out' from scheduler_step (which is
                # noisy_state_next), add stochastic noise if enabled.
                if self.add_stochastic_noise and t > 0.1:
                    noise_level = 0.005  # Small epsilon
                    out = out + torch.randn_like(out) * noise_level

                # Reconstruct the TensorDict from the flat tensor output of the
                # scheduler
                num_surface_vars = len(self.cfg.surface_variables)
                num_level_vars = len(self.cfg.level_variables)
                num_pressure_levels = len(self.cfg.pressure_levels)
                num_depth_levels = len(self.cfg.depth_levels)

                surface_end = num_surface_vars
                level_end = surface_end + num_level_vars * num_pressure_levels
                lev_end = level_end + num_depth_levels

                noisy_state["surface"] = out[:, :surface_end].unsqueeze(2)

                noisy_state["level"] = out[:, surface_end:level_end].reshape(
                    -1,
                    num_level_vars,
                    num_pressure_levels,
                    self.lat_dim,
                    self.lon_dim,
                )

                noisy_state["lev"] = out[:, level_end:lev_end].reshape(
                    -1,
                    1,
                    num_depth_levels,
                    self.lat_dim,
                    self.lon_dim,
                )

                # at the end
                if step_index is not None:
                    scheduler._step_index = step_index + 1

        if self.save_v_fields:
            torch.save(torch.stack(v_fields), f"v_fields_{seed}.pt")

        final_state = noisy_state.detach()
        del final_state["spatial_forcings"]
        del final_state["non_spatial_forcings"]

        if self.state_normalization:
            final_state = tensordict_apply(
                torch.mul, final_state, self._get_state_scaler(loop_batch["lead_time"])
            )

        if self.learn_residual == "default":
            final_state = loop_batch["state"] + final_state
        elif (
            (self.learn_residual == "pred")
            or (self.learn_residual == "partial")
            or (self.learn_residual == "partial_with_temp")
        ):
            final_state = loop_batch["pred_state"] + final_state

        return final_state

    def sample_trainable(
        self,
        batch,
        seed=None,
        num_steps: int | None = None,
        cf_guidance: float | None = None,
        disable_tqdm: bool = True,
        rescale: float = 1.0,
        forcing_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> TensorDict:
        """Generates a sample from the diffusion model, with gradients enabled.

        This is used for differentiable multi-step training.

        Args:
            batch: The input batch containing the initial state.
            seed: Random seed for reproducibility.
            num_steps: Number of inference steps. Overrides the default.
            cf_guidance: Classifier-free guidance scale. Overrides the default.
            disable_tqdm: Whether to disable the progress bar.
            rescale: Rescaling factor (unused).
            forcing_mask: Precomputed forcing keep-mask (B, F+1). If None and
                forcing_dropout exists, one is sampled here and reused for
                det_model.forward (computing pred_state) and every
                self.forward call across the ODE integration loop below --
                keeping the residual target and this model's own
                conditioning consistent within a single sample, and across
                the whole trajectory rather than flickering per scheduler
                step. Callers running a multi-step rollout (forward_multistep)
                should pass their own draw so it also matches the pred_state
                they compute for the *next* step.
            **kwargs: Additional arguments for the scheduler step.

        Returns:
            The final denoised state as a TensorDict, with gradients attached.
        """
        if cf_guidance is None:
            cf_guidance = self.cf_guidance
        scheduler = self.inference_scheduler
        num_steps = num_steps or self.num_inference_steps
        scheduler.set_timesteps(num_steps)
        if seed is not None:
            torch.manual_seed(seed)

        # get args to scheduler
        if self.scheduler == "flow":
            scheduler_kwargs = dict(s_churn=self.s_churn)
        else:
            # TODO: put some churning here
            scheduler_kwargs = {}

        if forcing_mask is None and hasattr(self, "forcing_dropout"):
            bs = batch["state"]["non_spatial_forcings"].shape[0]
            forcing_mask = self.forcing_dropout.sample_mask(bs, batch["state"].device)

        if ("pred_state" not in batch) and (self.learn_residual == "pred"):
            with torch.no_grad():
                batch["pred_state"] = self.det_model.forward(
                    batch, use_condition=True, forcing_mask=forcing_mask
                ).detach()

        noisy_state = batch["state"].apply(torch.randn_like)  # amazing it works
        scale_input_noise = self.scale_input_noise
        if scale_input_noise is not None:
            noisy_state = noisy_state * scale_input_noise
        loop_batch = {k: v for k, v in batch.items() if "next" not in k}

        for t in scheduler.timesteps:
            pred = self.forward(
                loop_batch,
                noisy_state,
                timesteps=torch.tensor([t]).to(self.device),
                use_condition=True,
                is_sampling=True,
                forcing_mask=forcing_mask,
            )
            if cf_guidance != 1.0:
                uncond_pred = self.forward(
                    loop_batch,
                    noisy_state,
                    timesteps=torch.tensor([t]).to(self.device),
                    use_condition=False,
                    is_sampling=True,
                    forcing_mask=forcing_mask,
                )
                epsilon = pred - uncond_pred
                pred = epsilon.apply(lambda x: cf_guidance * x) + uncond_pred

            step_index = getattr(scheduler, "_step_index", None)

            def scheduler_step(*args, **kwargs) -> Any:
                out = scheduler.step(*args, **kwargs)
                if hasattr(scheduler, "_step_index"):
                    scheduler._step_index = step_index
                return out.prev_sample

            # The scheduler expects a flat tensor, so we concatenate the
            # variables in the TensorDict.
            pred_stacked = torch.concatenate(
                [
                    pred["surface"].squeeze(2),
                    pred["level"].reshape(
                        -1,
                        len(self.cfg.level_variables) * len(self.cfg.pressure_levels),
                        self.lat_dim,
                        self.lon_dim,
                    ),
                    pred["lev"].squeeze(1),
                ],
                dim=1,
            )
            noisy_state_stacked = torch.concatenate(
                [
                    noisy_state["surface"].squeeze(2),
                    noisy_state["level"].reshape(
                        -1,
                        len(self.cfg.level_variables) * len(self.cfg.pressure_levels),
                        self.lat_dim,
                        self.lon_dim,
                    ),
                    noisy_state["lev"].squeeze(1),
                ],
                dim=1,
            )
            out = scheduler_step(pred_stacked, t, noisy_state_stacked, **scheduler_kwargs)

            if self.add_stochastic_noise and t > 0.1:
                noise_level = 0.005
                out = out + torch.randn_like(out) * noise_level

            # Reconstruct the TensorDict from the flat tensor output of the
            # scheduler
            num_surface_vars = len(self.cfg.surface_variables)
            num_level_vars = len(self.cfg.level_variables)
            num_pressure_levels = len(self.cfg.pressure_levels)
            num_depth_levels = len(self.cfg.depth_levels)

            surface_end = num_surface_vars
            level_end = surface_end + num_level_vars * num_pressure_levels
            lev_end = level_end + num_depth_levels

            noisy_state["surface"] = out[:, :surface_end].unsqueeze(2)

            noisy_state["level"] = out[:, surface_end:level_end].reshape(
                -1,
                num_level_vars,
                num_pressure_levels,
                self.lat_dim,
                self.lon_dim,
            )

            noisy_state["lev"] = out[:, level_end:lev_end].reshape(
                -1, 1, num_depth_levels, self.lat_dim, self.lon_dim
            )

            if step_index is not None:
                scheduler._step_index = step_index + 1

        final_state = noisy_state
        del final_state["spatial_forcings"]
        del final_state["non_spatial_forcings"]

        if self.state_normalization:
            final_state = tensordict_apply(
                torch.mul, final_state, self._get_state_scaler(batch["lead_time"])
            )

        if self.learn_residual == "default":
            final_state = batch["state"] + final_state
        elif (
            (self.learn_residual == "pred")
            or (self.learn_residual == "partial")
            or (self.learn_residual == "partial_with_temp")
        ):
            final_state = batch["pred_state"] + final_state

        return final_state

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
        num_perturbations_per_member: int = 12,
        debug: bool = True,
        seed_index: int | None = None,
        zero_spatial_forcing_indices: list[int] | None = None,
        zero_non_spatial_forcing_indices: list[int] | None = None,
        zero_forcing_source: str = "null",
        clamp_spatial_forcing_indices: list[int] | None = None,
        clamp_spatial_forcing_std: float | None = None,
        clamp_polar_tas_std: float | None = None,
        clamp_polar_n_rows: int = 5,
        save_all_vars: bool = False,
        energy_score_noise_scale: float = 1.0,
        teacher_force: bool = False,
        batch_seeds: bool = False,
    ) -> None:
        """Generates and saves multiple long-term rollouts.

        Args:
            cfg: The configuration object.
            batch_index: The base index for seeding (int).
            test_dataset: The dataset for evaluation.
            target_name: The name of the target experiment.
            out_dir: The directory to save rollout files.
            start_member: The starting ensemble member index.
            end_member: The ending ensemble member index.
            flat_forcings: Whether to use flat forcings (bool).
            num_rollout_steps: The number of steps in each rollout.
            num_perturbations_per_member: Number of seeds per member.
                When seed_index is None, all seeds are batched together.
                When seed_index is set, only that single seed is run.
            debug: If True, runs in debug mode.
            seed_index: If set, run only this seed (0-indexed) instead of
                all seeds batched together. Use this to parallelise seeds
                across separate jobs.
            zero_spatial_forcing_indices: Channel indices to zero in spatial forcings.
            zero_non_spatial_forcing_indices: Channel indices to zero in non-spatial forcings.
            clamp_spatial_forcing_indices: Channel indices in spatial forcings
                to clip to +/- clamp_spatial_forcing_std, applied each rollout
                step to the *next* step's forcings before they're fed back in
                (same semantics as ForecastModule.generate_rollouts'
                clamp_spatial_forcing_indices -- see the black-carbon/AIBCM
                SSP4-3.4 spike note in CLAUDE.md). Values are in normalized
                (post CMIPBaseDataset.normalize) standard-deviation units.
            clamp_spatial_forcing_std: Symmetric clamp bound, in standard
                deviations, for clamp_spatial_forcing_indices. Required if
                clamp_spatial_forcing_indices is given.
            clamp_polar_tas_std: If set, clip the model's own predicted
                surface tas (channel 0) to +/- this many normalized standard
                deviations at the poles, applied each rollout step (see
                sample_rollout's clamp_polar_tas_std).
            clamp_polar_n_rows: Number of grid rows from each pole to clamp.
                Only used when clamp_polar_tas_std is set.
            zero_forcing_source: "null" (default) substitutes the learned
                null_spatial_map "channel absent" value. "pi" substitutes
                pre-industrial-level forcing values instead -- a physical
                "held at pre-industrial level" counterfactual (see
                model/forecast.py's load_pi_forcing_values; falls back to a
                historical-first-year proxy for models with no piControl run
                processed, e.g. CanESM5).
            save_all_vars: No-op for this class -- sample_rollout already
                saves level/lev whenever debug=False (see the `debug` check
                around the rollout_level_*.pt/rollout_lev_*.pt torch.save
                calls below), so there's no separate "save all vars" state to
                gate. Kept as a parameter purely for call-site parity with
                ForecastModule.generate_rollouts.
            energy_score_noise_scale: No-op for this class -- this is an
                energy-score-deterministic-model-only knob (there is no
                energy-score sampling path here, generation is flow
                matching). Kept as a parameter purely for call-site parity
                with ForecastModule.generate_rollouts, whose caller
                (analysis/long_rollout.py) passes it unconditionally.
            teacher_force: If True, condition every step's autoregressive
                state (surface/level/lev) on the real ground-truth state
                from test_dataset instead of the model's own previous
                sample -- isolates whether artifacts seen in free-running
                flow-model rollouts come from accumulated autoregressive
                exposure bias vs. the model's single-pass sample itself.
                Mirrors ForecastModule.forward_multistep's teacher_force.
                The model's own sample is still saved/returned for
                evaluation as normal; only what feeds back into the next
                step's loop_batch changes. See sample_rollout.
            batch_seeds: No-op for this class -- this only exists as a
                ForecastModule.generate_rollouts performance toggle (batch
                every energy-score seed into one forward pass instead of
                looping sequentially). This class already batches every
                seed of a member together via sample_rollout's num_seeds
                (see "Tile initial batch so all seeds run in parallel"),
                unconditionally, so there is nothing extra to toggle here.
                Kept as a parameter purely for call-site parity with
                ForecastModule.generate_rollouts.
        """
        assert zero_forcing_source in ("null", "pi"), (
            f"Unknown zero_forcing_source {zero_forcing_source!r}, expected 'null' or 'pi'"
        )
        pi_spatial_values = None
        pi_non_spatial_values = None
        if zero_forcing_source == "pi" and (
            zero_spatial_forcing_indices or zero_non_spatial_forcing_indices
        ):
            from ArchesClimate.model.forecast import load_pi_forcing_values

            memmap_dir = str(Path(test_dataset.files[0]).parent)
            pi_spatial_values, pi_non_spatial_values = load_pi_forcing_values(
                memmap_dir, test_dataset
            )
            pi_spatial_values = pi_spatial_values.to(self.device)
            pi_non_spatial_values = pi_non_spatial_values.to(self.device)

        starting_indexes = [i for i, t in enumerate(test_dataset.id2pt.values()) if t[1] == 0]
        out_dir = out_dir.replace("local", "scratch")

        os.makedirs(out_dir, exist_ok=True)
        print("starting_indexes", starting_indexes)
        for ensemble_member in range(start_member, end_member):
            b_index = starting_indexes[ensemble_member]

            batch = test_dataset.__getitem__(b_index)
            batch = {k: batch[k].unsqueeze(0) for k in batch.keys()}
            batch = {k: batch[k].to(self.device) for k in batch.keys()}
            if "lead_time" in batch:
                # sample_rollout steps one reference timedelta (1 month) at a
                # time and never updates lead_time after this initial fetch --
                # but __getitem__ draws it at random from module.lead_times
                # (e.g. [1, 12] for the multi-lead-time training ablations),
                # so left alone this would silently condition the entire
                # rollout on a 12-month lead-time embedding for any member
                # whose initial draw wasn't 1. Pin it to the reference step.
                ref_step = min(test_dataset.lead_times) // test_dataset.timedelta
                batch["lead_time"] = torch.full_like(batch["lead_time"], ref_step)
                if "lead_time_months" in batch:
                    batch["lead_time_months"] = torch.full_like(batch["lead_time_months"], ref_step)
            base_seed = (batch_index * 10) + ensemble_member

            if seed_index is not None:
                # Run a single seed so jobs can be parallelised without
                # multiplying the batch size.  The seed arithmetic matches
                # the batched path (seed + s) so filenames are identical.
                self.sample_rollout(
                    deepcopy(batch),
                    lead_time_months=num_rollout_steps,
                    seed=base_seed + seed_index,
                    test_dataset=test_dataset,
                    index=b_index,
                    out_dir=out_dir,
                    ensemble_member_index=ensemble_member,
                    flat_forcings=flat_forcings,
                    debug=debug,
                    num_seeds=1,
                    zero_spatial_forcing_indices=zero_spatial_forcing_indices,
                    zero_non_spatial_forcing_indices=zero_non_spatial_forcing_indices,
                    pi_spatial_values=pi_spatial_values,
                    pi_non_spatial_values=pi_non_spatial_values,
                    clamp_spatial_forcing_indices=clamp_spatial_forcing_indices,
                    clamp_spatial_forcing_std=clamp_spatial_forcing_std,
                    clamp_polar_tas_std=clamp_polar_tas_std,
                    clamp_polar_n_rows=clamp_polar_n_rows,
                    teacher_force=teacher_force,
                )
            else:
                self.sample_rollout(
                    deepcopy(batch),
                    lead_time_months=num_rollout_steps,
                    seed=base_seed,
                    test_dataset=test_dataset,
                    index=b_index,
                    out_dir=out_dir,
                    ensemble_member_index=ensemble_member,
                    flat_forcings=flat_forcings,
                    debug=debug,
                    num_seeds=num_perturbations_per_member,
                    zero_spatial_forcing_indices=zero_spatial_forcing_indices,
                    zero_non_spatial_forcing_indices=zero_non_spatial_forcing_indices,
                    pi_spatial_values=pi_spatial_values,
                    pi_non_spatial_values=pi_non_spatial_values,
                    clamp_spatial_forcing_indices=clamp_spatial_forcing_indices,
                    clamp_spatial_forcing_std=clamp_spatial_forcing_std,
                    clamp_polar_tas_std=clamp_polar_tas_std,
                    clamp_polar_n_rows=clamp_polar_n_rows,
                    teacher_force=teacher_force,
                )
