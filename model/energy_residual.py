import importlib.resources
from typing import Any

import lovely_tensors as lt
import torch
from tensordict.tensordict import TensorDict

lt.monkey_patch()
import ArchesClimate.stats as ArchesClimate_stats  # noqa: E402
from ArchesClimate.model.forecast import ForecastModuleWithCond  # noqa: E402

ArchesClimate_stats_path = importlib.resources.files(ArchesClimate_stats)


class EnergyScoreResidual(ForecastModuleWithCond):
    """Energy-score-trained residual model around a frozen deterministic model.

    Same idea as diffusion.py's DCPPDiffusion with learn_residual="pred" +
    state_normalization=True (predict a normalized residual around a frozen
    det_model's one-step prediction, conditioned on that same prediction),
    but trained under the (fair/unbiased) energy score
    (ForecastModule._energy_score_loss) over a direct M-member stochastic
    ensemble instead of flow matching's iterative denoiser -- reusing
    ForecastModule's existing use_energy_score/n_energy_score_members/
    energy_score_noise_mode machinery.

    Implementation: forward() is the ONLY thing overridden. It runs the
    frozen det_model (no grad) to get this step's deterministic prediction,
    feeds it in as conditioning (batch["pred_state"], requires
    conditional to include "det", e.g. "prev+det+forcings" -- see
    shared_forward_logic's "det" in conditional_keys branch), and returns
    det_pred + state_scaler * (raw backbone output). Everywhere else in
    ForecastModule/ForecastModuleWithCond -- training_step (including its
    pushforward loop, lambda_pf > 0), loss, validation_step,
    forward_multistep, generate_rollouts -- calls self.forward(...) and
    treats its return value as a plain full-state prediction, so all of
    that machinery works completely unmodified.

    This composition is loss-equivalent to training directly on the
    residual: _state_rmse (and the patch/variogram terms) are translation
    invariant, so rmse(det+res_i, gt) == rmse(res_i, gt-det) and
    rmse(det+res_i, det+res_j) == rmse(res_i, res_j) exactly, for the same
    (frozen, gradient-free) det_pred added to every member and to gt alike.
    The pushforward loop rolling this composed prediction forward and
    re-running det_model at each step (again no grad -- it's frozen) is
    therefore also training the residual head to correct det_model's own
    compounding drift, not just its single-step error.

    Args (beyond ForecastModuleWithCond's, all of which pass straight
    through via **kwargs):
        load_deterministic_model: modelstore-relative or absolute path to
            the frozen deterministic model to wrap (see model/base_module.py
            load_module), e.g.
            "/scratch/gclyne/model_output/cmip-interpolation/deterministic_damip_pf2_fixed".
        det_ckpt_fname: Checkpoint filename under that path's checkpoints/,
            e.g. "step-step=022000.ckpt". None picks the most recent.
        delta_std_name: Name of the delta-std stats file under
            ArchesClimate/stats/ (see utils/generate_delta_stats.py), used to
            build the per-channel state_scaler the backbone's raw output is
            multiplied by before being added to det_pred. Reusing a
            different (but architecturally/variable-set-identical) model's
            stats is fine -- this is a per-channel scale, not a learned
            parameter.
        state_scaler_multiplier: Scalar multiplier on the loaded state_scaler
            (see diffusion.py's DCPPDiffusion, same option, same rationale).
    """

    def __init__(
        self,
        *args: Any,
        load_deterministic_model: str,
        det_ckpt_fname: str | None = None,
        delta_std_name: str,
        state_scaler_multiplier: float = 1.0,
        **kwargs: Any,
    ) -> None:
        kwargs["use_energy_score"] = True
        super().__init__(*args, **kwargs)
        assert self.use_energy_score, "EnergyScoreResidual always trains with use_energy_score=True"
        assert "det" in self.conditional.split("+"), (
            "EnergyScoreResidual requires 'det' in conditional (e.g. "
            f"'prev+det+forcings'), got {self.conditional!r} -- it "
            "conditions its residual prediction on the frozen det_model's "
            "own output, see the class docstring."
        )

        from ArchesClimate.model.base_module import load_module

        self.det_model, _ = load_module(load_deterministic_model, ckpt_fname=det_ckpt_fname)
        for param in self.det_model.parameters():
            param.requires_grad = False

        self.delta_std_name = delta_std_name
        raw = torch.load(ArchesClimate_stats_path.joinpath(delta_std_name))
        if "per_lead_time" in raw:
            self.state_scaler_by_lead_time = {
                int(lt_): self._build_state_scaler(stats, state_scaler_multiplier)
                for lt_, stats in raw["per_lead_time"].items()
            }
            self.state_scaler = None
        else:
            self.state_scaler = self._build_state_scaler(raw, state_scaler_multiplier)
            self.state_scaler_by_lead_time = None

    @staticmethod
    def _build_state_scaler(raw: dict, state_scaler_multiplier: float) -> TensorDict:
        """Same construction as diffusion.py's DCPPDiffusion._build_state_scaler."""
        scaler = TensorDict(
            {
                "surface": raw["surface"].nanmean(axis=(-1, -2), keepdims=True).unsqueeze(1),
                "level": raw["level"].nanmean(axis=(-1, -2), keepdims=True).unsqueeze(0),
                "lev": raw["lev"].nanmean(axis=(-1, -2), keepdims=True).unsqueeze(0),
            }
        )
        if state_scaler_multiplier != 1.0:
            scaler = scaler.apply(lambda x: x * state_scaler_multiplier)
        return scaler

    def _get_state_scaler(self, lead_time: torch.Tensor) -> TensorDict:
        if self.state_scaler_by_lead_time is None:
            return self.state_scaler.to(self.device)
        fallback = next(iter(self.state_scaler_by_lead_time.values()))
        per_sample = [
            self.state_scaler_by_lead_time.get(int(lt_), fallback)
            for lt_ in lead_time.reshape(-1).tolist()
        ]
        return torch.stack(per_sample, dim=0).to(self.device)

    def forward(
        self, batch: dict[str, Any], use_condition: bool | torch.Tensor, **kwargs: Any
    ) -> TensorDict:
        """det_pred + state_scaler * residual_backbone_output. See class docstring.

        Draws one fresh stochastic residual member per call (same mechanism
        as any other use_energy_score model, via shared_forward_logic) --
        det_model's own forward is deterministic-ish (its own forcing_dropout
        still samples per call, shared with this model's own draw below so
        both agree on which forcings were visible, matching
        diffusion.py's learn_residual="pred" rationale) but carries no
        gradient and is not itself a second source of ensemble spread.
        """
        forcing_mask = kwargs.get("forcing_mask", None)
        if forcing_mask is None and hasattr(self, "forcing_dropout"):
            B = batch["state"]["non_spatial_forcings"].shape[0]
            forcing_mask = self.forcing_dropout.sample_mask(B, batch["state"].device)
            kwargs["forcing_mask"] = forcing_mask

        with torch.no_grad():
            det_batch = {
                "state": batch["state"],
                "prev_state": batch["prev_state"],
                "lead_time": batch["lead_time"],
                "timestamp": batch["timestamp"],
            }
            det_pred = self.det_model.forward(
                det_batch, use_condition=True, forcing_mask=forcing_mask
            ).detach()
        det_pred["surface"] = det_pred["surface"] * self.surface_mask.to(self.device)
        det_pred["level"] = det_pred["level"] * self.level_mask.to(self.device)
        det_pred["lev"] = det_pred["lev"] * self.lev_mask.to(self.device)

        batch = dict(batch)
        batch["pred_state"] = det_pred
        residual = super().forward(batch, use_condition, **kwargs)

        scaler = self._get_state_scaler(batch["lead_time"])
        return det_pred + residual.apply(lambda x, y: x * y, scaler)
