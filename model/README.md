# `model/forecast.py` vs. `model/diffusion.py`

Both are `ClimateLightningModule` subclasses (`model/base_climate_module.py`) that share the same
backbone/encoder-decoder plumbing, forcing dropout, EMA, and lat-weighted loss — they differ in
*what the network is trained to predict and how a rollout step is produced*:

- **`forecast.py`** (`ForecastModule`, `ForecastModuleWithCond`) — **deterministic regression**.
  The backbone directly predicts next month's state in one forward pass; training minimizes MSE
  (plus optional spectral/gradient loss terms and the multi-step "pushforward" loss,
  `pf_n_steps`/`pf_use_anomaly`/`pf_mean_trend_weight`) between that single prediction and the
  target. `ForecastModuleWithCond` adds the conditioning machinery used by every current CMIP
  module config: `ForcingDropout`, lead-time embedding, `conditional: prev+forcings`. This is
  what the plain `forcing_dropout_no_random_lt_*` module configs use (`_target_:
  ArchesClimate.model.forecast.ForecastModuleWithCond`).
- **`diffusion.py`** (`DCPPDiffusion`) — **generative, iterative sampling**. Wraps a
  [diffusers](https://github.com/huggingface/diffusers) noise scheduler (`scheduler: ddpm | heun
  | flow`, i.e. `FlowMatchHeunDiscreteScheduler`/`FlowMatchEulerDiscreteScheduler` for the latter
  two) around the same backbone; training samples a noise level/timestep and trains the backbone
  to predict the noise (or flow velocity, depending on scheduler), not the state directly.
  Producing one rollout step at inference time means running `inference.num_inference_steps`
  iterative denoising steps through the scheduler, not one forward pass — that's why
  `configs/inference/cmip_80_year.yaml` has `num_inference_steps`/`cf_guidance`/`s_churn`/
  `scale_input_noise` fields that `forecast.py`-based runs don't use. This is what every
  `flow_*` module config uses (`_target_: ArchesClimate.model.diffusion.DCPPDiffusion`,
  `scheduler: flow`) — e.g. `flow_damip`, `flow_damip_no_spec`.

In short: if a module config's `_target_` is `ForecastModuleWithCond`, one rollout step is one
backbone forward pass; if it's `DCPPDiffusion`, one rollout step is a full sampling loop.
