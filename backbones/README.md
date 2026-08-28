# Backbones (`backbones/`)

- **`archesweather.py`** — `ArchesWeatherCondBackbone`, the actual spatial network used by every
  module config (`configs/module/*.yaml`'s `backbone:` block). A 4-stage Swin-Transformer-style
  U-Net over the patch-embedded 3-D (level, lat, lon) tensor: `layer1` (full resolution) →
  `downsample` (2x2 avg-pool in lat/lon) → `layer2` → `layer3` (bottleneck, same resolution as
  layer2) → `upsample` → `layer4` (back to full resolution, concatenated with `layer1`'s output
  as a skip connection if `use_skip=True`). Each `CondBasicLayer` is a stack of windowed
  self-attention blocks (`EarthAttention3D`/`EarthSpecificBlock` in `archesweather_layers.py`,
  adapted from [WeatherLearn](https://github.com/lizhuoq/WeatherLearn)'s Pangu implementation)
  that also takes per-token conditioning (`cond_emb`) — forcings and lead-time/timestep embeddings
  injected via adaLN, not concatenation, at every block. Window partitioning zero-pads to a
  window-size multiple internally, so `window_size` doesn't need to evenly divide the grid.
  Also defines `SpatialForcingProjector`, which patch-embeds 2-D forcing fields into that
  per-token conditioning sequence (the "SPADE"-style pathway — see `spatial_forcing_mode` in
  `model/base_climate_module.py`). CMIP training's own encode/decode layer is
  `model/cmip_encoder_decoder.py`'s `CMIPEncodeDecodeLayer` (handles the extra ocean (`lev`)
  field and CMIP's forcing channels; not in `backbones/` since it's tied to the CMIP TensorDict
  layout, not the backbone architecture). The original ERA5/reanalysis-shaped
  `WeatherEncodeDecodeLayer` this repo grew out of has been removed as dead code.
- **`archesweather_layers.py`** — the building blocks the above assembles: `Conv3dSimple` (a
  from-scratch 3-D conv reimplemented via reshape+matmul for MPS compatibility), `DownSample`/
  `UpSample`, `EarthAttention3D` (the windowed attention itself, with Earth-specific relative
  position bias), `EarthSpecificBlock` (one attention+MLP block), `BasicLayer`/`CondBasicLayer`
  (a depth-stack of blocks; `CondBasicLayer` adds the per-token conditioning and the optional
  CO2-strength/CO2-spatial-embedding adaLN paths), `LinVert` (an optional linear layer before
  `layer1`, `first_interaction_layer`).
- **`dit.py`** — small standalone embedding modules used elsewhere in the model, not part of the
  backbone forward pass itself: `LearnedEmbedder`, `TimestepEmbedder` (diffusion timestep → adaLN
  conditioning), `CO2LinearGain` (a learned scalar gain applied to the CO2 forcing channel; see
  the `co2spadegain`/`co2lineargain` module config variants).
- **`weatherlearn_utils/`** — vendored helpers from WeatherLearn (window partition/reverse, earth
  position index, 3-D padding) that `archesweather_layers.py` depends on directly.
