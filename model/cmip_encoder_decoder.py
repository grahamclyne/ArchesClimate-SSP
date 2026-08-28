from typing import Any

import lovely_tensors as lt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict

from ArchesClimate.backbones.archesweather_layers import ICNR_init

lt.monkey_patch()


class CMIPEncodeDecodeLayer(nn.Module):
    """gathers layers for the encoder and decoder."""

    def __init__(
        self,
        img_size,
        emb_dim,
        out_emb_dim,
        patch_size,
        surface_ch,
        n_concatenated_states,
        boundary_padding=False,
        double_prev_state=False,
        total_boundary_calc=False,
        level_ch=7,
        conditional_forcings=True,
        forcing_ch=11,
        ocean_depth_channels=10,
        surface_variables=8,
        level_variables=5,
        load_prev=1,
        is_flow=False,
        spatial_forcing_ch=0,
        pole_padding_mode="zero",
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        self.lat_dim, self.lon_dim = img_size[1], img_size[2]

        self.surface_proj = nn.Conv2d(
            surface_ch + spatial_forcing_ch,
            emb_dim,
            kernel_size=patch_size[1:],
            stride=patch_size[1:],
        )
        self.boundary_padding = boundary_padding
        # padding=0: boundary handling is done explicitly in decode() below,
        # since longitude (width) is periodic (needs circular padding, not
        # zero) while latitude (height) terminates at the poles (zero padding
        # is fine there).
        self.surface_deconv = nn.Conv2d(
            out_emb_dim,
            surface_ch * patch_size[-1] ** 2,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=0,
        )
        self.pixelshuffle = nn.PixelShuffle(patch_size[-1])
        ICNR_init(
            self.surface_deconv.weight,
            initializer=nn.init.kaiming_normal_,
            upscale_factor=patch_size[-1],
        )

    def encode(
        self,
        state: TensorDict,
        cond_state: TensorDict = None,
        spatial_forcings: torch.Tensor = None,
    ) -> Any:
        """Encode state and optional conditioning state to latent tokens.

        Args:
            state: Current state TensorDict.
            cond_state: Optional conditioning state concatenated to state.
            spatial_forcings: Optional (B, F, H, W) spatial forcing tensor,
                already dropout-masked, appended as extra input channels.

        Returns:
            Encoded token tensor with shape [..., 1, H, W].
        """
        bs = state.shape[0]
        # flatten atmos and ocean to match surface
        surface = state["surface"].squeeze(-3)
        level = state["level"].reshape(bs, -1, self.lat_dim, self.lon_dim)
        ocean = state["lev"].reshape(bs, -1, self.lat_dim, self.lon_dim)
        # add conditional state if given
        if cond_state is not None:
            cond_surface = cond_state["surface"].squeeze(-3)
            cond_level = cond_state["level"].reshape(bs, -1, self.lat_dim, self.lon_dim)
            cond_ocean = cond_state["lev"].reshape(bs, -1, self.lat_dim, self.lon_dim)
            surface = torch.cat([surface, cond_surface], dim=1)
            level = torch.cat([level, cond_level], dim=1)
            ocean = torch.cat([ocean, cond_ocean], dim=1)

        surface = torch.cat([surface, level, ocean], dim=1)

        if spatial_forcings is not None:
            surface = torch.cat([surface, spatial_forcings], dim=1)

        surface = self.surface_proj(surface)

        return surface.unsqueeze(-3)  # unsqueeze to add channel dim for backbone

    def decode(self, x: Any) -> Any:
        """Decode latent tokens to surface, level, and ocean state TensorDict.

        Args:
            x: Latent token tensor from backbone.

        Returns:
            TensorDict with surface, level, and lev keys.
        """
        plev_vars = self.level_variables
        plev_indices = self.level_ch * plev_vars
        if self.is_flow:
            output_dim = (self.surface_ch // 4) - plev_indices
        else:  # deterministic
            output_dim = self.surface_ch // (1 + self.load_prev) - plev_indices

        output = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        # Longitude (width) wraps around (date line), so pad it circularly.
        output = F.pad(output, (1, 1, 0, 0), mode="circular")
        # Latitude (height) terminates at the poles. Two modes:
        #   "zero" (default): zero-pad, as before -- the model sees a
        #     fabricated absent neighbour beyond each pole.
        #   "reflect_antipodal": going one step past a pole re-enters the
        #     grid at the same latitude on the opposite side of the globe
        #     (longitude + 180 deg), which is the physically correct
        #     neighbour on a sphere -- standard for equirectangular global
        #     forecasting nets (FourCastNet/GraphCast-style). Implemented as
        #     reflect-padding (mirror the row one step in from each edge,
        #     matching how a flat boundary's "beyond the edge" neighbour is
        #     normally constructed) combined with a half-circumference
        #     torch.roll of that mirrored row's longitude axis.
        if self.pole_padding_mode == "reflect_antipodal":
            lon_dim = output.shape[-1]
            shift = lon_dim // 2
            south_pad = torch.roll(output[..., 1:2, :], shifts=shift, dims=-1)
            north_pad = torch.roll(output[..., -2:-1, :], shifts=shift, dims=-1)
            output = torch.cat([south_pad, output, north_pad], dim=-2)
        elif self.pole_padding_mode == "zero":
            output = F.pad(output, (0, 0, 1, 1), mode="constant", value=0)
        else:
            raise ValueError(f"Unknown pole_padding_mode: {self.pole_padding_mode!r}")
        output = self.surface_deconv(output)
        output = self.pixelshuffle(output)
        output = output.unsqueeze(-3)
        # Explicit batch size (not -1) below: when plev_vars * level_ch == 0 (a
        # module config with no level_variables at all -- see
        # configs/module/deterministic_tas_only_patchenergy.yaml), the sliced
        # tensor has 0 elements and reshape can't infer -1 from "0 elements /
        # 0-sized dims" (any batch size satisfies that, so it's ambiguous and
        # errors instead of guessing). output.shape[0] is always well-defined
        # regardless of channel counts, so it works for both the normal and
        # zero-level-variables case.
        output_level = output[:, output_dim : output_dim + plev_indices].reshape(
            output.shape[0], plev_vars, self.level_ch, self.lat_dim, self.lon_dim
        )
        output_lev = output[
            :,
            output_dim + plev_indices : output_dim + plev_indices + self.ocean_depth_channels,
        ].reshape(output.shape[0], 1, self.ocean_depth_channels, self.lat_dim, self.lon_dim)
        output_surface = output[
            :,
            output_dim + plev_indices + self.ocean_depth_channels : output_dim
            + plev_indices
            + self.ocean_depth_channels
            + self.surface_variables,
        ]
        return TensorDict(
            surface=output_surface,
            level=output_level,
            lev=output_lev,
            batch_size=output_surface.shape[0],
        ).to(x.device)
