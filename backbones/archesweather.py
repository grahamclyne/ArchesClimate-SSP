import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa N812
import torch.utils.checkpoint as gradient_checkpoint
from timm.layers.mlp import SwiGLU

from .archesweather_layers import (
    CondBasicLayer,
    DownSample,
    LinVert,
    Mlp,
    UpSample,
)


class SpatialForcingProjector(nn.Module):
    """Patch-embeds 2-D forcing fields to a per-token conditioning sequence.

    Input:  (B, forcing_ch, H, W)
    Output: (B, H_p * W_p, cond_dim)  where H_p = H // patch, W_p = W // patch
    """

    def __init__(self, forcing_ch: int, cond_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(forcing_ch, cond_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, forcings: torch.Tensor) -> torch.Tensor:
        x = self.proj(forcings)  # (B, cond_dim, H_p, W_p)
        return x.flatten(2).transpose(1, 2)  # (B, N_latlon, cond_dim)


class ArchesWeatherCondBackbone(nn.Module):
    def __init__(
        self,
        tensor_size=(8, 60, 120),
        emb_dim=192,
        cond_dim=256,  # dim of the conditioning
        num_heads=(6, 12, 12, 6),
        window_size=(1, 6, 10),
        droppath_coeff=0.2,
        depth_multiplier=2,
        dropout=0.0,
        mlp_ratio=4.0,
        use_skip=True,
        first_interaction_layer="linear",
        gradient_checkpointing=False,
        mlp_layer="mlp",
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(locals())
        drop_path = np.linspace(
            0, droppath_coeff / depth_multiplier, int(8 * depth_multiplier)
        ).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        self.zdim = tensor_size[0]

        self.layer1_shape = tensor_size[1:]

        self.layer2_shape = (self.layer1_shape[0] // 2, self.layer1_shape[1] // 2)

        if first_interaction_layer == "linear":
            self.interaction_layer = LinVert(in_features=emb_dim)

        layer_args = dict(
            cond_dim=cond_dim,
            window_size=window_size,
            act_layer=nn.GELU,
            drop=dropout,
            mlp_layer=Mlp,
            mlp_ratio=mlp_ratio,
        )

        if mlp_layer == "swiglu":
            layer_args["mlp_ratio"] = mlp_ratio * 2 / 3
            layer_args["mlp_layer"] = SwiGLU

        self.layer1 = CondBasicLayer(
            dim=emb_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=int(2 * depth_multiplier),
            num_heads=num_heads[0],
            drop_path=drop_path[: int(2 * depth_multiplier)],
            **layer_args,
            **kwargs,
        )
        self.downsample = DownSample(
            in_dim=emb_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            output_resolution=(self.zdim, *self.layer2_shape),
            pole_padding_mode=kwargs.get("pole_padding_mode", "zero"),
        )
        self.layer2 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=int(6 * depth_multiplier),
            num_heads=num_heads[1],
            drop_path=drop_path[int(2 * depth_multiplier) :],
            **layer_args,
            **kwargs,
        )
        self.layer3 = CondBasicLayer(
            dim=emb_dim * 2,
            input_resolution=(self.zdim, *self.layer2_shape),
            depth=int(6 * depth_multiplier),
            num_heads=num_heads[2],
            drop_path=drop_path[int(2 * depth_multiplier) :],
            **layer_args,
            **kwargs,
        )
        self.upsample = UpSample(
            emb_dim * 2, emb_dim, (self.zdim, *self.layer2_shape), (self.zdim, *self.layer1_shape)
        )
        out_dim = emb_dim if not self.use_skip else 2 * emb_dim
        self.layer4 = CondBasicLayer(
            dim=out_dim,
            input_resolution=(self.zdim, *self.layer1_shape),
            depth=int(2 * depth_multiplier),
            num_heads=num_heads[3],
            drop_path=drop_path[: int(2 * depth_multiplier)],
            **layer_args,
            **kwargs,
        )

    def _downsample_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Average-pool token-wise conditioning from layer1 to layer2 resolution."""
        B, N, D = cond.shape
        lat, lon = self.layer1_shape
        c = cond.reshape(B, self.zdim, lat, lon, D).permute(0, 4, 1, 2, 3)
        c = F.avg_pool3d(c.float(), kernel_size=(1, 2, 2), stride=(1, 2, 2)).to(cond.dtype)
        return c.permute(0, 2, 3, 4, 1).flatten(1, 3)  # (B, N_down, D)

    def forward(self, x, cond_emb, co2_strength=None, co2_spatial_emb=None, **kwargs):
        # cond_emb: (B, N, cond_dim) — token-wise conditioning
        # co2_strength: (B,) unbounded CO2 forcing-strength scalar, or None
        # co2_spatial_emb: (B, N, cond_dim) token-wise CO2 projection ("spade"
        # mode), or None -- mutually exclusive with co2_strength, see
        # CondBasicLayer.forward
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        if self.first_interaction_layer:
            x = self.interaction_layer(x)

        x = self.layer1(x, cond_emb, co2_strength=co2_strength, co2_spatial_emb=co2_spatial_emb)

        skip = x
        x = self.downsample(x)
        cond_emb_down = self._downsample_cond(cond_emb)
        co2_spatial_emb_down = (
            self._downsample_cond(co2_spatial_emb) if co2_spatial_emb is not None else None
        )

        x = self.layer2(
            x, cond_emb_down, co2_strength=co2_strength, co2_spatial_emb=co2_spatial_emb_down
        )

        if self.gradient_checkpointing:
            x = gradient_checkpoint.checkpoint(
                self.layer3,
                x,
                cond_emb_down,
                co2_strength,
                co2_spatial_emb_down,
                use_reentrant=False,
            )
        else:
            x = self.layer3(
                x, cond_emb_down, co2_strength=co2_strength, co2_spatial_emb=co2_spatial_emb_down
            )

        x = self.upsample(x)
        if self.use_skip and skip is not None:
            x = torch.concat([x, skip], dim=-1)
        x = self.layer4(x, cond_emb, co2_strength=co2_strength, co2_spatial_emb=co2_spatial_emb)

        output = x
        output = output.transpose(1, 2).reshape(output.shape[0], -1, 8, *self.layer1_shape)

        return output
