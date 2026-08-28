import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings.

        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class CO2LinearGain(nn.Module):
    """Affine (unbounded, no activation) map from a CO2 forcing scalar to a "forcing strength".

    Per batch element.

    Deliberately has no nonlinearity: the whole point is that scaling CO2's
    forcing beyond the training range scales this output by the same factor,
    so whatever consumes it (see CondBasicLayer's co2_pattern) can't saturate
    the way the shared adaLN_modulation (SiLU + Linear, fed by a sum of every
    forcing's embedding) empirically does OOD. Fed the spatial mean of the
    normalized "carbon" channel (see base_climate_module.py's co2_strength).
    """

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, co2_mean: torch.Tensor) -> torch.Tensor:
        return self.gain * co2_mean + self.bias
