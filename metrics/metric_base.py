# Base class for metrics.
from collections.abc import Callable

import torch


def compute_lat_weights_weatherbench(latitude_resolution: int) -> torch.tensor:
    """Calculate the area overlap as a function of latitude.

    The weatherbench version gives slightly different coeffs.
    """
    latitudes = torch.linspace(-90, 90, latitude_resolution)
    points = torch.deg2rad(latitudes)
    pi_over_2 = torch.tensor([torch.pi / 2], dtype=torch.float32)
    bounds = torch.concatenate([-pi_over_2, (points[:-1] + points[1:]) / 2, pi_over_2])
    upper = bounds[1:]
    lower = bounds[:-1]
    # normalized cell area: integral from lower to upper of cos(latitude)
    weights = torch.sin(upper) - torch.sin(lower)
    weights = weights / weights.mean()
    return weights[:, None]


class MetricBase:
    """Implement latitude-weighted base functions."""

    def __init__(
        self,
        compute_lat_weights_fn: Callable[[int], torch.tensor] = compute_lat_weights_weatherbench,
    ):
        """Initialize the metric base.

        Args:
        variable_indices: dict used to extract indices from output tensor.
        compute_lat_weights_fn: Function to compute latitude weights given latitude shape.
            Used for error and variance calculations. Expected shape of weights: [..., lat, 1].
        """
        super().__init__()
        self.compute_lat_weights_fn = compute_lat_weights_fn

    def wmse(self, x: torch.Tensor, y: torch.Tensor | int = 0):
        """Latitude weighted mse error.

        Args:
            x: preds with shape (..., lat, lon)
            y: targets with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return (x - y).pow(2).mul(lat_coeffs).nanmean((-2, -1))

    def wmae(self, x: torch.Tensor, y: torch.Tensor | int = 0):
        """Latitude weighted mae error.

        Args:
            x: preds with shape (..., lat, lon)
            y: targets with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return (x - y).abs().mul(lat_coeffs).nanmean((-2, -1))

    def wvar(self, x: torch.Tensor, dim: int = 1):
        """Latitude weighted variance along axis.

        Args:
            x: preds with shape (..., lat, lon)
            dim: over which dimension to compute variance.
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return x.var(dim).mul(lat_coeffs).nanmean((-2, -1))

    def weighted_mean(self, x: torch.Tensor):
        """Latitude weighted mean over grid.

        Args:
            x: preds with shape (..., lat, lon)
        """
        lat_coeffs = self.compute_lat_weights_fn(latitude_resolution=x.shape[-2]).to(x.device)
        return x.mul(lat_coeffs).nanmean((-2, -1))
