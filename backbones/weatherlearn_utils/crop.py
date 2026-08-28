import torch


def crop3d(x: torch.Tensor, resolution):
    """Crop x to resolution.

    Args:
    x (torch.Tensor): B, C, Pl, Lat, Lon
    resolution (tuple[int]): Pl, Lat, Lon.
    """
    _, _, Pl, Lat, Lon = x.shape
    pl_pad = Pl - resolution[0]
    lat_pad = Lat - resolution[1]
    lon_pad = Lon - resolution[2]

    padding_front = pl_pad // 2
    padding_back = pl_pad - padding_front

    padding_top = lat_pad // 2
    padding_bottom = lat_pad - padding_top

    padding_left = lon_pad // 2
    padding_right = lon_pad - padding_left
    return x[
        :,
        :,
        padding_front : Pl - padding_back,
        padding_top : Lat - padding_bottom,
        padding_left : Lon - padding_right,
    ]
