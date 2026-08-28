import torch


def window_partition(x: torch.Tensor, window_size):
    """Partition x into windows.

    Args:
        x: (B, Pl, Lat, Lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon].

    Returns:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
    """
    B, Pl, Lat, Lon, C = x.shape
    win_pl, win_lat, win_lon = window_size
    x = x.view(B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
    windows = (
        x.permute(0, 5, 1, 3, 2, 4, 6, 7)
        .contiguous()
        .view(-1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C)
    )
    return windows


def window_reverse(windows, window_size, Pl, Lat, Lon):
    """Reverse window_partition, reassembling windows back into x.

    Args:
        windows: (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
        window_size (tuple[int]): [win_pl, win_lat, win_lon].
        Pl (int): Number of pressure levels.
        Lat (int): Number of latitude points.
        Lon (int): Number of longitude points.

    Returns:
        x: (B, Pl, Lat, Lon, C)
    """
    win_pl, win_lat, win_lon = window_size
    B = int(windows.shape[0] / (Lon / win_lon))
    x = windows.view(B, Lon // win_lon, Pl // win_pl, Lat // win_lat, win_pl, win_lat, win_lon, -1)
    x = x.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
    return x
