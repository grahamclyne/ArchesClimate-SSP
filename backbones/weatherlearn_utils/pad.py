import torch
import torch.nn.functional as F


def spherical_pad3d(x, padding, mode="zero"):
    """Pad a (B, C, Pl, Lat, Lon) tensor for window alignment.

    padding: (left, right, top, bottom, front, back), same convention as
        nn.ZeroPad3d -- (lon_left, lon_right, lat_top, lat_bottom, pl_front, pl_back).
    mode="zero": identical to nn.ZeroPad3d(padding)(x) -- previous behaviour.
    mode="spherical": longitude is padded circularly (it genuinely wraps at
        the date line, unlike the previous zero-fill which fabricated an
        absent neighbour there); latitude is padded via reflect + a
        half-circumference longitude roll (the physically correct neighbour
        when a padded window would extend past a pole); pressure level is
        still zero-padded (no periodicity assumption there).
    """
    pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back = padding
    if mode == "zero":
        return F.pad(x, padding, mode="constant", value=0)
    if mode != "spherical":
        raise ValueError(f"Unknown pole_padding_mode: {mode!r}")

    out = x
    if pad_top or pad_bottom:
        reflected = F.pad(out, (0, 0, pad_top, pad_bottom, 0, 0), mode="reflect")
        lat_total = reflected.shape[-2]
        lon = reflected.shape[-1]
        shift = lon // 2
        top = torch.roll(reflected[..., :pad_top, :], shifts=shift, dims=-1)
        mid = reflected[..., pad_top : lat_total - pad_bottom, :]
        bottom = torch.roll(reflected[..., lat_total - pad_bottom :, :], shifts=shift, dims=-1)
        out = torch.cat([top, mid, bottom], dim=-2)
    if pad_left or pad_right:
        out = F.pad(out, (pad_left, pad_right, 0, 0, 0, 0), mode="circular")
    if pad_front or pad_back:
        out = F.pad(out, (0, 0, 0, 0, pad_front, pad_back), mode="constant", value=0)
    return out


def get_pad3d(input_resolution, window_size):
    """Compute the padding needed to make input_resolution divide evenly into window_size.

    Args:
        input_resolution (tuple[int]): (Pl, Lat, Lon)
        window_size (tuple[int]): (Pl, Lat, Lon).

    Returns:
        padding (tuple[int]): (padding_left, padding_right, padding_top,
            padding_bottom, padding_front, padding_back)
    """
    Pl, Lat, Lon = input_resolution
    win_pl, win_lat, win_lon = window_size

    padding_left = padding_right = padding_top = padding_bottom = padding_front = padding_back = 0
    pl_remainder = Pl % win_pl
    lat_remainder = Lat % win_lat
    lon_remainder = Lon % win_lon

    if pl_remainder:
        pl_pad = win_pl - pl_remainder
        padding_front = pl_pad // 2
        padding_back = pl_pad - padding_front
    if lat_remainder:
        lat_pad = win_lat - lat_remainder
        padding_top = lat_pad // 2
        padding_bottom = lat_pad - padding_top
    if lon_remainder:
        lon_pad = win_lon - lon_remainder
        padding_left = lon_pad // 2
        padding_right = lon_pad - padding_left

    return padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back
