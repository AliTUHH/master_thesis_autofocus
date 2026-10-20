"""Hologram preprocessing of [1] Sec. 4.1 / Alg. 1: flat field, mirroring, padding, fading.

A measured hologram covers only the detector. To sample the Fresnel kernel (``N >= 1/Fr``)
and to avoid truncation edges, [1] mirrors the hologram into all directions, pads the result
with the constant illumination value ``A0`` and applies a modified Blackman window that fades
only the extended area, leaving the original field of view unchanged (Fig. 3-4 of [1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .arrays import center_slices, pad_center, reflect_pad


def flat_field_correct(raw: torch.Tensor, flat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """``I_corr = I_raw / I_flat``, [1] Eq. (10)."""
    return raw / flat.clamp(min=eps)


def blackman(width: int, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Blackman window ``0.42 - 0.5 cos(2 pi x/(w-1)) + 0.08 cos(4 pi x/(w-1))``, [1] Eq. (17)."""
    if width < 2:
        return torch.ones(width, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=torch.float64)
    y = (
        0.42
        - 0.5 * torch.cos(2 * math.pi * x / (width - 1))
        + 0.08 * torch.cos(4 * math.pi * x / (width - 1))
    )
    return y.clamp(min=0.0).to(dtype)


def fade_window_1d(
    n: int, start: int, stop: int, fade_width: int, device=None, dtype=torch.float32
) -> torch.Tensor:
    """1-D window of [1] Fig. 4(b).

    A Blackman window of width ``fade_width`` is split at its centre; the rising half is
    placed at ``start``, the falling half ends at ``stop`` and the gap is filled with ones.
    Outside ``[start, stop)`` the window is zero.
    """
    half = fade_width // 2
    if stop - start < 2 * half:
        raise ValueError("fade width larger than the region to be faded")
    window = torch.zeros(n, device=device, dtype=dtype)
    window[start:stop] = 1.0
    if half > 0:
        ramp = blackman(2 * half, device=device, dtype=dtype)
        window[start : start + half] = ramp[:half]
        window[stop - half : stop] = ramp[half:]
    return window


def fade_window_2d(
    shape: tuple[int, int],
    rows: tuple[int, int],
    cols: tuple[int, int],
    fade_width: int,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """Separable 2-D window of [1] Fig. 4(c): outer product of two 1-D windows."""
    w_rows = fade_window_1d(shape[0], rows[0], rows[1], fade_width, device, dtype)
    w_cols = fade_window_1d(shape[1], cols[0], cols[1], fade_width, device, dtype)
    return w_rows[:, None] * w_cols[None, :]


@dataclass
class PreprocessedHologram:
    """Padded hologram plus the bookkeeping needed to go back to the detector field of view."""

    data: torch.Tensor
    window: torch.Tensor
    fov: tuple[slice, slice]
    a0: float

    def crop(self, array: torch.Tensor) -> torch.Tensor:
        """Cut the original field of view out of a padded array (``cutfov`` in [1] Alg. 2)."""
        return array[..., self.fov[0], self.fov[1]]


def preprocess_hologram(
    hologram: torch.Tensor,
    padded_shape: tuple[int, int],
    a0: float = 1.0,
    mirror: bool = True,
    fade_width: int | None = None,
) -> PreprocessedHologram:
    """[1] Alg. 1 after flat-field correction: ``fade(pad_A0(mirr(I)))``.

    Parameters
    ----------
    hologram:
        Flat-field corrected hologram ``(..., h, w)``.
    padded_shape:
        Target array size, typically ``>= 1/Fr`` per dimension.
    a0:
        Constant used for padding and as the fade target. [1] calls it ``A0`` (constant
        probe model) and tunes it per hologram (Table 4: 0.96-1.1); it is 1 for an ideal
        flat-field correction and for noise-free simulations.
    mirror:
        Extend the hologram by mirroring before padding (``MIRROR_ALL`` in the paper code).
        The extension per side is limited to ``size - 1`` and by ``padded_shape``.
    fade_width:
        Width of the Blackman window whose halves fade the outer border of the extended
        block. Defaults to the mirrored margin, so each half fades the outer half of the
        mirrored copies; without mirroring it defaults to 0, so the measured field of view
        is never modified.
    """
    h, w = hologram.shape[-2:]
    big_h, big_w = padded_shape
    if big_h < h or big_w < w:
        raise ValueError(f"padded shape {padded_shape} smaller than hologram {(h, w)}")
    if mirror:
        pad_r = min(h - 1, (big_h - h) // 2)
        pad_c = min(w - 1, (big_w - w) // 2)
        extended = reflect_pad(hologram, (pad_c, pad_c, pad_r, pad_r))
    else:
        pad_r = pad_c = 0
        extended = hologram
    if fade_width is None:
        fade_width = min(pad_r, pad_c) if mirror else 0
    data = pad_center(extended, (big_h, big_w), a0)
    ext_rows, ext_cols = center_slices((big_h, big_w), tuple(extended.shape[-2:]))
    window = fade_window_2d(
        (big_h, big_w),
        (ext_rows.start, ext_rows.stop),
        (ext_cols.start, ext_cols.stop),
        fade_width,
        device=hologram.device,
        dtype=hologram.dtype if not hologram.is_complex() else torch.float32,
    )
    data = torch.where(window < 1, (data - a0) * window + a0, data)
    fov_rows, fov_cols = center_slices((big_h, big_w), (h, w))
    return PreprocessedHologram(data=data, window=window, fov=(fov_rows, fov_cols), a0=a0)
