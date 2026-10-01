"""Fresnel propagation with the kernel convention of HoloForge and ``holowizard.core`` (PyTorch, CPU/GPU).

Both HoloWizard components use the transfer-function propagator

    D_Fr(psi) = IFFT2( FFT2(psi) * exp(-i * pi / Fr * (xi^2 + eta^2)) ),

with ``xi, eta = torch.fft.fftfreq(N)`` in **cycles per pixel** (no ``d`` argument) and ``Fr`` the
dimensionless *pixel* Fresnel number of the grid that is transformed (:func:`src.utils.physics.fresnel_number`).
Back-propagation uses ``-Fr`` (the complex conjugate kernel).  HoloForge simulates the detector-plane
**amplitude** ``|D_Fr(exp(i O) P)|`` of the complex object ``O = phi + i mu`` (phase shift ``phi <= 0``,
absorption ``mu >= 0``) on a grid enlarged by ``padding_factor`` (``O`` zero-padded, i.e. the wave is ``1``
outside the object) and crops the central ``N x N`` pixels; :func:`hologram_amplitude` reproduces that.

The single-sided Fresnel-number error of an autofocus method acts like a residual defocus: back-propagating
with ``Fr' = Fr (1 + e)`` and re-propagating with ``Fr`` leaves a Fresnel propagator with the effective pixel
Fresnel number ``Fr_eff = Fr Fr' / |Fr' - Fr| ~ Fr / |e|``.  The radius of its first Fresnel zone in detector
pixels, ``b = sqrt(|e| / Fr)`` (:func:`defocus_blur_px`), is a geometry- and binning-independent measure of
the resulting blur (target: ``b <= 1-2`` px at the resolution used for the reconstruction).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "fresnel_kernel",
    "pad_center",
    "crop_center",
    "propagate",
    "backpropagate",
    "hologram_amplitude",
    "defocus_blur_px",
    "blur_px_from_fresnel_numbers",
]


def fresnel_kernel(shape: tuple[int, int], fr: float, device: torch.device | str | None = None) -> torch.Tensor:
    """Transfer function ``exp(-i pi / Fr (xi^2 + eta^2))`` on an ``(H, W)`` grid (complex64, unshifted FFT order).

    Args:
        shape: Grid size ``(H, W)`` of the array that will be transformed (the *padded* grid).
        fr: Pixel Fresnel number of that grid; negative values give the back-propagation kernel.
        device: Torch device of the returned tensor (default: CPU).
    """
    if fr == 0:
        raise ValueError("fr must be non-zero")
    height, width = int(shape[0]), int(shape[1])
    xi = torch.fft.fftfreq(height, device=device, dtype=torch.float64)
    eta = torch.fft.fftfreq(width, device=device, dtype=torch.float64)
    u = xi[:, None] ** 2 + eta[None, :] ** 2
    return torch.exp((-1j * np.pi / float(fr)) * u).to(torch.complex64)


def _as_tensor(x: ArrayLike | torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    tensor = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x))
    return tensor.to(dtype)


def pad_center(x: torch.Tensor, pad_factor: float, value: float = 0.0) -> torch.Tensor:
    """Symmetrically pad the last two axes of ``x`` to ``round(pad_factor * size)`` with a constant ``value``."""
    if pad_factor < 1:
        raise ValueError(f"pad_factor must be >= 1, got {pad_factor}")
    height, width = x.shape[-2], x.shape[-1]
    target_h, target_w = round(height * pad_factor), round(width * pad_factor)
    top, left = (target_h - height) // 2, (target_w - width) // 2
    pads = [left, target_w - width - left, top, target_h - height - top]
    if not any(pads):
        return x
    if x.is_complex():
        # F.pad does not support complex constants; pad the real and imaginary parts separately.
        return torch.complex(F.pad(x.real, pads, value=float(value)), F.pad(x.imag, pads, value=0.0))
    return F.pad(x, pads, value=float(value))


def crop_center(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Central crop of the last two axes to ``shape = (H, W)``."""
    height, width = x.shape[-2], x.shape[-1]
    top, left = (height - int(shape[0])) // 2, (width - int(shape[1])) // 2
    return x[..., top : top + int(shape[0]), left : left + int(shape[1])]


def propagate(field: ArrayLike | torch.Tensor, fr: float, pad_factor: float = 2.0, pad_value: float = 0.0) -> torch.Tensor:
    """Fresnel-propagate a complex field ``[..., H, W]`` with the pixel Fresnel number ``fr``.

    The field is padded to ``pad_factor`` times its size with the constant ``pad_value`` (``0`` = dark
    surroundings; use ``1.0`` for a plane wave outside the field of view), transformed with
    :func:`fresnel_kernel` and cropped back to the input size.  Returns a complex64 tensor on the device
    of the input (NumPy inputs are converted and the result stays on the CPU).
    """
    psi = _as_tensor(field, torch.complex64)
    shape = (psi.shape[-2], psi.shape[-1])
    padded = pad_center(psi, pad_factor, value=pad_value)
    kernel = fresnel_kernel((padded.shape[-2], padded.shape[-1]), fr, device=padded.device)
    propagated = torch.fft.ifft2(torch.fft.fft2(padded) * kernel)
    return crop_center(propagated, shape)


def backpropagate(field: ArrayLike | torch.Tensor, fr: float, pad_factor: float = 2.0, pad_value: float = 0.0) -> torch.Tensor:
    """Inverse propagation ``D_{-Fr}`` (conjugate kernel); see :func:`propagate`."""
    return propagate(field, -float(fr), pad_factor=pad_factor, pad_value=pad_value)


def hologram_amplitude(
    obj: ArrayLike | torch.Tensor,
    fr: float,
    pad_factor: float = 2.0,
    probe: ArrayLike | torch.Tensor | None = None,
) -> torch.Tensor:
    """Detector-plane amplitude ``|D_Fr(exp(i O) P)|`` of a complex object ``O`` (HoloForge convention).

    ``O`` (real part = phase shift in rad, imaginary part = absorption) is zero-padded to ``pad_factor``
    times its size before the exponential, so the wave equals the (optional) probe ``P`` outside the object;
    the probe is padded with ones.  The result is cropped to the object size and returned as float32.
    The stored HoloForge ``images/hologram`` equals this amplitude plus noise; the intensity is its square.
    """
    o = _as_tensor(obj, torch.complex64)
    shape = (o.shape[-2], o.shape[-1])
    psi = torch.exp(1j * pad_center(o, pad_factor, value=0.0))
    if probe is not None:
        p = _as_tensor(probe, torch.complex64)
        psi = psi * (pad_center(p, pad_factor, value=1.0) if p.shape[-2:] != psi.shape[-2:] else p)
    amplitude = torch.abs(propagate(psi, fr, pad_factor=1.0))
    return crop_center(amplitude, shape).to(torch.float32)


def defocus_blur_px(rel_err: ArrayLike, fr: ArrayLike) -> NDArray[np.float64] | float:
    """Defocus blur ``b = sqrt(|e| / Fr)`` in detector pixels caused by a relative Fresnel-number error ``e``.

    Args:
        rel_err: Relative error ``(Fr_est - Fr_true) / Fr_true`` as a fraction (``0.01`` = 1 %).
        fr: True pixel Fresnel number (of the grid the reconstruction runs on).

    Returns:
        Radius of the first Fresnel zone of the residual propagator in pixels (scalar or array).
    """
    err = np.abs(np.asarray(rel_err, dtype=np.float64))
    fr_arr = np.asarray(fr, dtype=np.float64)
    if np.any(fr_arr <= 0):
        raise ValueError("fr must be positive")
    blur = np.sqrt(err / fr_arr)
    return blur if blur.ndim else float(blur)


def blur_px_from_fresnel_numbers(fr_est: ArrayLike, fr_true: ArrayLike) -> NDArray[np.float64] | float:
    """:func:`defocus_blur_px` evaluated for estimated and true Fresnel numbers."""
    est = np.asarray(fr_est, dtype=np.float64)
    true = np.asarray(fr_true, dtype=np.float64)
    return defocus_blur_px((est - true) / true, true)
