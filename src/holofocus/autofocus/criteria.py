"""Focus criteria: model fit error ([2] Eq. (14)-(15)) and the image-based criteria of [2].

The model fit error (MFE) compares the hologram predicted by a de-focused wave field that was
projected onto the physical constraint set with the measured one::

    MFE = || |D_{Fr(z12_hat)}(P(D_{Fr(dz01)} psi_exit))| - sqrt(I) ||^2          (Eq. (15))
        = || |D_{-Fr(dz01)} D_{Fr(z12)}(P(D_{Fr(dz01)} psi_exit))| - sqrt(I) ||^2   (Eq. (14))

Both forms are identical because ``1/Fr = 1/Fr(dz01) + 1/Fr_hat`` on the pixel grid (Fresnel
scaling, :meth:`ConeBeamGeometry.fresnel_number_at`). As in the reference code the squared
norm is averaged over the field of view (a mean squared error).

The image criteria VAR, SPEC, GRA, LAP, ToG and GoG are ports of
``holowizard.core.find_focus.focus_loss_metrics`` (MIT license) and act on a real image,
in [2] the reconstructed (or projected) phase. ``prefilter`` reproduces the Gaussian
smoothing (FWHM 4 px) that the reference script applies before evaluating them.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch

from ..asrm import FWHM_TO_SIGMA
from ..constraints import project_wavefield
from ..device import one_thread
from ..propagation import FresnelPropagator


def _crop(array: torch.Tensor, fov: tuple[slice, slice] | None) -> torch.Tensor:
    return array if fov is None else array[..., fov[0], fov[1]]


def _mean_square(
    modulus: torch.Tensor, sqrt_hologram: torch.Tensor, fov: tuple[slice, slice] | None
) -> float:
    squares = (_crop(modulus, fov) - _crop(sqrt_hologram, fov)).square()
    with one_thread():
        return float(squares.mean())


def model_fit_error(
    psi_projected: torch.Tensor,
    sqrt_hologram: torch.Tensor,
    fresnel_number_hat: float,
    fov: tuple[slice, slice] | None = None,
) -> float:
    """[2] Eq. (15): propagate the projected field with the assumed ``Fr_hat`` and compare.

    ``psi_projected`` and ``sqrt_hologram`` live on the same (padded) array; ``fov`` selects
    the detector field of view over which the squared differences are averaged.
    """
    shape = tuple(psi_projected.shape[-2:])
    propagator = FresnelPropagator(
        shape, fresnel_number_hat, psi_projected.device, psi_projected.dtype
    )
    modulus = propagator(psi_projected).abs()
    return _mean_square(modulus, sqrt_hologram, fov)


def model_fit_error_eq14(
    psi_exit: torch.Tensor,
    sqrt_hologram: torch.Tensor,
    fresnel_number: float,
    fresnel_number_defocus: float,
    fov: tuple[slice, slice] | None = None,
) -> float:
    """[2] Eq. (14) from the true exit wave: defocus, project, ``D_Fr``, then undo the defocus.

    ``D^{-dz01}`` is the inverse of ``D^{dz01}``, i.e. Fresnel number ``-Fr(dz01)``.
    """
    shape = tuple(psi_exit.shape[-2:])
    device, dtype = psi_exit.device, psi_exit.dtype
    defocus = FresnelPropagator(shape, fresnel_number_defocus, device, dtype)
    to_detector = FresnelPropagator(shape, fresnel_number, device, dtype)
    projected = project_wavefield(defocus(psi_exit))
    modulus = defocus.inverse()(to_detector(projected)).abs()
    return _mean_square(modulus, sqrt_hologram, fov)


def prefilter(image: torch.Tensor, fwhm: float = 4.0) -> torch.Tensor:
    """Periodic Gaussian smoothing with ``sigma = fwhm / 2.35`` (``plot_loss_chart.py``)."""
    if fwhm == 0:
        return image
    sigma = fwhm * FWHM_TO_SIGMA
    h, w = image.shape[-2:]
    ky = torch.fft.fftfreq(h, dtype=torch.float64, device=image.device)[:, None]
    kx = torch.fft.rfftfreq(w, dtype=torch.float64, device=image.device)[None, :]
    transfer = torch.exp(-2.0 * math.pi**2 * sigma**2 * (ky**2 + kx**2)).to(image.dtype)
    return torch.fft.irfft2(torch.fft.rfft2(image) * transfer, s=(h, w))


def variance(x: torch.Tensor) -> float:
    """VAR: (unbiased) variance of the image."""
    return float(torch.var(x))


def spectral(x: torch.Tensor) -> float:
    """SPEC: ``sum log(1 + |F[x]|)`` without the zero frequency.

    The reference sets ``x[0, 0] = 0`` after the transform, which leaves the zero frequency
    in; it is excluded here as intended (one of ``H W`` terms).
    """
    spectrum = torch.fft.fft2(x).abs()
    spectrum[..., 0, 0] = 0
    return float(torch.log1p(spectrum).sum())


def gradient_sum(x: torch.Tensor) -> float:
    """GRA: sum of the gradient magnitudes (periodic backward differences)."""
    gy = x - torch.roll(x, 1, -2)
    gx = x - torch.roll(x, 1, -1)
    return float(torch.sqrt(gy**2 + gx**2).sum())


def laplacian(x: torch.Tensor) -> float:
    """LAP: sum of the squared 4-neighbour Laplacian (periodic)."""
    lap = (
        torch.roll(x, 1, -2)
        + torch.roll(x, -1, -2)
        + torch.roll(x, 1, -1)
        + torch.roll(x, -1, -1)
        - 4 * x
    )
    return float((lap**2).sum())


def _gradient_magnitudes(x: torch.Tensor) -> torch.Tensor:
    d0 = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    d1 = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    return torch.sqrt(d0[..., :, :-1] ** 2 + d1[..., :-1, :] ** 2).flatten()


def tenengrad_of_gradient(x: torch.Tensor) -> float:
    """ToG: ``sqrt(var(g) / mean(g))`` of the gradient magnitudes ``g`` (forward differences)."""
    g = _gradient_magnitudes(x)
    return float(torch.sqrt(torch.var(g) / torch.mean(g)))


def gini_of_gradient(x: torch.Tensor) -> float:
    """GoG: Gini index of the gradient magnitudes, ``1 - 2 sum (g_i / sum g) (N + 1/2 - i) / N``."""
    g, _ = torch.sort(_gradient_magnitudes(x))
    n = g.numel()
    i = torch.arange(1, n + 1, device=g.device, dtype=g.dtype)
    return float(1 - 2 * (g / g.sum() * (n + 0.5 - i) / n).sum())


IMAGE_CRITERIA: dict[str, Callable[[torch.Tensor], float]] = {
    "VAR": variance,
    "SPEC": spectral,
    "GRA": gradient_sum,
    "LAP": laplacian,
    "ToG": tenengrad_of_gradient,
    "GoG": gini_of_gradient,
}
"""The six image-based criteria of [2] Fig. 7-8 in the order of the figure panels."""

FOCUS_EXTREMUM = {
    "MFE": "min",
    "VAR": "max",
    "SPEC": "max",
    "GRA": "min",
    "LAP": "max",
    "ToG": "max",
    "GoG": "max",
}
"""Extremum that marks the focus for noise-free simulations ([2] Fig. 7); VAR has none."""


def image_criteria(
    image: torch.Tensor, fwhm: float = 4.0, names: tuple[str, ...] | None = None
) -> dict[str, float]:
    """All (or ``names``) image criteria of a real image after :func:`prefilter`."""
    filtered = prefilter(image, fwhm)
    names = tuple(IMAGE_CRITERIA) if names is None else names
    return {name: IMAGE_CRITERIA[name](filtered) for name in names}


def normalize_curve(values) -> np.ndarray:
    """Scale a focus curve to ``[0, 1]`` as in [2] Fig. 7-8."""
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    return np.zeros_like(values) if span == 0 else (values - values.min()) / span


def extremum_offset(offsets, values, kind: str) -> float:
    """Position of the minimum (``kind="min"``) or maximum of a sampled focus curve."""
    values = np.asarray(values, dtype=float)
    index = int(np.argmin(values) if kind == "min" else np.argmax(values))
    return float(np.asarray(offsets, dtype=float)[index])
