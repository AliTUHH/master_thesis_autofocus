"""Detector noise for simulated, flat-field corrected holograms.

Assumption (not taken from [1] or [2]; [2] Sec. 3.5 uses noise-free holograms): the detector
counts photons with Poisson statistics around ``photons * I`` per pixel, where ``I`` is the
flat-field corrected intensity (1 in free space) and ``photons`` the mean count of the flat
field per (binned) pixel. Read-out noise is additive Gaussian with standard deviation
``read_noise`` in counts. The flat field itself is taken as noise-free (many averaged
flats). Detector blur is part of the forward model (:func:`holofocus.forward.detector_blur`).

:func:`flat_field_residual` (Stage 5) adds what the flat-field correction leaves in the
measured holograms of [1]: a spatially correlated, multiplicative background that dominates
the noise after 4x binning (lab notebook 02, Sec. 5.4: 1.5-2 % on the 4x grid, power falling
by about an order of magnitude from low frequencies to the Nyquist frequency).
"""

from __future__ import annotations

import numpy as np

NOISE_MODELS = ("none", "poisson", "poisson_gaussian")


def add_detector_noise(
    intensity: np.ndarray,
    rng: np.random.Generator,
    model: str = "poisson",
    photons: float = 1.0e4,
    read_noise: float = 0.0,
) -> np.ndarray:
    """Noisy flat-field corrected hologram ``(Poisson(N I) + Normal(0, s^2)) / N``.

    ``model`` is ``"none"`` (returns a copy), ``"poisson"`` or ``"poisson_gaussian"``. The
    result keeps the mean of ``intensity``; its variance is ``I / N + (s / N)^2``.
    """
    if model not in NOISE_MODELS:
        raise ValueError(f"unknown noise model {model!r}, choose from {NOISE_MODELS}")
    intensity = np.asarray(intensity, dtype=np.float64)
    if model == "none":
        return intensity.copy()
    if photons <= 0:
        raise ValueError("photons must be positive")
    counts = rng.poisson(photons * np.clip(intensity, 0.0, None)).astype(np.float64)
    if model == "poisson_gaussian" and read_noise > 0:
        counts += rng.normal(0.0, read_noise, size=counts.shape)
    return counts / photons


def flat_field_residual(
    shape: tuple[int, int], rng: np.random.Generator, rms: float, cutoff: float = 0.15
) -> np.ndarray:
    """Relative flat-field error ``r``; the corrected hologram is ``I (1 + r)``.

    A Gaussian random field with zero mean, standard deviation ``rms`` and the power spectrum
    ``1 / (1 + (f / cutoff)^2)`` (``f`` in cycles per pixel), periodic over ``shape``.
    """
    if rms < 0 or cutoff <= 0:
        raise ValueError("need rms >= 0 and cutoff > 0")
    if rms == 0:
        return np.zeros(shape)
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.rfftfreq(shape[1])[None, :]
    amplitude = 1.0 / np.sqrt(1.0 + (fx**2 + fy**2) / cutoff**2)
    field = np.fft.irfft2(np.fft.rfft2(rng.standard_normal(shape)) * amplitude, s=shape)
    field -= field.mean()
    return field * (rms / field.std())
