"""Detector noise for simulated, flat-field corrected holograms.

Assumption (not taken from [1] or [2]; [2] Sec. 3.5 uses noise-free holograms): the detector
counts photons with Poisson statistics around ``photons * I`` per pixel, where ``I`` is the
flat-field corrected intensity (1 in free space) and ``photons`` the mean count of the flat
field per (binned) pixel. Read-out noise is additive Gaussian with standard deviation
``read_noise`` in counts. The flat field itself is taken as noise-free (many averaged
flats), and detector blur (scintillator PSF) is not modelled.
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
