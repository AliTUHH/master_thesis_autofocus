"""Input representations of a hologram for autofocus networks: hologram, log power spectrum, radial profile.

Physics background
------------------
HoloForge stores the detector-plane *amplitude* ``a = |D_Fr(psi)|``.  For a weak object the contrast
transfer function (CTF) of the Fresnel propagator ``exp(-i*pi/Fr*(xi^2+eta^2))`` makes the spectrum of
the hologram oscillate with ``sin(pi*|xi|^2/Fr)`` (phase) and ``cos(pi*|xi|^2/Fr)`` (absorption).  In the
power spectrum ``|FFT(a/mean(a) - 1)|^2`` this shows up as concentric rings whose squared radii grow
*linearly* with the ring index: zeros of the phase CTF at ``|xi|^2 = n * Fr`` (``n = 1, 2, ...``).  The
Fresnel number is therefore directly encoded in the ring spacing of the spectrum, which is why the
spectrum (or its radial profile) is a strong input representation for a regressor and the basis of the
classical ring-fit baseline in :mod:`src.baseline.ctf_ringfit`.

Frequency axis conventions
--------------------------
All frequencies are in **cycles per pixel of the tensor that is transformed** (``torch.fft.fftfreq(N)``),
i.e. ``xi, eta in [-0.5, 0.5)`` and ``u = xi^2 + eta^2 in [0, 0.5]`` (``u <= 0.25`` on full circles).
The pixel Fresnel number ``Fr`` of the labels refers to the pixel size of the *stored* hologram:

* **Downsampling** (average pooling) by an integer ``s`` enlarges the pixel by ``s`` and therefore the
  apparent Fresnel number of the transformed array to ``Fr_eff = s^2 * Fr`` -- the rings move outwards
  in cycles/pixel.  :class:`src.data.dataset.HologramHDF5Dataset` does *not* rescale its labels or
  ``setup_constants['px_mm']`` when ``downsample > 1``; a network simply learns the constant offset
  ``log Fr_eff - log Fr = 2 log s``.  Physics-based estimators working on downsampled arrays must divide
  their estimate by ``s^2`` before comparing with the labels.
* **Cropping** to ``N_c`` pixels does not change ``Fr`` (cycles/pixel are unchanged) but coarsens the
  frequency resolution from ``1/N`` to ``1/N_c``; rings at ``|xi|`` stay resolvable only while their
  spacing ``Fr / (2 |xi|)`` in ``|xi|`` exceeds ``1/N_c``.
* Resizing/zooming is **not** allowed as augmentation (it scales the apparent ``Fr`` by the zoom factor
  squared); flips and 90-degree rotations leave the spectrum (up to the same symmetry) and ``Fr`` unchanged.

Radial profile axis
-------------------
:func:`radial_profile` averages the spectrum in bins that are uniform in ``u = |xi|^2`` by default
(``axis="freq_sq"``).  Reasons: (1) the CTF zeros are *equidistant* in ``u`` (spacing ``Fr``), so the
1-D profile becomes a periodic signal whose period *is* the Fresnel number -- a translation-equivariant
1-D model or a periodogram can read it off directly; (2) the annulus between ``u`` and ``u + du`` has
the constant area ``pi * du * N^2`` pixels, so every bin averages the same number of spectrum pixels
(uniform noise statistics).  ``axis="freq"`` (uniform in ``|xi|``) is available for comparison.

All functions are pure tensor functions (``torch``), operate on the last two dimensions, accept
``[H, W]``, ``[C, H, W]`` and ``[B, C, H, W]`` inputs and run on any device.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

import torch

__all__ = [
    "Representation",
    "REPRESENTATIONS",
    "RADIAL_AXES",
    "relative_contrast",
    "hann_window_2d",
    "power_spectrum",
    "log_power_spectrum",
    "radial_profile",
    "radial_bin_centres",
    "radial_log_power_profile",
    "frequency_grid",
    "RepresentationTransform",
    "build_representation",
]


class Representation(str, Enum):
    """Names accepted by ``data.representation`` in experiment configs."""

    HOLOGRAM = "hologram"
    LOG_POWER_SPECTRUM = "log_power_spectrum"
    HOLOGRAM_SPECTRUM = "hologram+spectrum"
    RADIAL_PROFILE = "radial_profile"

    @property
    def channels(self) -> int:
        """Number of image channels produced (1 for the 1-D radial profile)."""
        return 2 if self is Representation.HOLOGRAM_SPECTRUM else 1

    @property
    def is_1d(self) -> bool:
        return self is Representation.RADIAL_PROFILE

    @property
    def uses_spectrum(self) -> bool:
        return self is not Representation.HOLOGRAM


REPRESENTATIONS: tuple[str, ...] = tuple(r.value for r in Representation)
RADIAL_AXES: tuple[str, ...] = ("freq_sq", "freq")

_EPS = 1e-12


def relative_contrast(x: torch.Tensor) -> torch.Tensor:
    """``x / mean(x) - 1`` per image (over the last two dims): removes the DC term and the flat-field scale."""
    mean = x.mean(dim=(-2, -1), keepdim=True)
    safe = torch.where(mean.abs() > _EPS, mean, torch.ones_like(mean))
    return x / safe - 1.0


@lru_cache(maxsize=16)
def _hann_window_2d_cached(height: int, width: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    dev = torch.device(device)
    wy = torch.hann_window(height, periodic=False, device=dev, dtype=dtype)
    wx = torch.hann_window(width, periodic=False, device=dev, dtype=dtype)
    return torch.outer(wy, wx)


def hann_window_2d(height: int, width: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Separable symmetric Hann window ``[H, W]`` (cached per shape/device/dtype)."""
    return _hann_window_2d_cached(int(height), int(width), str(torch.device(device)), dtype)


@lru_cache(maxsize=16)
def _frequency_grid_cached(height: int, width: int, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device(device)
    xi = torch.fft.fftshift(torch.fft.fftfreq(height, device=dev, dtype=dtype))
    eta = torch.fft.fftshift(torch.fft.fftfreq(width, device=dev, dtype=dtype))
    return torch.meshgrid(xi, eta, indexing="ij")


def frequency_grid(height: int, width: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """fftshifted ``(xi, eta)`` grids in cycles/pixel matching the output of :func:`power_spectrum`."""
    return _frequency_grid_cached(int(height), int(width), str(torch.device(device)), dtype)


def power_spectrum(x: torch.Tensor, window: bool = True) -> torch.Tensor:
    """``|FFT2(relative contrast [* Hann])|^2 / (H*W)``, fftshifted, same shape as ``x``.

    Dividing by the number of pixels makes the value for white noise of variance ``sigma^2`` equal to
    ``sigma^2`` independent of the image size (``E[P] = sigma^2``), so the log scale below is comparable
    across crop sizes.
    """
    if x.ndim < 2:
        raise ValueError(f"expected at least 2 dims [..., H, W], got shape {tuple(x.shape)}")
    height, width = x.shape[-2], x.shape[-1]
    y = relative_contrast(x.float() if not x.is_floating_point() else x)
    if window:
        y = y * hann_window_2d(height, width, y.device, y.dtype)
    spectrum = torch.fft.fftshift(torch.fft.fft2(y), dim=(-2, -1))
    return (spectrum.real**2 + spectrum.imag**2) / float(height * width)


def _standardize(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True, unbiased=False)
    return (x - mean) / torch.where(std > _EPS, std, torch.ones_like(std))


def _log_scale_factor(spectrum: torch.Tensor, log_scale: float | str) -> torch.Tensor | float:
    """Fixed factor, or ``"auto"``: the inverse of the per-image noise floor (median outside ``|xi| > 0.4``)."""
    if isinstance(log_scale, str):
        if log_scale != "auto":
            raise ValueError(f"log_scale must be a positive number or 'auto', got {log_scale!r}")
        xi, eta = frequency_grid(spectrum.shape[-2], spectrum.shape[-1], spectrum.device, spectrum.dtype)
        outer = (xi**2 + eta**2) > 0.16
        flat = spectrum.reshape(-1, spectrum.shape[-2] * spectrum.shape[-1])
        floor = flat[:, outer.reshape(-1)].median(dim=-1).values.clamp_min(_EPS)
        return (1.0 / floor).reshape(*spectrum.shape[:-2], 1, 1)
    if log_scale <= 0:
        raise ValueError(f"log_scale must be positive, got {log_scale}")
    return float(log_scale)


def log_power_spectrum(
    x: torch.Tensor,
    window: bool = True,
    log_scale: float | str = "auto",
    standardize: bool = True,
) -> torch.Tensor:
    """``log1p(scale * P)`` of the fftshifted power spectrum, optionally standardised per image.

    Args:
        x: Hologram amplitude(s) ``[..., H, W]``.
        window: Multiply the relative contrast with a Hann window to suppress the leakage cross caused by
            the non-periodic image border (costs a factor ~2 in frequency resolution).
        log_scale: Factor inside ``log1p``; ``"auto"`` scales each spectrum so that its noise floor
            (median power at ``|xi| > 0.4``) maps to ``log1p(1)`` -- values below the floor are compressed,
            the signal above it is log-compressed.
        standardize: Zero mean / unit variance per image after the log.
    """
    spectrum = power_spectrum(x, window=window)
    out = torch.log1p(_log_scale_factor(spectrum, log_scale) * spectrum)
    return _standardize(out, (-2, -1)) if standardize else out


@lru_cache(maxsize=16)
def _radial_bins_cached(height: int, width: int, n_bins: int, axis: str, max_value: float, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bin index per spectrum pixel (``n_bins`` = outside the range), pixel counts per bin and bin centres."""
    xi, eta = frequency_grid(height, width, device, torch.float64)
    coord = xi**2 + eta**2 if axis == "freq_sq" else torch.sqrt(xi**2 + eta**2)
    index = torch.floor(coord / max_value * n_bins).long()
    index = torch.where((index >= 0) & (index < n_bins), index, torch.full_like(index, n_bins))
    counts = torch.bincount(index.reshape(-1), minlength=n_bins + 1)[:n_bins]
    edges = torch.linspace(0.0, max_value, n_bins + 1, dtype=torch.float64, device=torch.device(device))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return index.reshape(-1), counts, centres


def _default_max_value(axis: str) -> float:
    return 0.25 if axis == "freq_sq" else 0.5


def radial_bin_centres(n_bins: int, axis: str = "freq_sq", max_value: float | None = None) -> torch.Tensor:
    """Centres of the radial bins (``u`` for ``freq_sq``, ``|xi|`` for ``freq``) as float64 tensor."""
    if axis not in RADIAL_AXES:
        raise ValueError(f"axis must be one of {RADIAL_AXES}, got {axis!r}")
    max_value = _default_max_value(axis) if max_value is None else float(max_value)
    edges = torch.linspace(0.0, max_value, n_bins + 1, dtype=torch.float64)
    return 0.5 * (edges[:-1] + edges[1:])


def _fill_empty_bins(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Replace bins without pixels (possible at small radii on coarse grids) by the nearest filled bin to the left
    (or right for leading bins)."""
    n_bins = values.shape[-1]
    if bool((counts > 0).all()):
        return values
    filled = counts > 0
    positions = torch.arange(n_bins, device=values.device)
    left = torch.where(filled, positions, torch.full_like(positions, -1))
    left = torch.cummax(left, dim=0).values
    right = torch.where(filled, positions, torch.full_like(positions, n_bins))
    right = torch.flip(torch.cummin(torch.flip(right, dims=(0,)), dim=0).values, dims=(0,))
    source = torch.where(left >= 0, left, right).clamp(0, n_bins - 1)
    return values[..., source]


def radial_profile(
    spectrum: torch.Tensor,
    n_bins: int = 256,
    axis: str = "freq_sq",
    max_value: float | None = None,
) -> torch.Tensor:
    """Azimuthal mean of an fftshifted spectrum ``[..., H, W]`` -> ``[..., n_bins]``.

    Bins are uniform in ``u = xi^2 + eta^2`` (``axis="freq_sq"``, default range ``[0, 0.25]`` = full
    circles inside the Nyquist square) or in ``|xi|`` (``axis="freq"``, default ``[0, 0.5]``).  Pixels
    outside the range are ignored; empty bins are filled from their neighbours.
    """
    if spectrum.ndim < 2:
        raise ValueError(f"expected [..., H, W], got shape {tuple(spectrum.shape)}")
    if axis not in RADIAL_AXES:
        raise ValueError(f"axis must be one of {RADIAL_AXES}, got {axis!r}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    max_value = _default_max_value(axis) if max_value is None else float(max_value)
    height, width = spectrum.shape[-2], spectrum.shape[-1]
    index, counts, _ = _radial_bins_cached(height, width, int(n_bins), axis, max_value, str(spectrum.device))
    flat = spectrum.reshape(-1, height * width)
    sums = torch.zeros(flat.shape[0], n_bins + 1, dtype=flat.dtype, device=flat.device)
    sums.index_add_(1, index, flat)
    means = sums[:, :n_bins] / counts.clamp_min(1).to(flat.dtype)
    means = _fill_empty_bins(means, counts)
    return means.reshape(*spectrum.shape[:-2], n_bins)


def radial_log_power_profile(
    x: torch.Tensor,
    n_bins: int = 256,
    axis: str = "freq_sq",
    window: bool = True,
    log_scale: float | str = "auto",
    standardize: bool = True,
    max_value: float | None = None,
) -> torch.Tensor:
    """1-D representation: ``log1p(scale * radial_mean(P))`` of the hologram ``[..., H, W]`` -> ``[..., n_bins]``.

    The power is averaged *before* the log (unbiased azimuthal power), then log-compressed with the same
    scale rule as :func:`log_power_spectrum` and standardised per sample.
    """
    spectrum = power_spectrum(x, window=window)
    scale = _log_scale_factor(spectrum, log_scale)
    if isinstance(scale, torch.Tensor):
        scale = scale.reshape(*spectrum.shape[:-2], 1)
    profile = torch.log1p(scale * radial_profile(spectrum, n_bins=n_bins, axis=axis, max_value=max_value))
    return _standardize(profile, (-1,)) if standardize else profile


@dataclass(frozen=True)
class RepresentationTransform:
    """Maps a preprocessed hologram ``[1, H, W]`` (or batch ``[B, 1, H, W]``) to the network input.

    ``__call__`` takes the *raw* (cropped/downsampled) amplitude and the already normalised hologram
    (per ``data.normalization``); spectral channels are always computed from the raw amplitude via the
    relative contrast, the hologram channel keeps the configured normalisation.
    """

    representation: Representation = Representation.HOLOGRAM
    window: bool = True
    log_scale: float | str = "auto"
    n_bins: int = 256
    axis: str = "freq_sq"

    def __post_init__(self) -> None:
        if self.axis not in RADIAL_AXES:
            raise ValueError(f"axis must be one of {RADIAL_AXES}, got {self.axis!r}")
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be positive, got {self.n_bins}")
        _log_scale_factor(torch.ones(1, 8, 8), self.log_scale)  # validates the value

    @property
    def channels(self) -> int:
        return self.representation.channels

    @property
    def is_1d(self) -> bool:
        return self.representation.is_1d

    def output_shape(self, image_shape: tuple[int, int, int]) -> tuple[int, ...]:
        """Output shape for a ``(1, H, W)`` input."""
        _, height, width = image_shape
        if self.is_1d:
            return (self.n_bins,)
        return (self.channels, height, width)

    def __call__(self, raw: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
        rep = self.representation
        if rep is Representation.HOLOGRAM:
            return normalized
        if rep is Representation.LOG_POWER_SPECTRUM:
            return log_power_spectrum(raw, window=self.window, log_scale=self.log_scale)
        if rep is Representation.HOLOGRAM_SPECTRUM:
            spectrum = log_power_spectrum(raw, window=self.window, log_scale=self.log_scale)
            return torch.cat([normalized, spectrum], dim=-3)
        profile = radial_log_power_profile(raw, n_bins=self.n_bins, axis=self.axis, window=self.window, log_scale=self.log_scale)
        return profile.squeeze(-2) if profile.ndim >= 2 and profile.shape[-2] == 1 else profile

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation.value,
            "window": self.window,
            "log_scale": self.log_scale,
            "n_bins": self.n_bins,
            "axis": self.axis,
        }


def build_representation(name: str | Representation = "hologram", **kwargs: Any) -> RepresentationTransform:
    """Factory: ``build_representation("hologram+spectrum", window=False, log_scale=1e3)``.

    Accepted keyword arguments: ``window`` (bool), ``log_scale`` (float or ``"auto"``), ``n_bins`` and
    ``axis`` (radial profile only).
    """
    try:
        representation = Representation(name)
    except ValueError as exc:
        raise ValueError(f"unknown representation {name!r}; expected one of {REPRESENTATIONS}") from exc
    allowed = {"window", "log_scale", "n_bins", "axis"}
    unknown = set(kwargs) - allowed
    if unknown:
        raise ValueError(f"unknown representation option(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    return RepresentationTransform(representation=representation, **kwargs)
