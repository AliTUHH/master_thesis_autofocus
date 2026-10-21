"""Network inputs computed from one flat-field corrected hologram (1 = free space).

``image``
    Mean over ``binning x binning`` pixels, then standardized per sample (zero mean, unit
    variance), shape ``(1, H / binning, W / binning)``.
``spectrum``
    Radially averaged power spectrum of the Hann-windowed hologram on a uniform ``f^2``
    axis, as natural log and standardized per sample, shape ``(1, bins)``. For a weak phase
    object ``I = 1 + 2 F^-1[sin(pi f^2 / Fr) F[phi]]`` (contrast transfer function), so the
    spectrum is ``sin^2(pi f^2 / Fr)`` times the object power spectrum: its minima lie at
    ``f^2 = k Fr`` and are periodic in ``f^2`` with the Fresnel number as period. Uniform
    bins in ``f^2`` make that period a shift-invariant pattern for the convolutions; the
    radial average is invariant to rotations by 90 degrees and flips.

Frequencies are in cycles per pixel of the stored (binned) grid, as for ``Fr``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import RepresentationConfig


def _standardize(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True)
    return (x - mean) / std.clamp_min(1e-12)


class Representation:
    """Callable mapping holograms ``(B, H, W)`` or ``(H, W)`` to ``(B, *input_shape)``."""

    def __init__(
        self,
        config: RepresentationConfig,
        shape: tuple[int, int],
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.shape = (int(shape[0]), int(shape[1]))
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        h, w = self.shape
        if config.kind == "image":
            if h % config.binning or w % config.binning:
                raise ValueError(f"hologram shape {self.shape} not divisible by the binning")
            self.input_shape = (1, h // config.binning, w // config.binning)
            return
        self.input_shape = (1, config.bins)
        fy = torch.fft.fftfreq(h, dtype=torch.float64)[:, None]
        fx = torch.fft.fftfreq(w, dtype=torch.float64)[None, :]
        index = torch.floor((fx**2 + fy**2) / config.f2_max * config.bins).long().flatten()
        valid = index < config.bins
        counts = torch.bincount(index[valid], minlength=config.bins)
        if int(counts.min()) == 0:
            raise ValueError(
                f"{config.bins} bins up to f^2 = {config.f2_max} leave bins empty on a "
                f"{h}x{w} grid; use fewer bins"
            )
        window = torch.outer(
            torch.hann_window(h, periodic=False, dtype=torch.float64),
            torch.hann_window(w, periodic=False, dtype=torch.float64),
        )
        self._positions = torch.nonzero(valid).flatten().to(self.device)
        self._bin_of = index[valid].to(self.device)
        self._counts = counts.to(self.device, torch.float32)
        self._window = window.to(self.device, torch.float32)

    def __call__(self, holograms: torch.Tensor) -> torch.Tensor:
        x = holograms.to(self.device, torch.float32)
        if x.ndim == 2:
            x = x[None]
        if tuple(x.shape[-2:]) != self.shape:
            raise ValueError(f"expected holograms of shape {self.shape}, got {tuple(x.shape)}")
        if self.config.kind == "image":
            x = x[:, None]
            if self.config.binning > 1:
                x = F.avg_pool2d(x, self.config.binning)
            return _standardize(x, (-2, -1))
        return self._spectrum(x)[:, None]

    def _spectrum(self, x: torch.Tensor) -> torch.Tensor:
        w = self._window
        mean = (x * w).sum(dim=(-2, -1), keepdim=True) / w.sum()
        spectrum = torch.fft.fft2((x - mean) * w)
        power = (spectrum.real.square() + spectrum.imag.square()).flatten(1)
        profile = torch.zeros((x.shape[0], self.config.bins), device=x.device)
        profile.index_add_(1, self._bin_of, power[:, self._positions])
        profile = profile / self._counts
        return _standardize(torch.log(profile.clamp_min(1e-30)), (-1,))
