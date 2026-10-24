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
``polar``
    The same power spectrum resolved in the direction ``theta = atan2(fy, fx)`` modulo 180
    degrees (a real hologram has ``P(-f) = P(f)``): ``angles`` sectors centred on
    ``theta = k 180 / angles`` degrees times the same ``bins`` of ``f^2``, shape
    ``(1, angles, bins)``, natural log and standardized over all cells. Every frequency goes
    to its two nearest sector centres and its two nearest bin centres with linear weights
    (cloud in cell): near ``f^2 = 0.125`` a bin of 1024 is a ring 0.09 pixels wide whose few
    grid frequencies leave some sectors of 22.5 degrees empty on 512^2 and 472^2 grids. The
    zero frequency is left out. A fiber puts its CTF pattern into the sectors across its
    axis, which the radial average mixes with the texture of all directions. A rotation by 90
    degrees shifts the sectors cyclically by ``angles / 2``, a flip reverses them
    (``k -> -k mod angles``).

Frequencies are in cycles per pixel of the stored (binned) grid, as for ``Fr``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import RepresentationConfig


def _standardize(x: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    mean = x.mean(dim=dims, keepdim=True)
    std = x.std(dim=dims, keepdim=True)
    return (x - mean) / std.clamp_min(1e-12)


def _polar_cells(
    fy: torch.Tensor, fx: torch.Tensor, config: RepresentationConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flat frequency index, cell ``sector * bins + bin`` and linear weight of every
    contribution to the polar spectrum (four per frequency ``0 < f^2 < f2_max``)."""
    a, bins = config.angles, config.bins
    f2 = (fx**2 + fy**2).flatten()
    theta = torch.remainder(torch.atan2(fy, fx), math.pi).flatten()
    inside = torch.nonzero((f2 > 0) & (f2 < config.f2_max)).flatten()
    u = theta[inside] / (math.pi / a)
    v = f2[inside] / config.f2_max * bins - 0.5
    s0, b0 = torch.floor(u), torch.floor(v)
    t, r = u - s0, v - b0
    s0, b0 = s0.long(), b0.long()
    positions, cells, weights = [], [], []
    for ds, ws in ((0, 1 - t), (1, t)):
        for db, wb in ((0, 1 - r), (1, r)):
            b, weight = b0 + db, ws * wb
            keep = (b >= 0) & (b < bins) & (weight > 0)
            positions.append(inside[keep])
            cells.append((s0[keep] + ds) % a * bins + b[keep])
            weights.append(weight[keep])
    return torch.cat(positions), torch.cat(cells), torch.cat(weights)


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
        fy = torch.fft.fftfreq(h, dtype=torch.float64)[:, None]
        fx = torch.fft.fftfreq(w, dtype=torch.float64)[None, :]
        if config.kind == "polar":
            positions, cells, weights = _polar_cells(fy, fx, config)
            self.input_shape = (1, config.angles, config.bins)
        else:
            index = torch.floor((fx**2 + fy**2) / config.f2_max * config.bins).long().flatten()
            valid = index < config.bins
            positions, cells, weights = torch.nonzero(valid).flatten(), index[valid], None
            self.input_shape = (1, config.bins)
        n_cells = math.prod(self.input_shape[1:])
        if weights is None:
            counts = torch.bincount(cells, minlength=n_cells).double()
        else:
            counts = torch.zeros(n_cells, dtype=torch.float64).index_add_(0, cells, weights)
        if float(counts.min()) == 0:
            raise ValueError(
                f"{self.input_shape[1:]} cells up to f^2 = {config.f2_max} leave cells empty "
                f"on a {h}x{w} grid; use fewer bins or angles"
            )
        window = torch.outer(
            torch.hann_window(h, periodic=False, dtype=torch.float64),
            torch.hann_window(w, periodic=False, dtype=torch.float64),
        )
        self._positions = positions.to(self.device)
        self._bin_of = cells.to(self.device)
        self._weights = None if weights is None else weights.to(self.device, torch.float32)
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
        power = (spectrum.real.square() + spectrum.imag.square()).flatten(1)[:, self._positions]
        if self._weights is not None:
            power = power * self._weights
        profile = torch.zeros((x.shape[0], self._counts.numel()), device=x.device)
        profile.index_add_(1, self._bin_of, power)
        profile = profile / self._counts
        profile = profile.reshape(x.shape[0], *self.input_shape[1:])
        dims = tuple(range(-len(self.input_shape) + 1, 0))
        return _standardize(torch.log(profile.clamp_min(1e-30)), dims)
