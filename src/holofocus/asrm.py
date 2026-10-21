"""ASRM, the artifact-suppressing reconstruction method of [1] Alg. 5, as used by [2].

Port of the reference implementation (holowizard 3.0.6, ``core/reconstruction`` and
``core/preprocessing``, MIT license; it reproduces livereco v1.0.0 that was used for [1] and
[2]). One iteration of a stage (regularization level ``j``) reads::

    O' = G_O fade(pad(mirr(O)))                       object padding and filter Omega_F
    v  = G_v v                                        momentum operator tau = gamma G_v
    y  = O' - gamma v                                 Nesterov look-ahead
    v  = gamma v + eta grad(y) + i beta Im(y) / ||Im(y)||_2
    O  = P_OmegaP(y - v)

The data term is ``1/2 || |D_Fr(exp(i y))| - sqrt(S_down I) ||^2`` and ``P_OmegaP`` clamps
``Re(O) <= 0`` and ``Im(O) >= 0`` ([2] Eq. (20)-(21), with the data divided by ``a0``).
Stages run on down-sampled data and are joined by bilinear up-sampling of ``O`` and ``v``.

Where the reference code differs from the printed algorithm, the reference behaviour is the
default and the printed one is available as an option:

* ``update="reference"`` takes the position step from the look-ahead point ``y``;
  [1] Alg. 5 prints ``P(O' - v)`` (``update="printed"``).
* ``gradient="reference"`` uses ``psi (1 - sqrt(A / |psi|))`` as detector-plane residual,
  where ``A`` is the measured amplitude; [1] Eq. (22) has ``psi (1 - A / |psi|)``
  (``gradient="amplitude"``). Near convergence the reference step is half as large.
* ``pin_phase=True`` shifts the phase inside the field of view so that its maximum is 0
  before every padding step (``flip_and_pad(..., shift_phases=True)`` of the reference).

The objective of the autofocus of [2] is the loss of the last iteration: the mean over the
field of view of ``(|D_Fr(exp(i y))| - A)^2``, i.e. the model fit error of [2] Eq. (15).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import torch
import torch.nn.functional as F

from .device import complex_dtype_of, real_dtype_of
from .propagation import FresnelPropagator

UPDATE_RULES = ("reference", "printed")
GRADIENT_RULES = ("reference", "amplitude")
REFERENCE_FADE_WIDTH = 80
"""Fade width of the reference code (``DataDimensions.fading_width``) on the 2048^2 detector."""
FWHM_TO_SIGMA = 1.0 / 2.35
"""Conversion used by the reference code (exactly ``1 / (2 sqrt(2 ln 2)) = 1 / 2.3548``)."""


@dataclass(frozen=True)
class ASRMStage:
    """One regularization level of [1] Alg. 5, i.e. one row of [2] Table 3.

    Filter widths are full widths at half maximum in pixels of the stage grid; ``0`` disables
    the filter. ``down_sampling`` refers to the grid of the hologram passed to
    :func:`asrm_reconstruct`.
    """

    iterations: int
    step: float
    down_sampling: int = 1
    momentum: float = 1.0
    momentum_fwhm: float = 0.0
    phase_fwhm: float = 0.0
    absorption_fwhm: float = 0.0
    l2_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.iterations < 0 or self.down_sampling < 1:
            raise ValueError("need iterations >= 0 and down_sampling >= 1")
        if min(self.momentum_fwhm, self.phase_fwhm, self.absorption_fwhm) < 0:
            raise ValueError("filter widths must be >= 0")


PAPER2_TABLE3: tuple[ASRMStage, ...] = (
    ASRMStage(700, 0.9, 16, 1.0, 8.0, 2.0, 0.0, 10.0),
    ASRMStage(300, 1.1, 4, 1.0, 16.0, 2.0, 8.0, 10.0),
    ASRMStage(500, 1.1, 4, 1.0, 16.0, 2.0, 8.0, 1.0),
)
"""[2] Appendix Table 3 for holograms of the full 2048^2 detector (6.5 um pixels)."""


def schedule_for_binning(stages: Sequence[ASRMStage], binning: int) -> tuple[ASRMStage, ...]:
    """Express the down-sampling of ``stages`` relative to a detector binned by ``binning``.

    Stage grids keep their physical pixel size where the binned data allow it (16x of [2]
    becomes 4x on 4x-binned data); stages finer than the binned grid run at its resolution.
    Filter widths stay in pixels of the stage grid, so they keep their physical meaning
    whenever the stage grid does.
    """
    if binning < 1:
        raise ValueError("binning must be >= 1")
    return tuple(replace(s, down_sampling=max(1, round(s.down_sampling / binning))) for s in stages)


@dataclass(frozen=True)
class ASRMSettings:
    """Everything :func:`asrm_reconstruct` needs besides the hologram and ``Fr``.

    ``padding_factor`` sets the array size ``round(p n / S_down)`` of every stage (the
    reference code calls it ``padding_factor``; [2] used 7 for 2048^2 holograms).
    ``fade_width`` is given in pixels of the input hologram and divided by ``S_down``
    per stage. ``a0`` divides the measured amplitude (constant probe ``A0`` of [1] Eq. (4)).
    """

    stages: tuple[ASRMStage, ...] = PAPER2_TABLE3
    padding_factor: float = 2.0
    fade_width: int = REFERENCE_FADE_WIDTH
    a0: float = 1.0
    update: str = "reference"
    gradient: str = "reference"
    pin_phase: bool = True

    def __post_init__(self) -> None:
        if self.update not in UPDATE_RULES:
            raise ValueError(f"update must be one of {UPDATE_RULES}, got {self.update!r}")
        if self.gradient not in GRADIENT_RULES:
            raise ValueError(f"gradient must be one of {GRADIENT_RULES}, got {self.gradient!r}")
        if self.padding_factor < 1:
            raise ValueError("padding_factor must be >= 1")
        if self.fade_width < 0 or self.a0 <= 0:
            raise ValueError("need fade_width >= 0 and a0 > 0")
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.stages:
            raise ValueError("need at least one stage")

    @classmethod
    def for_binning(cls, binning: int, **kwargs) -> ASRMSettings:
        """[2] Table 3 and the reference fade width for a detector binned by ``binning``."""
        return cls(
            stages=schedule_for_binning(kwargs.pop("stages", PAPER2_TABLE3), binning),
            fade_width=kwargs.pop("fade_width", REFERENCE_FADE_WIDTH // binning),
            **kwargs,
        )

    @property
    def iterations(self) -> int:
        return sum(s.iterations for s in self.stages)


@dataclass
class ASRMResult:
    """Output of :func:`asrm_reconstruct`.

    ``obj`` is ``O~`` in the field of view on the grid of the last stage, ``padded`` the full
    array of that stage. ``loss[k]`` is the model fit error at the look-ahead point of
    iteration ``k``; ``stage_ends`` holds the cumulative iteration counts.
    """

    obj: torch.Tensor
    padded: torch.Tensor
    loss: list[float]
    stage_ends: list[int] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def final_loss(self) -> float:
        return self.loss[-1] if self.loss else math.nan


@dataclass(frozen=True)
class StageGrid:
    """Array sizes of one stage: field of view ``fov_shape`` inside ``shape``."""

    shape: tuple[int, int]
    fov_shape: tuple[int, int]

    @property
    def fov(self) -> tuple[slice, slice]:
        """``crop_center`` convention of the reference code: start ``N // 2 - n // 2``."""
        return tuple(
            slice(n // 2 - m // 2, n // 2 - m // 2 + m)
            for n, m in zip(self.shape, self.fov_shape, strict=True)
        )


def stage_grid(
    input_shape: tuple[int, int], down_sampling: int, padding_factor: float
) -> StageGrid:
    """Grid of a stage: ``n / S_down`` field of view in ``round(p n / S_down)`` pixels."""
    if any(n % down_sampling for n in input_shape):
        raise ValueError(f"hologram shape {input_shape} not divisible by {down_sampling}")
    fov = tuple(n // down_sampling for n in input_shape)
    shape = tuple(
        max(round(n * padding_factor / down_sampling), m)
        for n, m in zip(input_shape, fov, strict=True)
    )
    return StageGrid(shape=shape, fov_shape=fov)


def fade_profile(
    size: int, start: int, stop: int, width: int, device=None, dtype=torch.float32
) -> torch.Tensor:
    """1-D window of the reference code (``get_2d_window_from_function``, Blackman).

    Zero outside ``[start, stop)``, one inside, with Blackman flanks of ``width // 2`` pixels
    at both ends. As in the reference, the Blackman window of length ``width`` is sampled at
    ``linspace(0, width, width)`` (slightly wider than ``0 .. width - 1``). ``start`` and
    ``stop`` may lie outside ``[0, size)``; the profile is then cut to the array.
    """
    lo, hi = min(0, start), max(size, stop)
    profile = torch.zeros(hi - lo, dtype=torch.float64)
    s0, s1 = start - lo, stop - lo
    half = width // 2
    if s1 - s0 < 2 * half:
        raise ValueError("fade width larger than the faded region")
    profile[s0:s1] = 1.0
    if half > 0:
        x = torch.linspace(0, width, width, dtype=torch.float64)
        values = (
            0.42
            - 0.5 * torch.cos(2 * math.pi * x / (width - 1))
            + 0.08 * torch.cos(4 * math.pi * x / (width - 1))
        )
        offset = width - 1 - 2 * half
        profile[s0 : s0 + half] = values[:half]
        profile[s1 - half : s1] = values[half + offset : width - 1]
    return profile[-lo : -lo + size].to(device=device, dtype=dtype)


def stage_window(grid: StageGrid, fade_width: int, device=None, dtype=torch.float32):
    """Fade window of a stage (``MIRROR_ALL``): ones on the 3x mirrored block, faded edges.

    The block extends one field of view beyond the original on every side; its outer
    ``fade_width // 2`` pixels are faded. For ``p < 4`` the reference computes the window on
    a ``4 n`` array and crops it, which this function reproduces.
    """
    profiles = []
    for rows, n, m in zip(grid.fov, grid.shape, grid.fov_shape, strict=True):
        work = n if n >= 4 * m else 4 * m
        offset = (work // 2 - m // 2) - rows.start
        start = round(work / 2) - round(m / 2) - m
        stop = round(work / 2) + round(m / 2) + m
        start, stop = max(start, 0), min(stop, work)
        full = fade_profile(work, start, stop, fade_width, device, dtype)
        profiles.append(full[offset : offset + n])
    return profiles[0][:, None] * profiles[1][None, :]


def mirror_embed(fov: torch.Tensor, grid: StageGrid, fill: float = 0.0) -> torch.Tensor:
    """Mirror the field of view into all directions and centre it in the stage array.

    ``ReflectionPad2d(min(n) - 1)`` of the reference; the field of view lands on
    :attr:`StageGrid.fov`, mirrored copies that exceed the array are cut, the rest is
    ``fill``.
    """
    pad = min(fov.shape[-2:]) - 1
    flat = fov.reshape(1, 1, *fov.shape[-2:])
    mirrored = F.pad(flat, (pad, pad, pad, pad), mode="reflect")[0, 0]
    out = torch.full(grid.shape, fill, dtype=fov.dtype, device=fov.device)
    src, dst = [], []
    for rows, n, size in zip(grid.fov, grid.shape, mirrored.shape, strict=True):
        start = rows.start - pad
        a, b = max(0, -start), max(0, start)
        length = min(size - a, n - b)
        src.append(slice(a, a + length))
        dst.append(slice(b, b + length))
    out[dst[0], dst[1]] = mirrored[src[0], src[1]]
    return out


def prepare_stage_data(
    amplitude: torch.Tensor, grid: StageGrid, down_sampling: int, fade_width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Down-sample, mirror, pad and fade the normalized amplitude for one stage.

    ``amplitude`` is ``sqrt(I) / a0`` on the input grid. Returns ``(data, window)`` on the
    stage array, with ``data = (mirror_embed(S_down A) - 1) * window + 1`` as in the
    reference ``process_image``.
    """
    if down_sampling > 1:
        amplitude = F.interpolate(
            amplitude[None, None],
            size=grid.fov_shape,
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )[0, 0]
    window = stage_window(grid, fade_width // down_sampling, amplitude.device, amplitude.dtype)
    data = (mirror_embed(amplitude, grid, fill=1.0) - 1.0) * window + 1.0
    return data, window


def gaussian_transfer(
    shape: tuple[int, int], fwhm: float, device=None, dtype=torch.float32, onesided: bool = False
) -> torch.Tensor | None:
    """Transfer function ``exp(-2 pi^2 sigma^2 k^2)`` of a periodic Gaussian filter.

    Identical to ``scipy.ndimage.fourier_gaussian(ones(shape), sigma)`` used by the reference
    code, with ``sigma = fwhm / 2.35``. ``onesided`` returns the ``rfft2`` half-plane.
    ``None`` for ``fwhm == 0`` (no filter).
    """
    if fwhm == 0:
        return None
    sigma = fwhm * FWHM_TO_SIGMA
    ky = torch.fft.fftfreq(shape[0], dtype=torch.float64)[:, None]
    if onesided:
        kx = torch.fft.rfftfreq(shape[1], dtype=torch.float64)[None, :]
    else:
        kx = torch.fft.fftfreq(shape[1], dtype=torch.float64)[None, :]
    transfer = torch.exp(-2.0 * math.pi**2 * sigma**2 * (ky**2 + kx**2))
    return transfer.to(device=device, dtype=dtype)


def _filter_real(x: torch.Tensor, transfer: torch.Tensor | None) -> torch.Tensor:
    if transfer is None:
        return x
    return torch.fft.irfft2(torch.fft.rfft2(x) * transfer, s=x.shape[-2:])


def _filter_complex(x: torch.Tensor, transfer: torch.Tensor | None) -> torch.Tensor:
    if transfer is None:
        return x
    return torch.fft.ifft2(torch.fft.fft2(x) * transfer)


def resize_complex(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize of real and imaginary part (``resize_guess`` of the reference)."""
    if tuple(x.shape[-2:]) == tuple(shape):
        return x
    parts = torch.stack((x.real, x.imag))[:, None]
    out = F.interpolate(parts, size=shape, mode="bilinear", antialias=True, align_corners=False)
    return torch.complex(out[0, 0], out[1, 0])


def pad_object(
    obj: torch.Tensor, grid: StageGrid, window: torch.Tensor, pin_phase: bool = True
) -> torch.Tensor:
    """``fade(pad(mirr(O)))`` of [1] Alg. 5 (``apply_padding_refractive`` of the reference)."""
    fov = obj[grid.fov]
    phase, absorption = fov.real, fov.imag
    if pin_phase:
        top = phase.max()
        phase = torch.where(top <= 0, phase - top, phase)
    phase = mirror_embed(phase, grid) * window
    absorption = mirror_embed(absorption, grid) * window
    return torch.complex(phase, absorption)


def data_gradient(
    lookahead: torch.Tensor,
    data: torch.Tensor,
    forward: torch.Tensor,
    rule: str = "reference",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient of the data term at ``y`` and the modelled detector amplitude ``|D(exp(iy))|``.

    ``forward`` is the transfer function of ``D_Fr``. With ``a = |D(exp(iy))|`` and the data
    ``A``, ``rule="amplitude"`` is the gradient of ``1/2 sum (a - A)^2`` ([1] Eq. (22)) and
    ``rule="reference"`` the gradient of ``sum (a^2 / 2 - 2/3 sqrt(A) a^(3/2))``, which is
    what the residual ``psi (1 - sqrt(A / a))`` of the reference code differentiates.
    Real and imaginary part of the result are the derivatives by ``Re y`` and ``Im y``.
    """
    psi = torch.exp(1j * lookahead)
    detector = torch.fft.ifft2(torch.fft.fft2(psi) * forward)
    modulus = detector.abs()
    ratio = data / modulus.clamp(min=torch.finfo(modulus.dtype).tiny)
    if rule == "reference":
        ratio = ratio.sqrt()
    residual = detector * (1.0 - ratio)
    grad = -1j * psi.conj() * torch.fft.ifft2(torch.fft.fft2(residual) * forward.conj())
    return grad, modulus


def project_refractive(obj: torch.Tensor) -> torch.Tensor:
    """``P_OmegaP``: ``Re(O) <= 0``, ``Im(O) >= 0`` ([2] Eq. (21) for data divided by ``a0``)."""
    return torch.complex(obj.real.clamp(max=0.0), obj.imag.clamp(min=0.0))


def asrm_reconstruct(
    hologram: torch.Tensor,
    fresnel_number: float,
    settings: ASRMSettings | None = None,
    init: torch.Tensor | None = None,
) -> ASRMResult:
    """Reconstruct ``O~`` from a flat-field corrected hologram with ASRM.

    Parameters
    ----------
    hologram:
        Intensity ``I`` of shape ``(H, W)`` (free space = ``a0**2``). Its real dtype sets
        the precision (``float32`` -> ``complex64``).
    fresnel_number:
        ``Fr`` of the assumed geometry on the grid of ``hologram``; stage ``j`` uses
        ``Fr * S_down**2``.
    settings:
        Schedule and options, default :class:`ASRMSettings` ([2] Table 3, padding 2).
    init:
        Optional start value for ``O~`` (any shape, resized to the first stage).
    """
    settings = ASRMSettings() if settings is None else settings
    if hologram.ndim != 2:
        raise ValueError(f"expected a single hologram (H, W), got shape {tuple(hologram.shape)}")
    start_time = time.perf_counter()
    real = real_dtype_of(hologram.dtype)
    cdtype = complex_dtype_of(real)
    device = hologram.device
    amplitude = hologram.to(real).clamp(min=0.0).sqrt() / settings.a0
    shape_in = tuple(hologram.shape)

    obj = None if init is None else init.to(device=device, dtype=cdtype)
    velocity = None
    total = settings.iterations
    losses = torch.zeros(total, dtype=real, device=device)
    stage_ends: list[int] = []
    k = 0
    grid = None
    for stage in settings.stages:
        grid = stage_grid(shape_in, stage.down_sampling, settings.padding_factor)
        data, window = prepare_stage_data(amplitude, grid, stage.down_sampling, settings.fade_width)
        if obj is None:
            obj = torch.zeros(grid.shape, dtype=cdtype, device=device)
        obj = resize_complex(obj, grid.shape)
        if velocity is None:
            velocity = torch.zeros(grid.shape, dtype=cdtype, device=device)
        velocity = resize_complex(velocity, grid.shape)

        forward = FresnelPropagator(
            grid.shape, fresnel_number * stage.down_sampling**2, device, cdtype
        ).kernel
        k_velocity = gaussian_transfer(grid.shape, stage.momentum_fwhm, device, real)
        k_phase = gaussian_transfer(grid.shape, stage.phase_fwhm, device, real, onesided=True)
        k_absorption = gaussian_transfer(
            grid.shape, stage.absorption_fwhm, device, real, onesided=True
        )
        fov = grid.fov
        data_fov = data[fov]
        gamma, eta, beta = stage.momentum, stage.step, stage.l2_weight

        for _ in range(stage.iterations):
            obj = pad_object(obj, grid, window, settings.pin_phase)
            velocity = _filter_complex(velocity, k_velocity)
            if k_phase is not None or k_absorption is not None:
                obj = torch.complex(
                    _filter_real(obj.real, k_phase), _filter_real(obj.imag, k_absorption)
                )
            lookahead = obj - gamma * velocity
            grad, modulus = data_gradient(lookahead, data, forward, settings.gradient)
            losses[k] = (modulus[fov] - data_fov).square().mean()

            velocity = gamma * velocity + eta * grad
            if beta != 0:
                imag = lookahead.imag
                velocity.imag.add_(imag * (beta / (imag.norm() + 1e-6)))
            base = lookahead if settings.update == "reference" else obj
            obj = project_refractive(base - velocity)
            k += 1
        stage_ends.append(k)

    assert grid is not None
    return ASRMResult(
        obj=obj[grid.fov].clone(),
        padded=obj,
        loss=losses.tolist(),
        stage_ends=stage_ends,
        seconds=time.perf_counter() - start_time,
    )
