"""Fresnel propagator in operator form, [1] Eq. (5) and [2] Eq. (3).

``D_Fr(psi) = F^-1[ exp(-i pi (kx^2 + ky^2) / Fr) * F[psi] ]`` with ``kx, ky`` in cycles per
pixel (``torch.fft.fftfreq(n)``), so that ``(kx^2 + ky^2) / Fr = lambda z (fx^2 + fy^2)`` for
physical frequencies ``fx = kx / dx``. A negative Fresnel number propagates backwards;
``D_{-Fr}`` is the exact inverse (and adjoint) of ``D_Fr`` because the kernel is unimodular.

The kernel is sampled without aliasing if the array size satisfies ``N >= 1 / |Fr|``
([1] Sec. 3.2.2); :func:`check_sampling` reports violations.
"""

from __future__ import annotations

import math
import warnings

import torch

from .device import complex_dtype_of, one_thread


def _kernel_factor(n: int, fresnel_number: float, device, dtype: torch.dtype) -> torch.Tensor:
    """1-D factor ``exp(-i pi k^2 / Fr)``; the phase is reduced mod 2 pi in float64 first."""
    k = torch.fft.fftfreq(n, dtype=torch.float64, device=device)
    phase = torch.remainder((-math.pi / fresnel_number) * k * k, 2.0 * math.pi)
    return torch.polar(torch.ones_like(phase), phase).to(dtype)


def check_sampling(shape: tuple[int, int], fresnel_number: float) -> bool:
    """True if every dimension satisfies ``N >= 1 / |Fr|``."""
    if math.isinf(fresnel_number):
        return True
    return min(shape) * abs(fresnel_number) >= 1.0


class FresnelPropagator:
    """Pre-computed Fresnel propagator ``D_Fr`` for a fixed array shape.

    The 2-D kernel is separable, ``exp(-i pi ky^2/Fr) exp(-i pi kx^2/Fr)``, so only the two
    1-D factors are stored. Fields may carry leading batch dimensions ``(..., H, W)``. The
    complex products run on one thread (:func:`~holofocus.device.one_thread`), so kernel and
    propagated field have the same bits for every thread count.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        fresnel_number: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.complex64,
        warn_undersampled: bool = True,
    ) -> None:
        if fresnel_number == 0:
            raise ValueError("Fresnel number must be non-zero (use inf for no propagation)")
        self.shape = (int(shape[-2]), int(shape[-1]))
        self.fresnel_number = float(fresnel_number)
        self.dtype = complex_dtype_of(dtype)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        if warn_undersampled and not check_sampling(self.shape, self.fresnel_number):
            warnings.warn(
                f"array {self.shape} undersamples the Fresnel kernel: need N >= 1/|Fr| = "
                f"{1 / abs(self.fresnel_number):.0f} ([1] Sec. 3.2.2)",
                stacklevel=2,
            )
        if math.isinf(self.fresnel_number):
            self._ky = self._kx = None
        else:
            self._ky = _kernel_factor(self.shape[0], self.fresnel_number, self.device, self.dtype)
            self._kx = _kernel_factor(self.shape[1], self.fresnel_number, self.device, self.dtype)
            self._ky = self._ky[:, None]
            self._kx = self._kx[None, :]

    @property
    def kernel(self) -> torch.Tensor:
        """The full 2-D transfer function (materialised on demand)."""
        if self._ky is None:
            return torch.ones(self.shape, dtype=self.dtype, device=self.device)
        with one_thread():
            return self._ky * self._kx

    def __call__(self, field: torch.Tensor) -> torch.Tensor:
        if tuple(field.shape[-2:]) != self.shape:
            raise ValueError(f"field shape {tuple(field.shape[-2:])} != propagator {self.shape}")
        field = field.to(device=self.device, dtype=self.dtype)
        if self._ky is None:
            return field.clone()
        spectrum = torch.fft.fft2(field)
        with one_thread():
            spectrum = spectrum * self._ky * self._kx
        return torch.fft.ifft2(spectrum)

    def inverse(self) -> FresnelPropagator:
        """``D_{-Fr}``, the inverse and adjoint of this propagator."""
        return FresnelPropagator(
            self.shape, -self.fresnel_number, self.device, self.dtype, warn_undersampled=False
        )


def propagate(field: torch.Tensor, fresnel_number: float, **kwargs) -> torch.Tensor:
    """Propagate ``field`` (complex, ``(..., H, W)``) with Fresnel number ``fresnel_number``."""
    if not field.is_complex():
        field = field.to(complex_dtype_of(field.dtype))
    propagator = FresnelPropagator(
        tuple(field.shape[-2:]), fresnel_number, device=field.device, dtype=field.dtype, **kwargs
    )
    return propagator(field)


def next_fast_len(n: int) -> int:
    """Smallest integer ``>= n`` whose prime factors are 2, 3, 5 or 7 (fast FFT sizes)."""
    m = max(int(math.ceil(n)), 1)
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def compose_fresnel_numbers(*fresnel_numbers: float) -> float:
    """Fresnel number of consecutive propagations on one grid: ``1/Fr = sum(1/Fr_i)``."""
    inverse = sum(0.0 if math.isinf(fr) else 1.0 / fr for fr in fresnel_numbers)
    return math.inf if inverse == 0 else 1.0 / inverse
