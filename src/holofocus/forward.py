"""Refractive forward model of near-field holography, [1] Sec. 3.1 and [2] Sec. 2.

Projected refractive object ([1] Eq. (3), [2] Eq. (1))::

    O~ = -k * integral(delta - i beta) dz = phi + i mu,      k = 2 pi / lambda

with phase ``phi <= 0`` and absorption ``mu >= 0``. Exit wave for a constant illumination
``A0 exp(i phi0)`` ([1] Eq. (4)): ``psi_exit = exp(i O~) A0 exp(i phi0)``. The hologram is
``I_det = |D_Fr(psi_exit)|^2`` ([1] Eq. (5)-(6)).
"""

from __future__ import annotations

import math

import torch

from .arrays import crop_center, pad_center
from .device import complex_dtype_of
from .propagation import FresnelPropagator


def refractive_object(
    delta_projection: torch.Tensor, beta_projection: torch.Tensor, wavelength: float
) -> torch.Tensor:
    """``O~`` from the projections ``int delta dz`` and ``int beta dz`` (in metres), Eq. (3)."""
    k = 2.0 * math.pi / wavelength
    return torch.complex(-k * delta_projection, k * beta_projection)


def single_material_object(thickness: torch.Tensor, delta: float, beta: float, wavelength: float):
    """``O~`` of a homogeneous material with projected thickness ``thickness`` (metres)."""
    return refractive_object(delta * thickness, beta * thickness, wavelength)


def object_from_phase(phase: torch.Tensor, delta_beta_ratio: float | None) -> torch.Tensor:
    """``O~ = phi + i mu`` for a single material with fixed ``delta / beta`` ([2] Sec. 3.5).

    ``mu = -phi / (delta / beta)``; ``delta_beta_ratio=None`` gives a pure phase object.
    """
    absorption = torch.zeros_like(phase) if delta_beta_ratio is None else -phase / delta_beta_ratio
    return torch.complex(phase, absorption)


def exit_wave(obj: torch.Tensor, a0: float = 1.0, phi0: float = 0.0) -> torch.Tensor:
    """``psi_exit = exp(i O~) A0 exp(i phi0)``, [1] Eq. (4) (constant probe model)."""
    if not obj.is_complex():
        obj = obj.to(complex_dtype_of(obj.dtype))
    psi = torch.exp(1j * obj)
    if a0 != 1.0 or phi0 != 0.0:
        psi = psi * (a0 * complex(math.cos(phi0), math.sin(phi0)))
    return psi


def simulate_hologram(
    obj: torch.Tensor,
    fresnel_number: float,
    padded_shape: tuple[int, int] | None = None,
    a0: float = 1.0,
) -> torch.Tensor:
    """Noise-free hologram ``|D_Fr(exp(i O~))|^2`` on the grid of ``obj``.

    With ``padded_shape`` the object is embedded in free space (``O~ = 0``) of that size
    before propagation and the intensity is cropped back to ``obj.shape``. This removes the
    periodic wrap-around of the FFT for objects inside the field of view; it is the physical
    counterpart of the data padding in [1] Sec. 4.1, which is needed only for measured data.
    """
    shape = tuple(obj.shape[-2:])
    work = obj if padded_shape is None else pad_center(obj, tuple(padded_shape), 0.0)
    if not work.is_complex():
        work = work.to(complex_dtype_of(work.dtype))
    propagator = FresnelPropagator(
        tuple(work.shape[-2:]), fresnel_number, device=work.device, dtype=work.dtype
    )
    field = propagator(exit_wave(work, a0=a0))
    intensity = field.real**2 + field.imag**2
    return intensity if padded_shape is None else crop_center(intensity, shape)


def intensity(field: torch.Tensor) -> torch.Tensor:
    """``|psi|^2`` without forming the complex modulus first."""
    return field.real**2 + field.imag**2
