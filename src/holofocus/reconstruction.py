"""Minimal reconstructions used to visualise de-focus (Stage 1).

* :func:`backpropagate`: holographic back-propagation ``D_{-Fr}(sqrt(I))`` (twin image).
* :func:`pgd_reconstruct`: projected gradient descent on ``O~`` with optional Nesterov
  momentum, i.e. the core of refAP ([1] Alg. 2, Eq. (19)-(23)) without the Gaussian object
  filter. It is *not* ASRM ([1] Alg. 5); the baseline of [2] is planned for Stage 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .constraints import project_object
from .device import complex_dtype_of
from .forward import exit_wave
from .preprocessing import preprocess_hologram
from .propagation import FresnelPropagator


def backpropagate(
    hologram: torch.Tensor,
    fresnel_number: float,
    padded_shape: tuple[int, int] | None = None,
    a0: float = 1.0,
    mirror: bool = True,
) -> torch.Tensor:
    """Back-propagate the measured amplitude with zero phase to the sample plane."""
    if padded_shape is None:
        amplitude = hologram.clamp(min=0).sqrt()
        propagator = FresnelPropagator(
            tuple(hologram.shape[-2:]),
            -fresnel_number,
            hologram.device,
            complex_dtype_of(hologram.dtype),
        )
        return propagator(amplitude)
    pre = preprocess_hologram(hologram, padded_shape, a0=a0, mirror=mirror)
    propagator = FresnelPropagator(
        tuple(padded_shape), -fresnel_number, hologram.device, complex_dtype_of(hologram.dtype)
    )
    return pre.crop(propagator(pre.data.clamp(min=0).sqrt()))


def data_fidelity_gradient(
    obj: torch.Tensor, sqrt_hologram: torch.Tensor, propagator: FresnelPropagator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss ``1/2 || |D(O~)| - sqrt(I) ||^2`` and its gradient, [1] Eq. (21)-(23).

    ``grad = -i conj(exp(i O~)) D^-1( D(O~) - sqrt(I) sgn(D(O~)) )`` with ``sgn(0) = 0``.
    The conjugate is printed in the PDF of [1] (it is lost in the text layer).
    """
    psi = exit_wave(obj)
    detector = propagator(psi)
    amplitude = detector.abs()
    sgn = torch.where(amplitude > 0, detector / amplitude.clamp(min=1e-30), 0)
    residual = detector - sqrt_hologram * sgn
    grad = -1j * psi.conj() * propagator.inverse()(residual)
    loss = 0.5 * (amplitude - sqrt_hologram).pow(2).sum()
    return grad, loss


@dataclass
class PGDResult:
    obj: torch.Tensor
    loss: list[float] = field(default_factory=list)


def pgd_reconstruct(
    hologram: torch.Tensor,
    fresnel_number: float,
    iterations: int = 200,
    step: float = 1.0,
    momentum: float = 0.0,
    padded_shape: tuple[int, int] | None = None,
    a0: float = 1.0,
    mirror: bool = True,
    init: torch.Tensor | None = None,
) -> PGDResult:
    """Projected (Nesterov) gradient descent for ``O~`` under ``Omega_P`` ([1] Eq. (18)-(20)).

    The hologram is preprocessed with [1] Alg. 1 when ``padded_shape`` is given; the result
    is cropped to the detector field of view. The loss is reported per pixel of the padded
    array (it measures convergence, not the model fit error of [2]).
    """
    dtype = complex_dtype_of(hologram.dtype)
    if padded_shape is None:
        target = hologram
        crop = None
    else:
        pre = preprocess_hologram(hologram, padded_shape, a0=a0, mirror=mirror)
        target, crop = pre.data, pre.crop
    sqrt_target = target.clamp(min=0).sqrt()
    propagator = FresnelPropagator(tuple(target.shape[-2:]), fresnel_number, target.device, dtype)
    obj = torch.zeros_like(target, dtype=dtype) if init is None else init.to(dtype).clone()
    velocity = torch.zeros_like(obj)
    losses: list[float] = []
    n_pixels = target.shape[-1] * target.shape[-2]
    for _ in range(iterations):
        lookahead = obj - momentum * velocity
        grad, loss = data_fidelity_gradient(lookahead, sqrt_target, propagator)
        velocity = momentum * velocity + step * grad
        obj = project_object(obj - velocity, a0=a0)
        losses.append(float(loss) / n_pixels)
    return PGDResult(obj=obj if crop is None else crop(obj), loss=losses)
