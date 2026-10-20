"""Physical constraint sets ("positive electron density").

* Refractive object: [1] Eq. (18)-(19) and [2] Eq. (21)
  ``Re(O~) in [-inf, 0]``, ``Im(O~) in [-log(A0), inf]``.
* Wave field: [2] Eq. (12) ``arg(psi) in [-inf, 0]``, ``|psi| in [0, 1]`` (valid only for the
  in-focus exit wave; this is what the model fit error of [2] exploits).
"""

from __future__ import annotations

import math

import torch


def project_object(obj: torch.Tensor, a0: float = 1.0) -> torch.Tensor:
    """``P_OmegaP(O~) = min(0, Re O~) + i max(-log A0, Im O~)``, [1] Eq. (19)."""
    return torch.complex(obj.real.clamp(max=0.0), obj.imag.clamp(min=-math.log(a0)))


def project_wavefield(psi: torch.Tensor) -> torch.Tensor:
    """Projection of a wave field onto [2] Eq. (12): phase clipped to <= 0, modulus to <= 1."""
    amplitude = psi.abs().clamp(max=1.0)
    phase = torch.angle(psi).clamp(max=0.0)
    return torch.polar(amplitude, phase)
