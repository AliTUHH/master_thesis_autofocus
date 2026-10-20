"""Nested autofocus of [2] Eq. (22): Nelder-Mead over ``z01`` with ASRM and MFE inside.

``z01* = argmin_{z01_hat in Omega_z} G(argmin_{O in Omega_P} L(O, z01_hat))``: every probing
point ``z01_hat`` is a full ASRM reconstruction (:func:`holofocus.asrm.asrm_reconstruct`) and
``G`` is its final model fit error. As in the reference ``find_focus``:

* ``Omega_z = [z01_guess - confidence, z01_guess + confidence]`` ([2] Eq. (17), 5 mm),
* the initial simplex is the pair of interval ends (10 mm apart),
* trial points are clipped to ``Omega_z`` (projection),
* a point that was evaluated before returns ``sys.float_info.max`` instead of a second
  reconstruction (the reference's guard against reflections that are clipped back onto a
  bound), so the optimizer contracts instead,
* the search stops when the simplex is shorter than ``xatol = 100 um``; ``fatol = 1e6``
  effectively disables the tolerance on the function values.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from ..asrm import ASRMSettings, asrm_reconstruct
from ..geometry import ConeBeamGeometry
from .nelder_mead import nelder_mead

PENALTY = sys.float_info.max


@dataclass
class FocusEvaluation:
    """One probing point: ``reconstructed=False`` marks a repeated point (penalty value)."""

    z01: float
    mfe: float
    seconds: float
    reconstructed: bool


@dataclass
class FocusSearchResult:
    """Estimate ``z01`` with the probing history of the Nelder-Mead search."""

    z01: float
    z01_guess: float
    bounds: tuple[float, float]
    mfe: float
    nm_iterations: int
    converged: bool
    seconds: float
    evaluations: list[FocusEvaluation] = field(default_factory=list)

    @property
    def n_reconstructions(self) -> int:
        return sum(e.reconstructed for e in self.evaluations)

    @property
    def n_evaluations(self) -> int:
        return len(self.evaluations)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["bounds"] = list(self.bounds)
        out["n_reconstructions"] = self.n_reconstructions
        out["n_evaluations"] = self.n_evaluations
        return out


def find_focus(
    hologram: torch.Tensor,
    geometry: ConeBeamGeometry,
    settings: ASRMSettings | None = None,
    confidence: float = 5e-3,
    xatol: float = 1e-4,
    fatol: float = 1e6,
    max_evaluations: int | None = None,
    log: Callable[[str], None] | None = None,
) -> FocusSearchResult:
    """Estimate ``z01`` of ``hologram`` starting from the guess ``geometry.z01``.

    ``geometry`` describes the (binned) detector the hologram was recorded with; only its
    ``z01`` is varied (``z02`` is known, [2] Sec. 2). Lengths in metres.
    """
    settings = ASRMSettings() if settings is None else settings
    z01_guess = geometry.z01
    lower, upper = z01_guess - confidence, z01_guess + confidence
    if not 0 < lower < upper < geometry.z02:
        raise ValueError(f"Omega_z = [{lower}, {upper}] m must lie between focus and detector")
    evaluations: list[FocusEvaluation] = []
    seen: set[float] = set()

    def objective(x: np.ndarray) -> float:
        z01 = float(x[0])
        if z01 in seen:
            evaluations.append(FocusEvaluation(z01, PENALTY, 0.0, False))
            return PENALTY
        seen.add(z01)
        start = time.perf_counter()
        result = asrm_reconstruct(hologram, geometry.with_z01(z01).fresnel_number, settings)
        seconds = time.perf_counter() - start
        evaluations.append(FocusEvaluation(z01, result.final_loss, seconds, True))
        if log is not None:
            log(f"z01 = {z01 * 1e3:.4f} mm  MFE = {result.final_loss:.6e}  ({seconds:.1f} s)")
        return result.final_loss

    start = time.perf_counter()
    nm = nelder_mead(
        objective,
        initial_simplex=[[lower], [upper]],
        bounds=[(lower, upper)],
        xatol=xatol,
        fatol=fatol,
        max_evaluations=max_evaluations,
    )
    return FocusSearchResult(
        z01=float(nm.x[0]),
        z01_guess=z01_guess,
        bounds=(lower, upper),
        mfe=nm.fun,
        nm_iterations=nm.iterations,
        converged=nm.converged,
        seconds=time.perf_counter() - start,
        evaluations=evaluations,
    )
