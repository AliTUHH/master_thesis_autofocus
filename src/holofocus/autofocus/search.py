"""Nested autofocus of [2] Eq. (22): Nelder-Mead over ``z01`` with ASRM and MFE inside.

``z01* = argmin_{z01_hat in Omega_z} G(argmin_{O in Omega_P} L(O, z01_hat))``: every probing
point ``z01_hat`` is a full ASRM reconstruction (:func:`holofocus.asrm.asrm_reconstruct`) and
``G`` is its final model fit error. As in the reference ``find_focus``:

* ``Omega_z = [z01_guess - confidence, z01_guess + confidence]`` ([2] Eq. (17), 5 mm),
* the initial simplex is the pair of interval ends (10 mm apart); a warm start (Stage 4) passes
  a shorter pair inside ``Omega_z`` instead, e.g. a learned estimate +- a half width,
* trial points are clipped to ``Omega_z`` (projection); ``bounds`` replaces ``Omega_z`` by
  another interval, e.g. a learned 95 % interval (Stage 8),
* a point that was evaluated before returns ``sys.float_info.max`` instead of a second
  reconstruction (the reference's guard against reflections that are clipped back onto a
  bound), so the optimizer contracts instead. The reference compares bit for bit; a simplex
  that does not start on the interval ends can reach a point twice on paths that differ in
  the last bit, which ``revisit_tolerance`` catches,
* the search stops when the simplex is shorter than ``xatol = 100 um``; ``fatol = 1e6``
  effectively disables the tolerance on the function values,
* the simplex lives in millimetres. Revisited points are then bit-identical to the first
  visit for intervals like 95-105 mm, whereas in metres they can differ in the last bit and
  would be reconstructed a second time.

``method="brent"`` replaces Nelder-Mead by Brent's bounded method
(``scipy.optimize.minimize_scalar(method="bounded")``, Stage 9) in the same interval and
units: golden-section steps and parabolic interpolation through the MFE values, starting
inside the interval (at 0.382 of its width) instead of at its ends. It stops when the
minimum is bracketed to about ``2/3 xatol`` around the best point; ``fatol`` and
``initial_simplex`` do not apply. A revisited point returns its earlier MFE, which Brent
fits, instead of the penalty.

:func:`search_points` gives the points the same search probes on a given MFE curve, without
reconstructing (Stage 8).
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from ..asrm import ASRMResult, ASRMSettings, asrm_reconstruct
from ..geometry import ConeBeamGeometry
from .nelder_mead import nelder_mead

PENALTY = sys.float_info.max
MM_PER_M = 1e3
METHODS = ("nelder-mead", "brent")


@dataclass
class FocusEvaluation:
    """One probing point: ``reconstructed=False`` marks a repeated point (penalty value)."""

    z01: float
    mfe: float
    seconds: float
    reconstructed: bool


@dataclass
class FocusSearchResult:
    """Estimate ``z01`` with the probing history of the search.

    ``reconstruction`` is the ASRM result at the estimate (the probing point with the lowest
    MFE), so the focused object needs no extra reconstruction. ``nm_iterations`` counts the
    iterations of the method (Nelder-Mead or Brent); Brent has no ``initial_simplex``.
    """

    z01: float
    z01_guess: float
    bounds: tuple[float, float]
    mfe: float
    nm_iterations: int
    converged: bool
    seconds: float
    evaluations: list[FocusEvaluation] = field(default_factory=list)
    reconstruction: ASRMResult | None = field(default=None, repr=False)
    initial_simplex: tuple[float, float] | None = None
    method: str = "nelder-mead"

    @property
    def n_reconstructions(self) -> int:
        return sum(e.reconstructed for e in self.evaluations)

    @property
    def n_evaluations(self) -> int:
        return len(self.evaluations)

    def to_dict(self) -> dict:
        """Numbers and probing history (without the reconstruction), JSON-serializable.
        ``method`` only for Brent, so Nelder-Mead records keep their keys."""
        record = {
            "z01": self.z01,
            "z01_guess": self.z01_guess,
            "bounds": list(self.bounds),
            "initial_simplex": None if self.initial_simplex is None else list(self.initial_simplex),
            "mfe": self.mfe,
            "nm_iterations": self.nm_iterations,
            "converged": self.converged,
            "seconds": self.seconds,
            "evaluations": [asdict(e) for e in self.evaluations],
            "n_reconstructions": self.n_reconstructions,
            "n_evaluations": self.n_evaluations,
        }
        if self.method != "nelder-mead":
            record["method"] = self.method
        return record


def find_focus(
    hologram: torch.Tensor,
    geometry: ConeBeamGeometry,
    settings: ASRMSettings | None = None,
    confidence: float = 5e-3,
    xatol: float = 1e-4,
    fatol: float = 1e6,
    max_evaluations: int | None = None,
    initial_simplex: tuple[float, float] | None = None,
    bounds: tuple[float, float] | None = None,
    revisit_tolerance: float = 0.0,
    log: Callable[[str], None] | None = None,
    method: str = "nelder-mead",
) -> FocusSearchResult:
    """Estimate ``z01`` of ``hologram`` starting from the guess ``geometry.z01``.

    ``geometry`` describes the (binned) detector the hologram was recorded with; only its
    ``z01`` is varied (``z02`` is known, [2] Sec. 2). The search interval is
    ``Omega_z = guess +- confidence`` unless ``bounds`` gives another ``(lower, upper)``.
    ``initial_simplex`` replaces the interval ends as the two starting points; both must lie
    in the search interval, which still bounds the search. A probing point within
    ``revisit_tolerance`` of an earlier one counts as repeated (0: bit-identical only, as in
    the reference). ``method`` is ``"nelder-mead"`` ([2]) or ``"brent"``. Lengths in metres.
    """
    settings = ASRMSettings() if settings is None else settings
    z01_guess = geometry.z01
    (lower, upper), interval_mm = _interval(z01_guess, confidence, bounds)
    if not 0 < lower < upper < geometry.z02:
        raise ValueError(
            f"search interval [{lower}, {upper}] m must lie between focus and detector"
        )
    best: list[ASRMResult] = []

    def reconstruct(z01: float) -> tuple[float, float]:
        start = time.perf_counter()
        result = asrm_reconstruct(hologram, geometry.with_z01(z01).fresnel_number, settings)
        seconds = time.perf_counter() - start
        if not best or result.final_loss < best[0].final_loss:
            best[:] = [result]
        if log is not None:
            log(f"z01 = {z01 * 1e3:.4f} mm  MFE = {result.final_loss:.6e}  ({seconds:.1f} s)")
        return result.final_loss, seconds

    start = time.perf_counter()
    outcome = _search(
        reconstruct,
        interval_mm,
        initial_simplex,
        xatol,
        fatol,
        max_evaluations,
        revisit_tolerance,
        method,
    )
    return FocusSearchResult(
        z01=outcome.x_mm / MM_PER_M,
        z01_guess=z01_guess,
        bounds=(lower, upper),
        mfe=outcome.fun,
        nm_iterations=outcome.iterations,
        converged=outcome.converged,
        seconds=time.perf_counter() - start,
        evaluations=outcome.evaluations,
        reconstruction=best[0] if best else None,
        initial_simplex=None
        if outcome.start_mm is None
        else (outcome.start_mm[0] / MM_PER_M, outcome.start_mm[1] / MM_PER_M),
        method=method,
    )


def search_points(
    mfe: Callable[[float], float],
    z01_guess: float,
    confidence: float = 5e-3,
    xatol: float = 1e-4,
    fatol: float = 1e6,
    max_evaluations: int | None = None,
    initial_simplex: tuple[float, float] | None = None,
    bounds: tuple[float, float] | None = None,
    revisit_tolerance: float = 0.0,
    method: str = "nelder-mead",
) -> list[FocusEvaluation]:
    """The points :func:`find_focus` probes from the guess ``z01_guess`` if the MFE at ``z01``
    were ``mfe(z01)``, in order, without reconstructing (``seconds`` 0).

    Nelder-Mead only compares MFE values, so every curve that is symmetric about its minimum
    and rises on both sides gives the points of a parabola about that minimum; Brent's points
    depend on the values. The other arguments as in :func:`find_focus`; lengths in metres.
    """
    _, interval_mm = _interval(z01_guess, confidence, bounds)
    outcome = _search(
        lambda z01: (float(mfe(z01)), 0.0),
        interval_mm,
        initial_simplex,
        xatol,
        fatol,
        max_evaluations,
        revisit_tolerance,
        method,
    )
    return outcome.evaluations


def _interval(
    z01_guess: float, confidence: float, bounds: tuple[float, float] | None
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The search interval in metres and, as the simplex sees it, in millimetres. Converted
    from metres, ``Omega_z`` can differ in the last bit and revisits go unnoticed."""
    if bounds is None:
        guess_mm, confidence_mm = z01_guess * MM_PER_M, confidence * MM_PER_M
        return (
            (z01_guess - confidence, z01_guess + confidence),
            (guess_mm - confidence_mm, guess_mm + confidence_mm),
        )
    lower, upper = (float(b) for b in bounds)
    return (lower, upper), (lower * MM_PER_M, upper * MM_PER_M)


@dataclass
class _Outcome:
    """Best point and value in millimetres, iterations, convergence, probing history and the
    starting simplex (``None`` for Brent)."""

    x_mm: float
    fun: float
    iterations: int
    converged: bool
    evaluations: list[FocusEvaluation]
    start_mm: tuple[float, float] | None


def _search(
    evaluate: Callable[[float], tuple[float, float]],
    interval_mm: tuple[float, float],
    initial_simplex: tuple[float, float] | None,
    xatol: float,
    fatol: float,
    max_evaluations: int | None,
    revisit_tolerance: float,
    method: str = "nelder-mead",
) -> _Outcome:
    """Nelder-Mead or Brent over ``z01`` in ``interval_mm``; ``evaluate(z01)`` gives
    ``(mfe, seconds)`` of a new point."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if revisit_tolerance < 0:
        raise ValueError("revisit_tolerance must not be negative")
    if method == "brent":
        if initial_simplex is not None:
            raise ValueError("Brent's method starts inside the interval, not from a simplex")
        return _brent(evaluate, interval_mm, xatol, max_evaluations, revisit_tolerance)
    lower_mm, upper_mm = interval_mm
    tolerance_mm = revisit_tolerance * MM_PER_M
    if initial_simplex is None:
        start_mm = (lower_mm, upper_mm)
    else:
        given = np.asarray(initial_simplex, dtype=float).ravel() * MM_PER_M
        clipped = np.clip(given, lower_mm, upper_mm)
        start_mm = tuple(float(z) for z in clipped)
        if (
            len(start_mm) != 2
            or not start_mm[0] < start_mm[1]
            or np.abs(given - clipped).max() > 1e-6
        ):
            raise ValueError(
                f"initial simplex {initial_simplex} m must be an increasing pair in the search "
                "interval"
            )
    evaluations: list[FocusEvaluation] = []
    seen: list[float] = []

    def objective(x: np.ndarray) -> float:
        z01_mm = float(x[0])
        z01 = z01_mm / MM_PER_M
        if any(abs(z01_mm - s) <= tolerance_mm for s in seen):
            evaluations.append(FocusEvaluation(z01, PENALTY, 0.0, False))
            return PENALTY
        seen.append(z01_mm)
        mfe, seconds = evaluate(z01)
        evaluations.append(FocusEvaluation(z01, mfe, seconds, True))
        return mfe

    nm = nelder_mead(
        objective,
        initial_simplex=[[start_mm[0]], [start_mm[1]]],
        bounds=[(lower_mm, upper_mm)],
        xatol=xatol * MM_PER_M,
        fatol=fatol,
        max_evaluations=max_evaluations,
    )
    return _Outcome(float(nm.x[0]), nm.fun, nm.iterations, nm.converged, evaluations, start_mm)


def _brent(
    evaluate: Callable[[float], tuple[float, float]],
    interval_mm: tuple[float, float],
    xatol: float,
    max_evaluations: int | None,
    revisit_tolerance: float,
) -> _Outcome:
    tolerance_mm = revisit_tolerance * MM_PER_M
    evaluations: list[FocusEvaluation] = []
    seen: list[tuple[float, float]] = []

    def objective(z01_mm: float) -> float:
        z01_mm = float(z01_mm)
        z01 = z01_mm / MM_PER_M
        for s, value in seen:
            if abs(z01_mm - s) <= tolerance_mm:
                evaluations.append(FocusEvaluation(z01, value, 0.0, False))
                return value
        mfe, seconds = evaluate(z01)
        seen.append((z01_mm, mfe))
        evaluations.append(FocusEvaluation(z01, mfe, seconds, True))
        return mfe

    options = {"xatol": xatol * MM_PER_M}
    if max_evaluations is not None:
        options["maxiter"] = max_evaluations
    brent = minimize_scalar(objective, bounds=interval_mm, method="bounded", options=options)
    return _Outcome(
        float(brent.x), float(brent.fun), int(brent.nit), brent.status == 0, evaluations, None
    )
