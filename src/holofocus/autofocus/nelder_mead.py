"""Nelder-Mead downhill simplex with projection onto box bounds ([2] Sec. 3.4, Fig. 4).

Port of ``scipy.optimize.minimize(method="Nelder-Mead", bounds=..., initial_simplex=...)``
(scipy 1.17, non-adaptive coefficients, BSD license), which the reference autofocus of [2]
calls: every trial point is clipped to the bounds (projection onto ``Omega_z``, [2] Eq. (17))
and the search stops when all vertices are within ``xatol`` of the best one and all function
values within ``fatol``. The port evaluates ``func`` at the same points in the same order as
scipy, so the number of reconstructions of the nested autofocus can be compared directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

REFLECTION, EXPANSION, CONTRACTION, SHRINK = 1.0, 2.0, 0.5, 0.5


@dataclass
class NelderMeadResult:
    """Best vertex, its value and the trace of all evaluations (``points[i]``, ``values[i]``)."""

    x: np.ndarray
    fun: float
    iterations: int
    evaluations: int
    converged: bool
    points: list[np.ndarray] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    simplex: np.ndarray | None = None


def nelder_mead(
    func: Callable[[np.ndarray], float],
    initial_simplex: Sequence[Sequence[float]],
    bounds: Sequence[tuple[float, float]] | None = None,
    xatol: float = 1e-4,
    fatol: float = 1e-4,
    max_iterations: int | None = None,
    max_evaluations: int | None = None,
) -> NelderMeadResult:
    """Minimize ``func`` starting from ``initial_simplex`` (shape ``(N + 1, N)``).

    ``bounds`` is a sequence of ``(low, high)`` per dimension; trial points are clipped to it.
    The limits default to ``200 N`` iterations and evaluations as in scipy.
    """
    sim = np.atleast_2d(np.asarray(initial_simplex, dtype=float)).copy()
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
        raise ValueError("initial_simplex must have shape (N + 1, N)")
    n = sim.shape[1]
    if bounds is not None:
        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)
        if lower.shape != (n,) or np.any(lower > upper):
            raise ValueError("bounds must give one increasing (low, high) pair per dimension")
        sim = np.clip(np.where(sim > upper, 2 * upper - sim, sim), lower, upper)
    max_iterations = 200 * n if max_iterations is None else max_iterations
    max_evaluations = 200 * n if max_evaluations is None else max_evaluations

    points: list[np.ndarray] = []
    values: list[float] = []

    def clip(x: np.ndarray) -> np.ndarray:
        return x if bounds is None else np.clip(x, lower, upper)

    def evaluate(x: np.ndarray) -> float:
        if len(values) >= max_evaluations:
            raise _BudgetExhausted
        value = float(func(x.copy()))
        points.append(x.copy())
        values.append(value)
        return value

    fsim = np.full(n + 1, np.inf)
    try:
        for k in range(n + 1):
            fsim[k] = evaluate(sim[k])
    except _BudgetExhausted:
        pass
    order = np.argsort(fsim)
    sim, fsim = sim[order], fsim[order]

    iterations = 1
    converged = False
    while len(values) < max_evaluations and iterations < max_iterations:
        try:
            if (
                np.max(np.abs(sim[1:] - sim[0])) <= xatol
                and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol
            ):
                converged = True
                break
            centroid = np.add.reduce(sim[:-1], 0) / n
            xr = clip((1 + REFLECTION) * centroid - REFLECTION * sim[-1])
            fxr = evaluate(xr)
            if fxr < fsim[0]:
                xe = clip(
                    (1 + REFLECTION * EXPANSION) * centroid - REFLECTION * EXPANSION * sim[-1]
                )
                fxe = evaluate(xe)
                if fxe < fxr:
                    sim[-1], fsim[-1] = xe, fxe
                else:
                    sim[-1], fsim[-1] = xr, fxr
            elif fxr < fsim[-2]:
                sim[-1], fsim[-1] = xr, fxr
            else:
                shrink = False
                if fxr < fsim[-1]:
                    xc = clip(
                        (1 + CONTRACTION * REFLECTION) * centroid
                        - CONTRACTION * REFLECTION * sim[-1]
                    )
                    fxc = evaluate(xc)
                    if fxc <= fxr:
                        sim[-1], fsim[-1] = xc, fxc
                    else:
                        shrink = True
                else:
                    xcc = clip((1 - CONTRACTION) * centroid + CONTRACTION * sim[-1])
                    fxcc = evaluate(xcc)
                    if fxcc < fsim[-1]:
                        sim[-1], fsim[-1] = xcc, fxcc
                    else:
                        shrink = True
                if shrink:
                    for j in range(1, n + 1):
                        sim[j] = clip(sim[0] + SHRINK * (sim[j] - sim[0]))
                        fsim[j] = evaluate(sim[j])
            iterations += 1
        except _BudgetExhausted:
            pass
        order = np.argsort(fsim)
        sim, fsim = sim[order], fsim[order]

    return NelderMeadResult(
        x=sim[0].copy(),
        fun=float(np.min(fsim)),
        iterations=iterations,
        evaluations=len(values),
        converged=converged,
        points=points,
        values=values,
        simplex=sim.copy(),
    )


class _BudgetExhausted(Exception):
    pass
