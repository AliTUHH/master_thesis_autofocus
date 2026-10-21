"""Error and uncertainty metrics shared by the learned model and the baseline (Stage 4).

Errors are ``estimate - truth`` in micrometres of ``z01``. Interval coverage uses the
central Gaussian interval ``mean +- z_p sigma`` with ``z_p = Phi^-1((1 + p) / 2)``
(``0.994`` for 68 %, ``1.960`` for 95 %).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from statistics import NormalDist

import numpy as np

THRESHOLDS_UM = (50, 100, 250, 1000)
LEVELS = (0.68, 0.95)
PHOTON_EDGES = (1e3, 3e3, 1e4, 3e4, 1e5)


def point_metrics(errors_um: Sequence[float], thresholds_um=THRESHOLDS_UM) -> dict:
    """Bias, spread and quantiles of the signed errors; fraction within each threshold."""
    err = np.asarray(errors_um, dtype=float)
    if err.size == 0:
        return {"n": 0}
    abs_err = np.abs(err)
    return {
        "n": int(err.size),
        "bias": float(err.mean()),
        "std": float(err.std(ddof=1)) if err.size > 1 else 0.0,
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "median_abs": float(np.median(abs_err)),
        "p90_abs": float(np.percentile(abs_err, 90)),
        "max_abs": float(abs_err.max()),
        "within": {str(int(t)): float(np.mean(abs_err <= t)) for t in thresholds_um},
    }


def interval_quantile(level: float) -> float:
    return NormalDist().inv_cdf(0.5 * (1.0 + level))


def interval_metrics(errors_um: Sequence[float], sigmas_um: Sequence[float], levels=LEVELS) -> dict:
    """Coverage of the central intervals, sharpness and the Gaussian NLL (per sample, in
    nats with ``z01`` in micrometres)."""
    err = np.asarray(errors_um, dtype=float)
    sigma = np.asarray(sigmas_um, dtype=float)
    if err.size == 0:
        return {"n": 0}
    z = err / sigma
    nll = 0.5 * np.log(2 * math.pi * sigma**2) + 0.5 * z**2
    return {
        "n": int(err.size),
        "coverage": {
            f"{round(100 * p)}": float(np.mean(np.abs(z) <= interval_quantile(p))) for p in levels
        },
        "sigma_mean": float(sigma.mean()),
        "sigma_median": float(np.median(sigma)),
        "z_rms": float(np.sqrt(np.mean(z**2))),
        "nll": float(nll.mean()),
    }


def calibration_scale(errors: Sequence[float], sigmas: Sequence[float]) -> float:
    """Factor ``s`` for ``sigma -> s sigma`` that minimizes the Gaussian NLL: ``rms(e / sigma)``."""
    z = np.asarray(errors, dtype=float) / np.asarray(sigmas, dtype=float)
    return float(np.sqrt(np.mean(z**2)))


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    level: float = 0.95,
    resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval of ``statistic`` (resampling the values with replacement).

    ``values`` may be 2-D ``(n, k)``; rows are resampled together (paired statistics).
    """
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(data), size=(resamples, len(data)))
    stats = np.array([statistic(data[p]) for p in picks])
    tail = 50.0 * (1.0 - level)
    return float(np.percentile(stats, tail)), float(np.percentile(stats, 100.0 - tail))


def photon_bin_names(edges=PHOTON_EDGES) -> list[str]:
    """``["1e3-3e3", "3e3-1e4", ...]`` for consecutive edges."""
    return [
        f"{a:.0e}-{b:.0e}".replace("+0", "") for a, b in zip(edges[:-1], edges[1:], strict=True)
    ]


def photon_bin_labels(photons: Sequence[float], edges=PHOTON_EDGES) -> list[str]:
    """Bin name of the flat-field counts per pixel of every sample (outliers to the ends)."""
    names = photon_bin_names(edges)
    k = np.searchsorted(edges, np.asarray(photons, dtype=float), side="right") - 1
    return [names[i] for i in np.clip(k, 0, len(names) - 1)]


def grouped_point_metrics(
    errors_um: Sequence[float],
    groups: Sequence,
    order: Sequence[str] | None = None,
    thresholds_um=THRESHOLDS_UM,
) -> dict:
    """:func:`point_metrics` per group label, in ``order`` (default: sorted labels)."""
    err = np.asarray(errors_um, dtype=float)
    keys = np.asarray([str(g) for g in groups])
    present = set(keys.tolist())
    names = sorted(present) if order is None else [k for k in order if k in present]
    return {key: point_metrics(err[keys == key], thresholds_um) for key in names}
