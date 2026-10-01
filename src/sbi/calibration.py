"""Calibration and uncertainty diagnostics for 1-D posteriors given as sample arrays.

All functions take posterior samples of shape ``[N_obs, N_samples]`` (one row per observation) and the
true parameter values of shape ``[N_obs]``; they are pure NumPy/SciPy and independent of ``sbi`` so
that they can be unit-tested on synthetic Gaussian posteriors.

Diagnostics
-----------
* **Coverage** of central credible intervals: for nominal level ``1 - alpha`` the interval is
  ``[q(alpha/2), q(1 - alpha/2)]`` of the samples; the empirical coverage is the fraction of observations
  whose true value lies inside.  A calibrated posterior has coverage == level; the binomial (Wilson)
  interval quantifies the sampling uncertainty of that fraction for small test sets.
* **SBC** (simulation-based calibration, Talts et al. 2018): the rank of the true value among the
  ``N_samples`` posterior samples is uniform on ``{0, ..., N_samples}`` if the posterior is calibrated;
  a KS test against the uniform distribution and the histogram shape (U = over-confident, hump = under-
  confident, slope = biased) summarise deviations.
* **Informativeness**: correlation between the posterior standard deviation and the absolute error of the
  point estimate, and the risk-coverage curve (error after discarding the most uncertain observations).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

__all__ = [
    "DEFAULT_COVERAGE_LEVELS",
    "central_interval",
    "wilson_interval",
    "empirical_coverage",
    "coverage_curve",
    "expected_coverage_error",
    "sbc_ranks",
    "rank_uniformity_test",
    "uncertainty_error_relation",
    "risk_coverage_curve",
]

DEFAULT_COVERAGE_LEVELS: tuple[float, ...] = (0.5, 0.68, 0.9, 0.95)


def _samples_2d(samples: ArrayLike) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"samples must have shape [N_obs, N_samples >= 2], got {arr.shape}")
    return arr


def _truth_1d(truth: ArrayLike, n_obs: int) -> np.ndarray:
    arr = np.asarray(truth, dtype=np.float64).reshape(-1)
    if arr.size != n_obs:
        raise ValueError(f"expected {n_obs} true values, got {arr.size}")
    return arr


def _check_level(level: float) -> float:
    if not 0.0 < level < 1.0:
        raise ValueError(f"credible level must be in (0, 1), got {level}")
    return float(level)


def central_interval(samples: ArrayLike, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds ``[N_obs]`` of the central ``level`` credible interval (sample quantiles)."""
    arr = _samples_2d(samples)
    alpha = 1.0 - _check_level(level)
    lo = np.quantile(arr, alpha / 2.0, axis=1)
    hi = np.quantile(arr, 1.0 - alpha / 2.0, axis=1)
    return lo, hi


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (robust for small ``trials`` and extreme rates)."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must be in [0, trials]")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    centre = (p + z**2 / (2.0 * trials)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / trials + z**2 / (4.0 * trials**2)) / denominator
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def empirical_coverage(
    samples: ArrayLike,
    truth: ArrayLike,
    levels: Sequence[float] = DEFAULT_COVERAGE_LEVELS,
) -> list[dict[str, float | int]]:
    """Empirical coverage of central credible intervals with 95 % Wilson intervals.

    Returns one record per level: ``level``, ``coverage``, ``n_covered``, ``n``, ``ci_low``, ``ci_high``
    and ``mean_width`` (mean interval width in parameter units).
    """
    arr = _samples_2d(samples)
    true = _truth_1d(truth, arr.shape[0])
    records: list[dict[str, float | int]] = []
    for level in levels:
        lo, hi = central_interval(arr, level)
        covered = (true >= lo) & (true <= hi)
        n_covered = int(covered.sum())
        ci_low, ci_high = wilson_interval(n_covered, true.size)
        records.append(
            {
                "level": float(level),
                "coverage": n_covered / true.size,
                "n_covered": n_covered,
                "n": int(true.size),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_width": float(np.mean(hi - lo)),
            }
        )
    return records


def coverage_curve(samples: ArrayLike, truth: ArrayLike, num_levels: int = 19) -> dict[str, np.ndarray]:
    """Nominal levels ``0.05 .. 0.95`` versus empirical coverage (for the calibration plot)."""
    levels = np.linspace(0.05, 0.95, num_levels)
    records = empirical_coverage(samples, truth, levels)
    return {
        "nominal": levels,
        "empirical": np.array([record["coverage"] for record in records]),
        "ci_low": np.array([record["ci_low"] for record in records]),
        "ci_high": np.array([record["ci_high"] for record in records]),
    }


def expected_coverage_error(curve: dict[str, np.ndarray]) -> float:
    """Mean absolute deviation between empirical and nominal coverage over the levels of a curve."""
    return float(np.mean(np.abs(curve["empirical"] - curve["nominal"])))


def sbc_ranks(samples: ArrayLike, truth: ArrayLike) -> np.ndarray:
    """Rank of each true value among its posterior samples: number of samples strictly below the truth.

    Ranks take values in ``{0, ..., N_samples}``; for a calibrated posterior they are uniformly distributed.
    A truth below the prior support yields rank 0, above it ``N_samples``.
    """
    arr = _samples_2d(samples)
    true = _truth_1d(truth, arr.shape[0])
    return (arr < true[:, None]).sum(axis=1).astype(np.int64)


def rank_uniformity_test(ranks: ArrayLike, n_samples: int) -> dict[str, float]:
    """Kolmogorov-Smirnov test of the SBC ranks against ``Uniform(0, n_samples)`` (as in ``sbi.check_sbc``).

    Returns the KS statistic, its p-value and the fraction of ranks in the two outermost 5 % bins
    (``tail_fraction``; 0.10 expected, larger values indicate over-confidence).
    """
    arr = np.asarray(ranks, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError("need at least two ranks")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    result = stats.kstest(arr, stats.uniform(loc=0.0, scale=float(n_samples)).cdf)
    edge = 0.05 * n_samples
    tail = float(np.mean((arr <= edge) | (arr >= n_samples - edge)))
    return {"ks_statistic": float(result.statistic), "ks_pvalue": float(result.pvalue), "tail_fraction": tail}


def uncertainty_error_relation(abs_error: ArrayLike, uncertainty: ArrayLike) -> dict[str, float]:
    """Pearson and Spearman correlation between an uncertainty measure (e.g. posterior std) and ``|error|``."""
    err = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if err.shape != unc.shape or err.size < 3:
        raise ValueError("abs_error and uncertainty must have the same length >= 3")
    if np.allclose(unc, unc[0]) or np.allclose(err, err[0]):
        return {"pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    pearson = stats.pearsonr(unc, err)
    spearman = stats.spearmanr(unc, err)
    return {
        "pearson_r": float(pearson[0]),
        "pearson_p": float(pearson[1]),
        "spearman_rho": float(spearman[0]),
        "spearman_p": float(spearman[1]),
    }


def risk_coverage_curve(
    abs_error: ArrayLike,
    uncertainty: ArrayLike,
    keep_fractions: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
) -> list[dict[str, Any]]:
    """Error statistics after discarding the most uncertain observations.

    For every ``keep_fraction`` the ``ceil(keep_fraction * N)`` observations with the *smallest*
    uncertainty are kept; reported are ``mae``, ``rmse``, ``p95`` of the kept errors and the uncertainty
    threshold used.  A useful uncertainty makes the error fall monotonically with decreasing fraction.
    """
    err = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    unc = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    if err.shape != unc.shape or err.size == 0:
        raise ValueError("abs_error and uncertainty must have the same non-zero length")
    order = np.argsort(unc, kind="stable")
    records: list[dict[str, Any]] = []
    for fraction in keep_fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {fraction}")
        n_keep = max(1, int(np.ceil(fraction * err.size)))
        kept = order[:n_keep]
        records.append(
            {
                "keep_fraction": float(fraction),
                "n_kept": int(n_keep),
                "uncertainty_threshold": float(unc[kept].max()),
                "mae": float(err[kept].mean()),
                "rmse": float(np.sqrt(np.mean(err[kept] ** 2))),
                "p95": float(np.percentile(err[kept], 95)),
            }
        )
    return records
