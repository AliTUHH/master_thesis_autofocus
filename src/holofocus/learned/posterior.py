"""Posterior densities on a grid and calibration diagnostics (Stage 7).

:class:`GridPosterior` holds the posterior densities of ``B`` observations on a uniform grid
of the parameter ``t`` over the prior box, each normalized to one there (trapezoidal rule).
It gives the median, mean, standard deviation and mode, central and highest-density (HPD)
intervals, and two statistics of the true value ``t*`` that are uniform on [0, 1] over a
calibration set if the posterior is calibrated ([6], Appendix A5):

* ``pit(t*) = F(t*)``, the probability integral transform: the rank of ``t*`` among
  infinitely many posterior samples divided by their number, the statistic of
  simulation-based calibration (SBC) for one parameter;
* ``hpd_level(t*)``: the posterior mass of ``{t: q(t) > q(t*)}``, the smallest level whose HPD
  region contains ``t*``; the fraction of a calibration set with ``hpd_level <= a`` is the
  expected coverage at level ``a`` ([6]: the rank with ``f = log q``).

For a Gaussian with mean ``m`` and standard deviation ``s`` (the CNN of Stage 4) both have
closed forms (:func:`gaussian_pit`, :func:`gaussian_hpd_level`). :func:`calibration_summary`
combines the coverage curve, the SBC histogram and Kolmogorov-Smirnov tests of uniformity.
A PIT histogram that rises at both ends and a coverage curve below the diagonal mean
overconfident posteriors; a PIT histogram tilted to one side means biased ones.

The second part turns posteriors of ``t = dz01 / scale`` into per-sample rows
(:data:`NPE_PREDICTION_FIELDS`, written by :func:`holofocus.learned.npe.predict_npe_split`)
and metrics (:func:`posterior_metrics`, :func:`summarize_npe_predictions`); none of it needs
sbi.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from ..phantoms import PHANTOM_TYPES
from .metrics import (
    LEVELS,
    grouped_point_metrics,
    photon_bin_labels,
    photon_bin_names,
    point_metrics,
)

COVERAGE_LEVELS = tuple(round(0.02 * k, 2) for k in range(1, 50))
PIT_BINS = 20
NPE_PREDICTION_FIELDS = (
    "index",
    "phantom_type",
    "photons",
    "z01_true",
    "z01_guess",
    "dz01_true",
    "dz01_est",
    "dz01_mean",
    "dz01_mode",
    "sigma",
    "z01_est",
    "error_um",
    "abs_error_um",
    "z_score",
    "z01_lower_68",
    "z01_upper_68",
    "z01_lower_95",
    "z01_upper_95",
    "width_68_um",
    "width_95_um",
    "hpd_width_68_um",
    "hpd_width_95_um",
    "pit",
    "hpd_level",
    "log_density_um",
    "prior_mass",
    "fresnel_number_true",
    "fresnel_number_est",
    "fresnel_error_rel",
    "seconds",
)
"""``dz01_est``: posterior median; ``sigma``: posterior standard deviation; intervals: central
(quantiles) and HPD widths; ``pit`` and ``hpd_level`` of the true ``dz01``;
``log_density_um``: log posterior density of the true ``z01`` per micrometre; ``prior_mass``:
mass of the unnormalized estimator inside the prior box. Lengths in metres unless ``_um``."""


class GridPosterior:
    """Densities ``(B, G)`` on the uniform grid ``theta`` ``(G,)``, normalized on the grid.

    ``prior_mass`` ``(B,)`` is the integral over the grid of the density before
    normalization: for an estimator normalized on the real line, the mass inside the prior
    box (one minus the leakage of [6]).
    """

    def __init__(
        self, theta: np.ndarray, density: np.ndarray, prior_mass: np.ndarray | None = None
    ) -> None:
        self.theta = np.asarray(theta, dtype=float)
        self.density = np.atleast_2d(np.asarray(density, dtype=float))
        if self.theta.ndim != 1 or self.density.shape[1] != len(self.theta):
            raise ValueError("density must have shape (B, G) for a grid theta of shape (G,)")
        self.step = float(self.theta[1] - self.theta[0])
        self.prior_mass = (
            np.ones(len(self.density)) if prior_mass is None else np.asarray(prior_mass, float)
        )
        self._cdf: np.ndarray | None = None

    @classmethod
    def from_log_density(cls, theta: np.ndarray, log_density: np.ndarray) -> GridPosterior:
        """From ``log q`` ``(B, G)`` on the grid ``theta``."""
        theta = np.asarray(theta, dtype=float)
        log_density = np.atleast_2d(np.asarray(log_density, dtype=float))
        peak = log_density.max(axis=1, keepdims=True)
        if not np.isfinite(peak).all():
            raise ValueError("log density without a finite value on the grid")
        unnormalized = np.exp(log_density - peak)
        step = float(theta[1] - theta[0])
        integral = _trapezoid(unnormalized, step)
        return cls(theta, unnormalized / integral[:, None], integral * np.exp(peak[:, 0]))

    @classmethod
    def mixture(cls, members: Sequence[GridPosterior]) -> GridPosterior:
        """Equal-weight mixture of posteriors on the same grid (an ensemble of NPEs)."""
        first = members[0]
        if any(m.density.shape != first.density.shape for m in members) or any(
            not np.array_equal(m.theta, first.theta) for m in members
        ):
            raise ValueError("ensemble members must share the grid and the batch")
        density = np.mean([m.density for m in members], axis=0)
        prior_mass = np.mean([m.prior_mass for m in members], axis=0)
        return cls(first.theta, density, prior_mass)

    def __len__(self) -> int:
        return len(self.density)

    def __getitem__(self, rows) -> GridPosterior:
        rows = np.atleast_1d(np.arange(len(self))[rows])
        return GridPosterior(self.theta, self.density[rows], self.prior_mass[rows])

    def _cell_mass(self) -> np.ndarray:
        return 0.5 * (self.density[:, 1:] + self.density[:, :-1]) * self.step

    def cdf(self) -> np.ndarray:
        """``F`` at the grid points, from 0 to 1."""
        if self._cdf is None:
            cum = np.concatenate(
                [np.zeros((len(self), 1)), np.cumsum(self._cell_mass(), axis=1)], axis=1
            )
            self._cdf = cum / cum[:, -1:]
        return self._cdf

    def quantile(self, q: float) -> np.ndarray:
        """``F^-1(q)`` per row, linear between grid points."""
        if not 0.0 < q < 1.0:
            raise ValueError("q must lie in (0, 1)")
        cdf = self.cdf()
        upper = np.clip((cdf < q).sum(axis=1), 1, len(self.theta) - 1)
        rows = np.arange(len(self))
        f0, f1 = cdf[rows, upper - 1], cdf[rows, upper]
        gap = f1 - f0
        weight = np.where(gap > 0, (q - f0) / np.where(gap > 0, gap, 1.0), 1.0)
        return self.theta[upper - 1] + weight * self.step

    def median(self) -> np.ndarray:
        return self.quantile(0.5)

    def interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Central interval with mass ``level``: the ``(1 -+ level) / 2`` quantiles."""
        return self.quantile(0.5 * (1.0 - level)), self.quantile(0.5 * (1.0 + level))

    def mean(self) -> np.ndarray:
        return _trapezoid(self.density * self.theta, self.step)

    def std(self) -> np.ndarray:
        centred = self.theta[None, :] - self.mean()[:, None]
        return np.sqrt(_trapezoid(self.density * centred**2, self.step))

    def mode(self) -> np.ndarray:
        return self.theta[np.argmax(self.density, axis=1)]

    def density_at(self, t: np.ndarray) -> np.ndarray:
        """Density at ``t`` ``(B,)``, linear between grid points, zero outside the grid."""
        return self._row_linear(self.density, t, outside=(0.0, 0.0))

    def pit(self, t: np.ndarray) -> np.ndarray:
        """``F(t)`` per row (0 below the grid, 1 above)."""
        return self._row_linear(self.cdf(), t, outside=(0.0, 1.0))

    def hpd_level(self, t: np.ndarray) -> np.ndarray:
        """Mass of the grid cells whose mean density exceeds the density at ``t``."""
        cell_density = 0.5 * (self.density[:, 1:] + self.density[:, :-1])
        above = cell_density > self.density_at(t)[:, None]
        return np.minimum((self._cell_mass() * above).sum(axis=1), 1.0)

    def hpd_width(self, level: float) -> np.ndarray:
        """Length of the smallest set of grid cells with mass ``level`` (HPD region; it may
        consist of several intervals)."""
        mass = self._cell_mass()
        order = np.argsort(-mass, axis=1)
        cum = np.cumsum(np.take_along_axis(mass, order, axis=1), axis=1) / mass.sum(axis=1)[:, None]
        cells = np.minimum((cum < level).sum(axis=1) + 1, mass.shape[1])
        return cells * self.step

    def _row_linear(
        self, values: np.ndarray, t: np.ndarray, outside: tuple[float, float]
    ) -> np.ndarray:
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(self),))
        position = (t - self.theta[0]) / self.step
        lower = np.clip(np.floor(position).astype(int), 0, len(self.theta) - 2)
        weight = np.clip(position - lower, 0.0, 1.0)
        rows = np.arange(len(self))
        out = (1.0 - weight) * values[rows, lower] + weight * values[rows, lower + 1]
        out = np.where(t < self.theta[0], outside[0], out)
        return np.where(t > self.theta[-1], outside[1], out)


def _trapezoid(values: np.ndarray, step: float) -> np.ndarray:
    return (values.sum(axis=-1) - 0.5 * (values[..., 0] + values[..., -1])) * step


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return torch.special.ndtr(torch.as_tensor(np.asarray(z, dtype=float))).numpy()


def gaussian_pit(errors: Sequence[float], sigmas: Sequence[float]) -> np.ndarray:
    """``F(truth)`` of a Gaussian centred on the estimate; ``errors = estimate - truth``."""
    return _normal_cdf(-np.asarray(errors, dtype=float) / np.asarray(sigmas, dtype=float))


def gaussian_hpd_level(errors: Sequence[float], sigmas: Sequence[float]) -> np.ndarray:
    """HPD level of the truth under a Gaussian: ``2 Phi(|z|) - 1``."""
    z = np.abs(np.asarray(errors, dtype=float) / np.asarray(sigmas, dtype=float))
    return 2.0 * _normal_cdf(z) - 1.0


def ks_uniform(values: Sequence[float]) -> dict:
    """Kolmogorov-Smirnov test of ``values`` against U(0, 1); p-value from the asymptotic
    Kolmogorov distribution with Stephens' finite-sample correction."""
    x = np.sort(np.clip(np.asarray(values, dtype=float), 0.0, 1.0))
    n = len(x)
    if n == 0:
        return {"n": 0}
    i = np.arange(1, n + 1)
    d = float(max((i / n - x).max(), (x - (i - 1) / n).max()))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    return {"n": n, "statistic": d, "p_value": kolmogorov_sf(lam)}


def kolmogorov_sf(lam: float) -> float:
    """``P(K > lam)`` of the Kolmogorov distribution (two series, each fast where used)."""
    if lam <= 0:
        return 1.0
    if lam < 1.18:
        y = math.exp(-(math.pi**2) / (8.0 * lam**2))
        cdf = math.sqrt(2.0 * math.pi) / lam * (y + y**9 + y**25 + y**49)
        return float(np.clip(1.0 - cdf, 0.0, 1.0))
    x = math.exp(-2.0 * lam**2)
    return float(np.clip(2.0 * (x - x**4 + x**9 - x**16), 0.0, 1.0))


def coverage_curve(hpd_levels: Sequence[float], levels: Sequence[float] = COVERAGE_LEVELS) -> dict:
    """Expected coverage: fraction of truths inside the HPD region of each level."""
    h = np.asarray(hpd_levels, dtype=float)
    levels = np.asarray(levels, dtype=float)
    coverage = np.array([float(np.mean(h <= a)) for a in levels])
    return {
        "levels": levels.tolist(),
        "coverage": coverage.tolist(),
        "max_abs_deviation": float(np.abs(coverage - levels).max()),
        "mean_deviation": float((coverage - levels).mean()),
    }


def pit_histogram(pit: Sequence[float], bins: int = PIT_BINS) -> dict:
    """SBC histogram of the PIT values in ``bins`` equal bins, with the chi-square statistic
    against equal counts (``bins - 1`` degrees of freedom)."""
    p = np.clip(np.asarray(pit, dtype=float), 0.0, 1.0)
    counts = np.histogram(p, bins=bins, range=(0.0, 1.0))[0]
    expected = len(p) / bins
    return {
        "bins": bins,
        "counts": counts.tolist(),
        "expected": expected,
        "chi2": float(((counts - expected) ** 2 / expected).sum()) if len(p) else 0.0,
    }


def calibration_summary(pit: Sequence[float], hpd_levels: Sequence[float]) -> dict:
    """PIT (SBC) and HPD (expected coverage) statistics of a calibration set."""
    pit = np.asarray(pit, dtype=float)
    hpd = np.asarray(hpd_levels, dtype=float)
    return {
        "n": len(pit),
        "pit": {
            "mean": float(pit.mean()),
            "ks": ks_uniform(pit),
            "histogram": pit_histogram(pit),
            "below_0.025": float(np.mean(pit < 0.025)),
            "above_0.975": float(np.mean(pit > 0.975)),
        },
        "hpd": {
            "ks": ks_uniform(hpd),
            "coverage": {"68": float(np.mean(hpd <= 0.68)), "95": float(np.mean(hpd <= 0.95))},
            "curve": coverage_curve(hpd),
        },
    }


def posterior_summaries(post: GridPosterior, scale: float) -> dict[str, np.ndarray]:
    """Median, mean, mode, standard deviation, central intervals and HPD widths of ``dz01``
    (metres) from a posterior of ``t = dz01 / scale``."""
    out = {
        "dz01": post.median() * scale,
        "dz01_mean": post.mean() * scale,
        "dz01_mode": post.mode() * scale,
        "sigma": post.std() * scale,
        "prior_mass": post.prior_mass,
    }
    for level in LEVELS:
        key = f"{round(100 * level)}"
        lower, upper = post.interval(level)
        out[f"lower_{key}"] = lower * scale
        out[f"upper_{key}"] = upper * scale
        out[f"hpd_width_{key}"] = post.hpd_width(level) * scale
    return out


def log_density_um(post: GridPosterior, truth: np.ndarray, scale: float) -> np.ndarray:
    """Log posterior density of ``dz01 = scale * t`` per micrometre at the true ``t``."""
    density = np.maximum(post.density_at(truth), 1e-300)
    return np.log(density) - math.log(scale * 1e6)


def truth_statistics(post: GridPosterior, truth: np.ndarray, scale: float) -> dict:
    """:func:`posterior_summaries` with the PIT, HPD level and log density of the true ``t``."""
    stats = posterior_summaries(post, scale)
    stats["pit"] = post.pit(truth)
    stats["hpd_level"] = post.hpd_level(truth)
    stats["log_density_um"] = log_density_um(post, truth, scale)
    return stats


def posterior_metrics(stats: dict[str, np.ndarray], dz01_true: np.ndarray) -> dict:
    """Point metrics of the median, coverage and widths of the intervals, NLL (nats per
    sample, ``z01`` in micrometres, as :func:`~holofocus.learned.metrics.interval_metrics`)
    and calibration statistics, from :func:`truth_statistics` in metres."""
    errors_um = (stats["dz01"] - dz01_true) * 1e6
    sigma_um = stats["sigma"] * 1e6
    interval = {
        "n": len(errors_um),
        "coverage": {},
        "width_um": {},
        "hpd_coverage": {},
        "hpd_width_um": {},
        "sigma_mean": float(np.mean(sigma_um)),
        "sigma_median": float(np.median(sigma_um)),
        "z_rms": float(np.sqrt(np.mean((errors_um / sigma_um) ** 2))),
        "nll": float(-np.mean(stats["log_density_um"])),
    }
    for level in LEVELS:
        key = f"{round(100 * level)}"
        lower, upper = stats[f"lower_{key}"], stats[f"upper_{key}"]
        width = (upper - lower) * 1e6
        hpd = stats[f"hpd_width_{key}"] * 1e6
        interval["coverage"][key] = float(np.mean((lower <= dz01_true) & (dz01_true <= upper)))
        interval["width_um"][key] = {"mean": float(width.mean()), "median": float(np.median(width))}
        interval["hpd_coverage"][key] = float(np.mean(stats["hpd_level"] <= level))
        interval["hpd_width_um"][key] = {"mean": float(hpd.mean()), "median": float(np.median(hpd))}
    return {
        "error_um": point_metrics(errors_um),
        "interval": interval,
        "calibration": calibration_summary(stats["pit"], stats["hpd_level"]),
        "prior_mass": {
            "min": float(np.min(stats["prior_mass"])),
            "median": float(np.median(stats["prior_mass"])),
        },
        "mean_error_um": point_metrics((stats["dz01_mean"] - dz01_true) * 1e6),
        "mode_error_um": point_metrics((stats["dz01_mode"] - dz01_true) * 1e6),
    }


def write_npe_predictions(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=NPE_PREDICTION_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: repr(float(v)) if isinstance(v, float) else v for k, v in row.items()}
            )


def prediction_stats(rows: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The columns of NPE prediction rows as :func:`posterior_metrics` expects them, and the
    true ``dz01``."""
    col = {k: np.array([r[k] for r in rows], dtype=float) for k in NPE_PREDICTION_FIELDS[3:]}
    guess = col["z01_guess"]
    stats = {"dz01": col["dz01_est"]}
    for key in ("dz01_mean", "dz01_mode", "sigma", "prior_mass", "pit", "hpd_level"):
        stats[key] = col[key]
    stats["log_density_um"] = col["log_density_um"]
    for key in ("68", "95"):
        stats[f"lower_{key}"] = col[f"z01_lower_{key}"] - guess
        stats[f"upper_{key}"] = col[f"z01_upper_{key}"] - guess
        stats[f"hpd_width_{key}"] = col[f"hpd_width_{key}_um"] * 1e-6
    return stats, col["dz01_true"]


def summarize_npe_predictions(rows: list[dict]) -> dict:
    """Accuracy of the median, intervals, calibration, cost and breakdowns of NPE predictions
    (rows of :data:`NPE_PREDICTION_FIELDS`, e.g. from
    :func:`~holofocus.learned.inference.read_predictions`)."""
    stats, dz01_true = prediction_stats(rows)
    err = np.array([r["error_um"] for r in rows])
    fr = np.array([r["fresnel_error_rel"] for r in rows])
    seconds = np.array([r["seconds"] for r in rows])
    names = [PHANTOM_TYPES[r["phantom_type"]] for r in rows]
    photons = [r["photons"] for r in rows]
    return {
        "n_samples": len(rows),
        **posterior_metrics(stats, dz01_true),
        "fresnel_error_rel": {
            "bias": float(fr.mean()),
            "mae": float(np.abs(fr).mean()),
            "rmse": float(np.sqrt(np.mean(fr**2))),
            "median_abs": float(np.median(np.abs(fr))),
        },
        "guess_error_um": {"mae": float(np.mean(np.abs(dz01_true))) * 1e6},
        "seconds_per_estimate": {
            "mean": float(seconds.mean()),
            "median": float(np.median(seconds)),
            "max": float(seconds.max()),
        },
        "by_phantom_type": grouped_point_metrics(err, names, order=PHANTOM_TYPES),
        "by_photons": grouped_point_metrics(
            err, photon_bin_labels(photons), order=photon_bin_names()
        ),
    }
