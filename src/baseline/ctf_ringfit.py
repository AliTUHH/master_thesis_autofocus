"""Learning-free Fresnel-number estimation from the CTF rings of a single hologram (``CTFFIND``-like).

Usage::

    python -m src.baseline.ctf_ringfit --data data/processed/small/test.hdf5 [--n N] [--out reports/ringfit/]
        [--hologram-key images/hologram] [--z01-range MIN MAX | --fr-min F --fr-max F] [--template cos|comb]

Physics
-------
For a weak object the Fourier transform of the hologram intensity (and, to first order, of the stored
amplitude ``a = |D_Fr(psi)|``) is ``2 sin(chi) phi~ - 2 cos(chi) mu~`` with ``chi = pi * u / Fr`` and
``u = xi^2 + eta^2`` in cycles^2/pixel^2.  Its power spectrum therefore oscillates **periodically in
u with period Fr**::

    |a~(u)|^2 ~ S(u) * [1 - m cos(2 pi u / Fr + theta)] + F

where ``S`` is the smooth object envelope (power law for edge-dominated phantoms), ``F`` the white-noise
floor, ``m`` the modulation depth and ``theta`` the mixing phase (``theta = 0`` for a pure phase object,
``theta = pi`` for pure absorption; strong objects fill the zeros and shift the phase).  Estimating
``Fr`` is thus a 1-D period estimation problem on the azimuthally averaged log power spectrum.

Algorithm (``template`` method, default)
----------------------------------------
1. relative contrast ``x / mean(x) - 1`` (optionally tapered with a Tukey/Hann window), ``|FFT2|^2``;
2. azimuthal mean in ``n_bins`` bins uniform in ``u`` (equal pixel count per bin, rings equidistant);
3. robust fit of the envelope ``log(a u^-b + F)`` (power law + noise floor, soft-L1 loss); SNR weights
   ``w = min(1, (S/F)^2)`` restrict the fit to frequencies where the object spectrum is above the noise;
4. for every candidate ``Fr`` on a logarithmic grid: residual ``r = log P - log env`` detrended with a
   running mean of width exactly ``Fr`` (removes the envelope misfit but not the period-``Fr`` signal),
   score = phase-free normalised correlation of ``r`` with ``cos/sin(2 pi u / Fr)`` (``template="cos"``)
   or the fixed-phase correlation with ``log(sin^2(pi u / Fr) + eps)`` (``template="comb"``, sharper,
   best for weak objects at low SNR);
5. Brent refinement of the best grid point (``scipy.optimize.minimize_scalar``).

The alternative ``minima`` method (plausibility check) locates the individual ring minima near
``n * Fr`` in the detrended residual and fits ``u_n = n * Fr + c`` by weighted least squares.

Resolution limit
----------------
The ring spacing in ``u`` is ``Fr``; the discrete spectrum of an ``N``-pixel image resolves ``u`` only to
``du = 2 |xi| / N`` at radius ``|xi|``.  Rings are therefore resolvable while ``Fr > 2 |xi| / N``, i.e. for
``|xi| < N Fr / 2`` (:func:`resolvable_frequency_limit`).  With a Hann window the resolution is ~2x
worse.  At least one full ring must fit inside the circle ``u <= u_max``: ``Fr < u_max``.  The number of
resolvable rings is ``min(u_max, (N Fr / 2)^2) / Fr`` (:func:`max_resolvable_rings`), i.e. ``N^2 Fr / 4``
when the window limit dominates.  Empirically (synthetic weak phase objects, 0-1 % noise) the fit is
reliable from about 10 resolvable rings on (``N^2 Fr >~ 40``), marginal with 6-8 and fails below ~4.

The radial binning must sample the smallest candidate period with at least :data:`MIN_BINS_PER_PERIOD`
bins (:func:`default_n_bins`); a template sampled with 1-2 bins per period aliases and yields spurious
correlations (e.g. ``Fr = 1e-4`` on 2048 px with ``4 N`` bins would have only 3 bins per period).

Downsampled input: a hologram pooled by ``s`` has ``Fr_eff = s^2 Fr``; pass ``--downsample s`` to the
CLI and the estimate is divided by ``s^2`` before it is compared with the labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal.windows import tukey

from src.baseline.results import make_result_row, summarize
from src.utils.config import PROJECT_ROOT, resolve_path
from src.utils.physics import fresnel_number, z01_from_fresnel_number

__all__ = [
    "RingFitConfig",
    "RingFitResult",
    "WINDOWS",
    "TEMPLATES",
    "radial_power_profile",
    "fit_envelope",
    "ring_fit",
    "estimate_fresnel_number",
    "resolvable_frequency_limit",
    "max_resolvable_rings",
    "fresnel_number_resolution_limit",
    "default_n_bins",
    "MIN_BINS_PER_PERIOD",
    "evaluate_file",
    "main",
]

WINDOWS: tuple[str, ...] = ("none", "tukey", "hann")
TEMPLATES: tuple[str, ...] = ("cos", "comb")
_EPS = 1e-30
DYNAMIC_RANGE = 1e-6  # lower clamp of the radial profile relative to its upper level (60 dB), see ring_fit


# --------------------------------------------------------------------------------------------------
# resolution criteria
# --------------------------------------------------------------------------------------------------
def resolvable_frequency_limit(fr: float, n_pixels: int, window: str = "none") -> float:
    """Largest ``|xi|`` (cycles/pixel) at which neighbouring CTF rings are still resolved: ``N Fr / 2``.

    Derivation: ring spacing ``du = Fr`` vs. frequency resolution ``du_res = 2 |xi| d|xi|`` with
    ``d|xi| = 1/N`` (``2/N`` for a Hann window): ``Fr > 2 |xi| / N  <=>  |xi| < N Fr / 2``.
    """
    factor = 0.5 if window == "hann" else 1.0
    return factor * n_pixels * fr / 2.0


def max_resolvable_rings(fr: float, n_pixels: int, u_max: float = 0.25, window: str = "none") -> int:
    """Number of rings ``n`` with ``n Fr <= min(u_max, (N Fr / 2)^2)``."""
    u_lim = min(u_max, resolvable_frequency_limit(fr, n_pixels, window) ** 2)
    return int(np.floor(u_lim / fr))


def fresnel_number_resolution_limit(n_pixels: int, u_max: float = 0.25, window: str = "none") -> float:
    """Smallest ``Fr`` whose rings are resolved out to ``|xi| = sqrt(u_max)``: ``2 sqrt(u_max) / N``."""
    factor = 2.0 if window == "hann" else 1.0
    return factor * 2.0 * np.sqrt(u_max) / n_pixels


# --------------------------------------------------------------------------------------------------
# configuration / result containers
# --------------------------------------------------------------------------------------------------
@dataclass
class RingFitConfig:
    """Parameters of :func:`ring_fit`; ``fr_min``/``fr_max`` bound the search (known setup range)."""

    fr_min: float
    fr_max: float
    grid_step: float = 0.01
    """Relative spacing of the logarithmic Fr grid (1 % by default)."""
    n_bins: int | None = None
    """Radial bins over ``[0, u_max]``; default ``4 N`` clipped to ``[256, 16384]``."""
    u_max: float = 0.25
    u_min: float = 1e-3
    window: str = "tukey"
    tukey_alpha: float = 0.2
    template: str = "cos"
    comb_eps: float = 0.1
    snr_weights: bool = True
    presmooth_bins: float = 1.0
    """Gaussian pre-smoothing (in bins) of the log profile; the bins oversample the FFT resolution."""
    matched_detrend: bool | None = None
    """Detrend the log ratio with a running mean of exactly one candidate period before scoring; ``None`` =
    automatic (on for ``cos``, where it removes the long-period bias of the envelope misfit; off for ``comb``)."""
    refine: bool = True
    minima: bool = True
    minima_min_depth: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.fr_min < self.fr_max:
            raise ValueError(f"require 0 < fr_min < fr_max, got {self.fr_min}, {self.fr_max}")
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {self.window!r}")
        if self.template not in TEMPLATES:
            raise ValueError(f"template must be one of {TEMPLATES}, got {self.template!r}")
        if not 0 < self.u_min < self.u_max <= 0.5:
            raise ValueError(f"require 0 < u_min < u_max <= 0.5, got {self.u_min}, {self.u_max}")
        if self.grid_step <= 0:
            raise ValueError("grid_step must be positive")
        if self.matched_detrend is None:
            self.matched_detrend = self.template == "cos"

    @property
    def fr_grid(self) -> np.ndarray:
        n_grid = int(np.ceil(np.log(self.fr_max / self.fr_min) / np.log1p(self.grid_step))) + 1
        return np.geomspace(self.fr_min, self.fr_max, max(n_grid, 2))


@dataclass
class RingFitResult:
    """Estimate and diagnostics of one ring fit."""

    fr: float
    """Fresnel number from the template method (grid search + Brent refinement)."""
    score: float
    """Normalised correlation of the residual with the template at ``fr`` (0..1)."""
    phase_rad: float
    """Phase of the fitted oscillation ``cos(2 pi u / Fr - phase)``: ``pi`` = pure phase object, ``0`` = absorption."""
    fr_grid_best: float
    fr_minima: float
    """Fresnel number from the ring-minima regression (``nan`` if fewer than two minima were found)."""
    minima_intercept: float
    n_minima: int
    n_rings_visible: int
    """Rings ``n Fr`` inside the frequency range where the object spectrum exceeds the noise floor (weight > 0.1)."""
    envelope: dict[str, float]
    elapsed_s: float
    u: np.ndarray = field(repr=False)
    profile: np.ndarray = field(repr=False)
    env: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    residual: np.ndarray = field(repr=False)
    fr_grid: np.ndarray = field(repr=False)
    scores: np.ndarray = field(repr=False)
    minima_u: np.ndarray = field(repr=False)
    n_evals: int = 0
    """Number of template-score evaluations (grid + Brent refinement)."""

    def summary(self) -> dict[str, Any]:
        """Scalar fields only (JSON friendly)."""
        return {
            key: (float(value) if isinstance(value, (float, np.floating)) else value)
            for key, value in asdict(self).items()
            if not isinstance(value, np.ndarray)
        }


# --------------------------------------------------------------------------------------------------
# spectrum and envelope
# --------------------------------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _radial_index(height: int, width: int, n_bins: int, u_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi = np.fft.fftshift(np.fft.fftfreq(height))
    eta = np.fft.fftshift(np.fft.fftfreq(width))
    u = (xi[:, None] ** 2 + eta[None, :] ** 2).reshape(-1)
    index = np.floor(u / u_max * n_bins).astype(np.int64)
    index[(index < 0) | (index >= n_bins)] = n_bins
    counts = np.bincount(index, minlength=n_bins + 1)[:n_bins]
    edges = np.linspace(0.0, u_max, n_bins + 1)
    return index, counts, 0.5 * (edges[:-1] + edges[1:])


@lru_cache(maxsize=8)
def _window(height: int, width: int, kind: str, alpha: float) -> np.ndarray:
    if kind == "none":
        return np.ones((height, width))
    if kind == "hann":
        return np.outer(np.hanning(height), np.hanning(width))
    return np.outer(tukey(height, alpha), tukey(width, alpha))


MIN_BINS_PER_PERIOD = 8  # the smallest candidate period must span this many radial bins (no aliasing of the template)


def default_n_bins(n_pixels: int, fr_min: float | None = None, u_max: float = 0.25) -> int:
    """``4 N`` bins (about ``pi N / 16`` pixels per bin), raised so that the smallest candidate Fresnel number
    still spans :data:`MIN_BINS_PER_PERIOD` bins -- a template sampled with ~1-2 bins per period aliases to a
    long-period pattern and produces spurious correlations."""
    n_bins = 4 * n_pixels
    if fr_min is not None and fr_min > 0:
        n_bins = max(n_bins, int(np.ceil(MIN_BINS_PER_PERIOD * u_max / fr_min)))
    return int(np.clip(n_bins, 256, 65536))


def radial_power_profile(
    hologram: np.ndarray,
    n_bins: int | None = None,
    u_max: float = 0.25,
    window: str = "tukey",
    tukey_alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Azimuthal mean power spectrum of ``x/mean(x) - 1`` in bins uniform in ``u = |xi|^2``.

    Returns ``(u_centres, profile, counts)``; the power is normalised by the pixel number so that white
    noise of variance ``sigma^2`` gives ``profile ~ sigma^2 * mean(window^2)``.  Empty bins (possible at
    the smallest radii of coarse grids) are linearly interpolated.
    """
    x = np.asarray(hologram, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D hologram, got shape {x.shape}")
    height, width = x.shape
    n_bins = default_n_bins(min(height, width)) if n_bins is None else int(n_bins)
    mean = x.mean()
    rel = (x / mean - 1.0) if mean != 0 else x
    rel = rel * _window(height, width, window, tukey_alpha)
    power = np.abs(np.fft.fftshift(np.fft.fft2(rel))) ** 2 / (height * width)
    index, counts, centres = _radial_index(height, width, n_bins, float(u_max))
    sums = np.bincount(index, weights=power.reshape(-1), minlength=n_bins + 1)[:n_bins]
    profile = np.full(n_bins, np.nan)
    filled = counts > 0
    profile[filled] = sums[filled] / counts[filled]
    if not filled.all():
        profile[~filled] = np.interp(centres[~filled], centres[filled], profile[filled])
    return centres, profile, counts


def fit_envelope(u: np.ndarray, profile: np.ndarray, select: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Robust fit of ``log P = log(a u^-b + F)`` on the selected bins.

    Returns ``(envelope, signal_part S, noise floor F, parameters)``; the CTF modulation averages into
    the amplitude ``a`` because the soft-L1 loss down-weights the deep ring minima.
    """
    us = u[select]
    ys = np.log(np.maximum(profile[select], _EPS))
    floor0 = float(np.median(profile[u > 0.8 * u.max()]))
    floor0 = max(floor0, _EPS)
    low = us <= np.percentile(us, 20)
    design = np.vstack([np.ones(low.sum()), -np.log(us[low])]).T
    coef, *_ = np.linalg.lstsq(design, ys[low], rcond=None)
    p0 = np.array([coef[0], float(np.clip(coef[1], 0.1, 6.0)), np.log(floor0)])

    def residuals(p: np.ndarray) -> np.ndarray:
        return np.log(np.exp(p[0]) * us ** (-p[1]) + np.exp(p[2])) - ys

    fit = least_squares(residuals, p0, loss="soft_l1", f_scale=0.5, max_nfev=300, bounds=([-np.inf, 0.0, -np.inf], [np.inf, 8.0, np.inf]))
    log_a, slope, log_f = fit.x
    signal = np.exp(log_a) * np.maximum(u, u[u > 0].min()) ** (-slope)
    floor = float(np.exp(log_f))
    return signal + floor, signal, floor, {"log_a": float(log_a), "power_law_exponent": float(slope), "noise_floor": floor}


def _running_mean(y: np.ndarray, width_bins: float) -> np.ndarray:
    """Boxcar mean with a fractional window length (via interpolated cumulative sums, edge-truncated)."""
    n = len(y)
    half = max(width_bins, 1.0) / 2.0
    cumulative = np.concatenate([[0.0], np.cumsum(y)])
    positions = np.arange(n) + 0.5
    lo = np.clip(positions - half, 0.0, n)
    hi = np.clip(positions + half, 0.0, n)
    grid = np.arange(n + 1, dtype=np.float64)
    return (np.interp(hi, grid, cumulative) - np.interp(lo, grid, cumulative)) / np.maximum(hi - lo, 1e-9)


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    return float((values * weights).sum() / total) if total > 0 else float(values.mean())


# --------------------------------------------------------------------------------------------------
# core estimator
# --------------------------------------------------------------------------------------------------
class _Scorer:
    """Evaluates the template score for arbitrary (continuous) Fresnel numbers on one profile."""

    def __init__(self, u: np.ndarray, log_ratio: np.ndarray, weights: np.ndarray, config: RingFitConfig) -> None:
        self.u = u
        self.y = log_ratio
        self.w = weights
        self.du = float(u[1] - u[0])
        self.cfg = config
        self.n_evals = 0

    def residual(self, fr: float) -> np.ndarray:
        r = self.y - _running_mean(self.y, fr / self.du) if self.cfg.matched_detrend else self.y
        return r - _weighted_average(r, self.w)

    def score(self, fr: float) -> tuple[float, float]:
        """``(normalised correlation, phase)`` of the residual with the template of period ``fr``; zero for
        periods the radial binning cannot represent (fewer than half of :data:`MIN_BINS_PER_PERIOD` bins)."""
        self.n_evals += 1
        if fr < 0.5 * MIN_BINS_PER_PERIOD * self.du:
            return 0.0, 0.0
        r = self.residual(fr)
        w = self.w
        rr = float((w * r * r).sum())
        if rr <= 0:
            return 0.0, 0.0
        arg = 2.0 * np.pi * self.u / fr
        if self.cfg.template == "comb":
            t = np.log(np.sin(0.5 * arg) ** 2 + self.cfg.comb_eps)
            t = t - _weighted_average(t, w)
            tt = float((w * t * t).sum())
            corr = float((w * r * t).sum()) / np.sqrt(rr * tt + _EPS)
            return max(corr, 0.0), np.pi
        c = np.cos(arg)
        s = np.sin(arg)
        c = c - _weighted_average(c, w)
        s = s - _weighted_average(s, w)
        gram = np.array([[(w * c * c).sum(), (w * c * s).sum()], [(w * c * s).sum(), (w * s * s).sum()]])
        rhs = np.array([(w * r * c).sum(), (w * r * s).sum()])
        try:
            coef = np.linalg.solve(gram + 1e-12 * np.eye(2), rhs)
        except np.linalg.LinAlgError:
            return 0.0, 0.0
        explained = float(rhs @ coef)
        phase = float(np.arctan2(coef[1], coef[0]))
        return float(np.sqrt(max(explained, 0.0) / rr)), phase


def _find_minima(u: np.ndarray, residual: np.ndarray, weights: np.ndarray, fr: float, min_depth: float) -> np.ndarray:
    """Positions of the ring minima near ``n * fr`` (parabolic sub-bin refinement); ``nan`` where none is found."""
    du = float(u[1] - u[0])
    usable = u[weights > 0.1]
    if usable.size == 0:
        return np.empty(0)
    u_lim = float(usable.max())
    n_max = int(np.floor(u_lim / fr))
    positions = []
    for n in range(1, n_max + 1):
        lo, hi = (n - 0.4) * fr, (n + 0.4) * fr
        window = (u >= lo) & (u <= hi)
        if window.sum() < 3:
            positions.append(np.nan)
            continue
        idx = np.flatnonzero(window)
        k = idx[np.argmin(residual[idx])]
        depth = residual[idx].mean() - residual[k]
        if depth < min_depth or k == 0 or k == len(u) - 1:
            positions.append(np.nan)
            continue
        y0, y1, y2 = residual[k - 1], residual[k], residual[k + 1]
        denominator = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denominator if denominator > 0 else 0.0
        positions.append(u[k] + float(np.clip(offset, -1.0, 1.0)) * du)
    return np.asarray(positions, dtype=np.float64)


def ring_fit(hologram: np.ndarray, config: RingFitConfig) -> RingFitResult:
    """Estimate the Fresnel number of one hologram (2-D array, amplitude or intensity) from its CTF rings."""
    start = time.perf_counter()
    x = np.asarray(hologram, dtype=np.float64)
    n_pixels = min(x.shape)
    n_bins = default_n_bins(n_pixels, config.fr_min, config.u_max) if config.n_bins is None else int(config.n_bins)
    u, profile, _ = radial_power_profile(x, n_bins, config.u_max, config.window, config.tukey_alpha)
    select = u >= config.u_min
    # Noise-free (simulated) holograms have CTF zeros many decades deep; clamping the dynamic range keeps those
    # bins from dominating the log-domain envelope fit. Real data has a noise floor far above this clamp.
    profile = np.maximum(profile, DYNAMIC_RANGE * float(np.percentile(profile[select], 95)))
    env, signal, floor, env_params = fit_envelope(u, profile, select)

    log_profile = np.log(np.maximum(profile, _EPS))
    if config.presmooth_bins > 0:
        log_profile = gaussian_filter1d(log_profile, config.presmooth_bins, mode="nearest")
    log_ratio = log_profile - np.log(env)
    weights = np.minimum(1.0, (signal / max(floor, _EPS)) ** 2) if config.snr_weights else np.ones_like(u)
    weights = weights * select

    scorer = _Scorer(u, log_ratio, weights, config)
    fr_grid = config.fr_grid
    scores = np.array([scorer.score(fr)[0] for fr in fr_grid])
    k_best = int(np.argmax(scores))
    fr_best = float(fr_grid[k_best])
    if config.refine and len(fr_grid) > 2:
        lo = float(fr_grid[max(k_best - 1, 0)])
        hi = float(fr_grid[min(k_best + 1, len(fr_grid) - 1)])
        if hi > lo:
            result = minimize_scalar(lambda f: -scorer.score(f)[0], bounds=(lo, hi), method="bounded", options={"xatol": lo * 1e-5})
            if -result.fun >= scores[k_best]:
                fr_best = float(result.x)
    score, phase = scorer.score(fr_best)
    residual = scorer.residual(fr_best)

    minima_u = np.empty(0)
    fr_minima, intercept, n_minima = float("nan"), float("nan"), 0
    if config.minima:
        minima_u = _find_minima(u, residual, weights, fr_best, config.minima_min_depth)
        valid = np.isfinite(minima_u)
        n_minima = int(valid.sum())
        if n_minima >= 2:
            orders = np.arange(1, len(minima_u) + 1, dtype=np.float64)[valid]
            design = np.vstack([orders, np.ones_like(orders)]).T
            coef, *_ = np.linalg.lstsq(design, minima_u[valid], rcond=None)
            fr_minima, intercept = float(coef[0]), float(coef[1])

    visible = u[weights > 0.1]
    n_rings_visible = int(np.floor(float(visible.max()) / fr_best)) if visible.size else 0
    return RingFitResult(
        fr=fr_best,
        score=float(score),
        phase_rad=float(phase),
        fr_grid_best=float(fr_grid[k_best]),
        fr_minima=fr_minima,
        minima_intercept=intercept,
        n_minima=n_minima,
        n_rings_visible=n_rings_visible,
        envelope=env_params,
        elapsed_s=time.perf_counter() - start,
        u=u,
        profile=profile,
        env=env,
        weights=weights,
        residual=residual,
        fr_grid=fr_grid,
        scores=scores,
        minima_u=minima_u,
        n_evals=scorer.n_evals,
    )


def estimate_fresnel_number(hologram: np.ndarray, fr_min: float, fr_max: float, **options: Any) -> float:
    """Convenience wrapper returning only the template estimate of ``Fr``."""
    return ring_fit(hologram, RingFitConfig(fr_min=fr_min, fr_max=fr_max, **options)).fr


# --------------------------------------------------------------------------------------------------
# evaluation on an HDF5 file
# --------------------------------------------------------------------------------------------------
def _percentiles(values: np.ndarray) -> dict[str, float]:
    abs_values = np.abs(values)
    return {
        "mae": float(abs_values.mean()),
        "median": float(np.median(abs_values)),
        "p95": float(np.percentile(abs_values, 95)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "bias": float(values.mean()),
    }


def evaluate_file(
    data_path: str | Path,
    out_dir: str | Path,
    num_samples: int | None = None,
    hologram_key: str = "images/hologram",
    z01_range_mm: tuple[float, float] | None = None,
    fr_range: tuple[float, float] | None = None,
    range_margin: float = 1.4,
    downsample: int = 1,
    n_examples: int = 3,
    **options: Any,
) -> dict[str, Any]:
    """Run the ring fit over the first ``num_samples`` holograms of an HDF5 file and write a report.

    The search range is derived from ``z01_range_mm`` (setup range) or ``fr_range``; by default the per-file
    z01 range widened by ``range_margin`` (``/1.4 .. *1.4``) is used and reported.  Writes ``samples.csv``
    (common per-sample schema of :mod:`src.baseline.results` followed by ring-fit diagnostics), ``summary.json``
    (with the common statistics under ``"common"``), a scatter plot and ``n_examples`` radial-profile figures
    into ``out_dir``.
    """
    data_path = resolve_path(data_path)
    out_dir = resolve_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path, "r") as handle:
        setup = handle["metadata/setup"]
        z01 = setup["z01"][...].astype(np.float64)
        z02 = setup["z02"][...].astype(np.float64)
        fr_true = setup["Fr"][...].astype(np.float64)
        energy = float(setup["energy"][0])
        px_mm = float(setup["detector_px_size"][0])
        total = handle[hologram_key].shape[0]
        n = total if num_samples is None else min(int(num_samples), total)
        shape = handle[hologram_key].shape[1:]

    if fr_range is None:
        if z01_range_mm is None:
            z01_range_mm = (float(z01[:n].min()) / range_margin, min(float(z01[:n].max()) * range_margin, float(z02.min()) * 0.9))
        fr_lo = float(fresnel_number(z01_range_mm[0], float(z02.min()), energy, px_mm))
        fr_hi = float(fresnel_number(z01_range_mm[1], float(z02.max()), energy, px_mm))
        fr_range = (min(fr_lo, fr_hi), max(fr_lo, fr_hi))
    scale = float(downsample) ** 2
    config = RingFitConfig(fr_min=fr_range[0] * scale, fr_max=fr_range[1] * scale, **options)
    method = f"ringfit_{config.template}"

    rows: list[dict[str, Any]] = []
    examples: list[tuple[int, RingFitResult]] = []
    with h5py.File(data_path, "r") as handle:
        dataset = handle[hologram_key]
        for i in range(n):
            x = dataset[i].astype(np.float64)
            if downsample > 1:
                h, w = (x.shape[0] // downsample) * downsample, (x.shape[1] // downsample) * downsample
                x = x[:h, :w].reshape(h // downsample, downsample, w // downsample, downsample).mean(axis=(1, 3))
            result = ring_fit(x, config)
            fr_hat = result.fr / scale
            fr_hat_minima = result.fr_minima / scale
            z01_hat = float(z01_from_fresnel_number(fr_hat, z02[i], energy, px_mm))
            z01_hat_minima = float(z01_from_fresnel_number(fr_hat_minima, z02[i], energy, px_mm)) if np.isfinite(fr_hat_minima) else float("nan")
            # common schema (src.baseline.results) first, ring-fit specific diagnostics after it; the historical
            # column names fr_rel_err_pct / z01_err_mm / elapsed_ms are kept for existing reports
            rows.append(
                make_result_row(
                    index=i,
                    source=data_path.name,
                    fr_true=float(fr_true[i]),
                    z01_true_mm=float(z01[i]),
                    fr_est=fr_hat,
                    z01_est_mm=z01_hat,
                    runtime_s=result.elapsed_s,
                    n_evals=result.n_evals,
                    method=method,
                    fr_rel_err_pct=(fr_hat - fr_true[i]) / fr_true[i] * 100.0,
                    fr_est_minima=fr_hat_minima,
                    fr_minima_rel_err_pct=(fr_hat_minima - fr_true[i]) / fr_true[i] * 100.0 if np.isfinite(fr_hat_minima) else float("nan"),
                    z01_err_mm=z01_hat - float(z01[i]),
                    z01_est_minima_mm=z01_hat_minima,
                    score=result.score,
                    phase_rad=result.phase_rad,
                    n_minima=result.n_minima,
                    n_rings_visible=result.n_rings_visible,
                    noise_floor=result.envelope["noise_floor"],
                    power_law_exponent=result.envelope["power_law_exponent"],
                    elapsed_ms=result.elapsed_s * 1e3,
                )
            )
            if len(examples) < n_examples:
                examples.append((i, result))

    fr_err = np.array([row["fr_rel_err_pct"] for row in rows])
    z01_err = np.array([row["z01_err_mm"] for row in rows])
    minima_err = np.array([row["fr_minima_rel_err_pct"] for row in rows])
    minima_ok = np.isfinite(minima_err)
    abs_fr = np.abs(fr_err)
    fr_est_all = np.array([row["fr_est"] for row in rows])
    fr_minima_all = np.array([row["fr_est_minima"] for row in rows])
    minima_agreement = (
        float((np.abs(fr_minima_all[minima_ok] / fr_est_all[minima_ok] - 1.0) < 0.02).mean()) if minima_ok.any() else None
    )
    summary: dict[str, Any] = {
        "data": _portable_path(data_path),
        "method": method,
        "common": summarize(rows),
        "hologram_key": hologram_key,
        "num_samples": n,
        "hologram_shape": [int(s) for s in shape],
        "downsample": int(downsample),
        "search_fr_range": [fr_range[0], fr_range[1]],
        "search_z01_range_mm": list(z01_range_mm) if z01_range_mm is not None else None,
        "config": {key: value for key, value in asdict(config).items()},
        "fr_rel_err_pct": _percentiles(fr_err),
        "z01_err_mm": _percentiles(z01_err),
        "fraction_within_2pct": float((abs_fr < 2.0).mean()),
        "fraction_within_5pct": float((abs_fr < 5.0).mean()),
        "fraction_within_10pct": float((abs_fr < 10.0).mean()),
        "minima_method": {
            "num_valid": int(minima_ok.sum()),
            "fr_rel_err_pct": _percentiles(minima_err[minima_ok]) if minima_ok.any() else None,
            "agreement_within_2pct": minima_agreement,
        },
        "ms_per_hologram": {
            "mean": float(np.mean([row["elapsed_ms"] for row in rows])),
            "median": float(np.median([row["elapsed_ms"] for row in rows])),
        },
        "resolution": {
            "fr_limit_full_circle": fresnel_number_resolution_limit(int(min(shape)), config.u_max, config.window),
            "rings_resolvable_at_fr_min": max_resolvable_rings(fr_range[0] * scale, int(min(shape)), config.u_max, config.window),
            "rings_resolvable_at_fr_max": max_resolvable_rings(fr_range[1] * scale, int(min(shape)), config.u_max, config.window),
        },
        "setup": {"energy_kev": energy, "px_mm": px_mm, "z02_mm_range": [float(z02.min()), float(z02.max())]},
    }

    with (out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    figures = _write_figures(rows, examples, out_dir)
    summary["figures"] = figures
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_figures(rows: list[dict[str, Any]], examples: list[tuple[int, RingFitResult]], out_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict[str, str] = {}
    fr_true = np.array([row["fr_true"] for row in rows])
    fr_est = np.array([row["fr_est"] for row in rows])
    z01_true = np.array([row["z01_true_mm"] for row in rows])
    z01_est = np.array([row["z01_est_mm"] for row in rows])
    score = np.array([row["score"] for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax = axes[0]
    sc = ax.scatter(fr_true, fr_est, c=score, cmap="viridis", s=18, alpha=0.85)
    lim = [min(fr_true.min(), fr_est.min()) * 0.9, max(fr_true.max(), fr_est.max()) * 1.1]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("true Fr")
    ax.set_ylabel("ring-fit Fr")
    ax.set_title("Fresnel number (colour: template score)")
    fig.colorbar(sc, ax=ax, label="score")
    ax = axes[1]
    ax.scatter(z01_true, z01_est, s=18, alpha=0.7)
    lim = [min(z01_true.min(), z01_est.min()) * 0.95, max(z01_true.max(), z01_est.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("true z01 [mm]")
    ax.set_ylabel("ring-fit z01 [mm]")
    ax.set_title("z01")
    ax = axes[2]
    rel = np.array([row["fr_rel_err_pct"] for row in rows])
    ax.scatter(fr_true, rel, s=18, alpha=0.7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("true Fr")
    ax.set_ylabel("relative Fr error [%]")
    ax.set_ylim(max(-100, rel.min() - 5), min(400, rel.max() + 5))
    ax.set_title("error over Fr")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "scatter.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    figures["scatter"] = path.name

    for i, result in examples:
        row = rows[i]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax = axes[0]
        ax.semilogy(result.u, result.profile, lw=0.7, label="radial power")
        ax.semilogy(result.u, result.env, lw=1.2, label="envelope fit")
        ax.axhline(result.envelope["noise_floor"], color="k", ls="--", lw=0.8, label="noise floor")
        for n in range(1, 60):
            un = n * row["fr_true"]
            if un > result.u.max():
                break
            ax.axvline(un, color="r", alpha=0.25, lw=0.7)
        ax.set_xlabel("u = |xi|^2 [cycles^2/px^2]")
        ax.set_title(f"sample {i}: true Fr={row['fr_true']:.4e} (red: n*Fr)")
        ax.legend(fontsize=8)
        ax = axes[1]
        ax.plot(result.u, result.residual, lw=0.7, label="detrended log ratio")
        amp = 0.5 * np.std(result.residual[result.weights > 0.1]) * 2 if np.any(result.weights > 0.1) else 1.0
        ax.plot(result.u, amp * np.cos(2 * np.pi * result.u / result.fr - result.phase_rad), lw=0.9, alpha=0.8, label="fitted template")
        ax.plot(result.u, result.weights * ax.get_ylim()[1] * 0.9, lw=0.8, color="gray", alpha=0.6, label="weight")
        for um in result.minima_u[np.isfinite(result.minima_u)]:
            ax.axvline(um, color="g", alpha=0.4, lw=0.8)
        ax.set_xlim(0, min(result.u.max(), max(8 * result.fr, 0.05)))
        ax.set_xlabel("u")
        ax.set_title(f"estimate Fr={row['fr_est']:.4e} ({row['fr_rel_err_pct']:+.1f} %), phase {result.phase_rad:+.2f} rad")
        ax.legend(fontsize=8)
        ax = axes[2]
        ax.semilogx(result.fr_grid, result.scores, lw=0.9)
        ax.axvline(row["fr_true"], color="r", label="true")
        ax.axvline(result.fr, color="g", ls="--", label="estimate")
        ax.set_xlabel("candidate Fr")
        ax.set_ylabel("template score")
        ax.set_title("score over the search grid")
        ax.legend(fontsize=8)
        for ax in axes:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"example_{i:03d}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        figures[f"example_{i:03d}"] = path.name
    return figures


def _portable_path(path: Path) -> str:
    """Path relative to the project root when possible (keeps ``summary.json`` free of machine-specific prefixes)."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="CTF ring-fit baseline: estimate Fr/z01 from single holograms without learning.")
    parser.add_argument("--data", required=True, help="HDF5 file, e.g. data/processed/small/test.hdf5")
    parser.add_argument("--n", type=int, default=None, help="number of holograms (default: all)")
    parser.add_argument("--out", default="reports/ringfit", help="output directory")
    parser.add_argument("--hologram-key", default="images/hologram", help="images/hologram or images/gt_hologram")
    parser.add_argument("--z01-range", type=float, nargs=2, metavar=("MIN", "MAX"), default=None, help="search range in mm (setup range)")
    parser.add_argument("--fr-min", type=float, default=None)
    parser.add_argument("--fr-max", type=float, default=None)
    parser.add_argument("--downsample", type=int, default=1, help="average-pool factor applied before the fit (estimate rescaled by 1/s^2)")
    parser.add_argument("--window", default="tukey", choices=WINDOWS)
    parser.add_argument("--template", default="cos", choices=TEMPLATES)
    parser.add_argument("--no-snr-weights", action="store_true")
    parser.add_argument("--matched-detrend", choices=("auto", "on", "off"), default="auto", help="running-mean detrending of one candidate period")
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--examples", type=int, default=3, help="number of example figures")
    args = parser.parse_args(argv)

    fr_range = (args.fr_min, args.fr_max) if args.fr_min is not None and args.fr_max is not None else None
    summary = evaluate_file(
        args.data,
        args.out,
        num_samples=args.n,
        hologram_key=args.hologram_key,
        z01_range_mm=tuple(args.z01_range) if args.z01_range else None,
        fr_range=fr_range,
        downsample=args.downsample,
        n_examples=args.examples,
        window=args.window,
        template=args.template,
        snr_weights=not args.no_snr_weights,
        matched_detrend={"auto": None, "on": True, "off": False}[args.matched_detrend],
        n_bins=args.n_bins,
        grid_step=args.grid_step,
    )
    fr = summary["fr_rel_err_pct"]
    z = summary["z01_err_mm"]
    print(
        f"ring fit on {summary['num_samples']} holograms ({summary['hologram_key']}, search Fr "
        f"{summary['search_fr_range'][0]:.3e}..{summary['search_fr_range'][1]:.3e}): "
        f"Fr rel. err median {fr['median']:.2f} %, MAE {fr['mae']:.2f} %, p95 {fr['p95']:.2f} % | "
        f"z01 MAE {z['mae']:.2f} mm, RMSE {z['rmse']:.2f} mm, p95 {z['p95']:.2f} mm, bias {z['bias']:+.2f} mm | "
        f"within 5 %: {summary['fraction_within_5pct'] * 100:.0f} % | {summary['ms_per_hologram']['median']:.1f} ms/hologram"
    )
    print(f"report written to {_portable_path(resolve_path(args.out))}")
    return summary


if __name__ == "__main__":
    main()
