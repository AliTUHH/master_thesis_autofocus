"""Gap between simulated and measured holograms for the learned autofocus (Stage 5).

* :data:`MEASURED_MODELS`: what the four measured holograms of the Zenodo record of [1] show
  about the assumptions of the matched datasets (Laborbuch 05): the white part of the noise
  and the correlated background on the 4x grid, the detector blur and the object;
* :func:`perturbation_changes` and :func:`perturbed_config`: a split config with one
  assumption changed (:data:`PERTURBATIONS`). The seed stays, so a perturbed test split pairs
  sample by sample with the original: same ``z01``, photon count (except ``noise``) and, except
  for the object perturbations, the same object. Detector effects draw from their own stream;
* :func:`paired_shifts`: estimate on the perturbed sample minus the estimate on the original;
* :func:`spectral_features`: numbers of the radial power spectrum (the input of the spectrum
  CNN) that place a measured hologram in the distribution of a simulated split;
* :func:`measured_estimate`: the learned estimate of a measured hologram with its intervals,
  the distances to reference values, the spread over the phases of the binning grid and the
  time per estimate.

Lengths in metres unless a key ends in ``_um`` or ``_mm``.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ..dataset.config import DatasetConfig
from ..measured import MeasuredSample, bin_mean
from .inference import LearnedAutofocus
from .metrics import LEVELS, bootstrap_ci, interval_quantile, point_metrics

PSF_SIGMA = 1.6
"""Detector PSF in unbinned pixels: Gaussian fit of the noise power spectrum of the background
corners at full resolution, ``A exp(-4 pi^2 sigma^2 f^2) + B`` (1.3-1.8 px for all samples)."""

FLAT_FIELD_CUTOFF = 0.18
"""Corner frequency of the correlated residual in cycles per binned pixel (fit of the corner
spectra on the 4x grid: 0.14-0.26 for spider hair and cactus needle)."""


@dataclass(frozen=True)
class MeasuredModel:
    """Properties of one measured hologram that the matched datasets do not reproduce.

    ``photons``: white part of the noise per binned pixel (``16 N_1`` of Laborbuch 02, 5.4);
    ``flat_field``: RMS of the correlated residual on the 4x grid (fit of the corner spectra);
    ``phase_shift_range`` (rad), ``fiber_diameter_range`` (relative to the field of view) and
    ``delta_beta``: the object as a fiber, from the phase reconstructions of Stage 3 and the
    composition of the sample.
    """

    photons: float
    flat_field: float
    phase_shift_range: tuple[float, float]
    fiber_diameter_range: tuple[float, float]
    delta_beta: float


MEASURED_MODELS: dict[str, MeasuredModel] = {
    "spider_hair": MeasuredModel(6.8e4, 0.026, (1.0, 3.0), (0.1, 0.25), 600.0),
    "tooth": MeasuredModel(3.5e4, 0.043, (5.0, 12.0), (0.4, 0.7), 65.0),
    "cactus_needle": MeasuredModel(3.9e4, 0.027, (2.0, 6.0), (0.08, 0.25), 1300.0),
    "magnesium_wire": MeasuredModel(3.1e4, 0.03, (8.0, 20.0), (0.55, 0.85), 120.0),
}
"""Spider hair, cactus needle, tooth: corner fits; Mg wire: object in the corners, residual set
to 3 %. Objects: phase and width of the Stage 3 reconstructions; ``delta / beta`` estimated from
the composition (protein, hydroxyapatite, cellulose, Mg) at the energy of the sample."""

PERTURBATIONS: dict[str, str] = {
    "psf": f"Gaussian detector PSF, sigma {PSF_SIGMA} detector pixels",
    "flat-field": "correlated flat-field residual of the measured hologram",
    "noise": "noise of the measured hologram: white part (photons fixed) and correlated residual",
    "fiber": "fibers crossing the field of view instead of compact objects",
    "object": "object of the measured hologram: fiber with its phase shift, width and delta/beta",
    "combined": "psf, noise and object together",
}


def perturbation_changes(name: str, sample: str) -> dict[str, dict]:
    """Config changes ``{section: {key: value}}`` of perturbation ``name`` for ``sample``."""
    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {name!r}; choose from {list(PERTURBATIONS)}")
    model = MEASURED_MODELS[sample]
    flat = {"flat_field_range": [model.flat_field] * 2, "flat_field_cutoff": FLAT_FIELD_CUTOFF}
    obj = {
        "types": {"fiber": 1.0},
        "phase_shift_range": list(model.phase_shift_range),
        "fiber_diameter_range": list(model.fiber_diameter_range),
        "delta_beta": model.delta_beta,
    }
    table = {
        "psf": {"simulation": {"psf_sigma_range": [PSF_SIGMA] * 2}},
        "flat-field": {"noise": flat},
        "noise": {"noise": {**flat, "photons_range": [model.photons] * 2}},
        "fiber": {"phantoms": {"types": {"fiber": 1.0}}},
        "object": {"phantoms": obj},
    }
    table["combined"] = {
        "simulation": table["psf"]["simulation"],
        "noise": table["noise"]["noise"],
        "phantoms": obj,
    }
    return table[name]


def perturbed_config(
    config: DatasetConfig, changes: Mapping[str, Mapping], name: str
) -> DatasetConfig:
    """``config`` with the keys of ``changes`` replaced, renamed to ``name``, same seed."""
    data = config.to_dict()
    for section, values in changes.items():
        if section not in data or not isinstance(data[section], dict):
            raise ValueError(f"unknown config section {section!r}")
        data[section] = {**data[section], **values}
    data["name"] = name
    return DatasetConfig.from_dict(data)


def paired_shifts(reference: Sequence[dict], perturbed: Sequence[dict], seed: int = 0) -> dict:
    """Shift of the estimates (perturbed minus reference, um) over the shared sample indices,
    and the change of the error metrics; rows as in ``predictions.csv``."""
    ref = {int(r["index"]): r for r in reference}
    pairs = [(ref[int(r["index"])], r) for r in perturbed if int(r["index"]) in ref]
    if not pairs:
        raise ValueError("the perturbed and the reference predictions share no sample")
    shift = np.array([(p["z01_est"] - r["z01_est"]) * 1e6 for r, p in pairs])
    err_ref = np.array([r["error_um"] for r, _ in pairs])
    err_pert = np.array([p["error_um"] for _, p in pairs])
    same_z01 = max(abs(p["z01_true"] - r["z01_true"]) for r, p in pairs) < 1e-12
    return {
        "n": len(pairs),
        "same_z01": bool(same_z01),
        "shift_um": {
            "mean": float(shift.mean()),
            "mean_ci": list(bootstrap_ci(shift, np.mean, seed=seed)),
            "median": float(np.median(shift)),
            "std": float(shift.std(ddof=1)) if len(shift) > 1 else 0.0,
            "mean_abs": float(np.abs(shift).mean()),
            "p90_abs": float(np.percentile(np.abs(shift), 90)),
        },
        "reference_error_um": point_metrics(err_ref),
        "perturbed_error_um": point_metrics(err_pert),
        "mae_change_um": float(np.abs(err_pert).mean() - np.abs(err_ref).mean()),
    }


SPECTRUM_BINS = 2048
SPECTRUM_F2_MAX = 0.25
FEATURE_BANDS = {
    "low": (0.0005, 0.005),
    "mid": (0.02, 0.04),
    "upper": (0.0625, 0.125),
    "floor": (0.2, 0.25),
}
CTF_BANDS = ((0.002, 0.03), (0.03, 0.0625), (0.0625, 0.125), (0.125, 0.25))
"""Bands in ``f^2`` (cycles per binned pixel squared); ``(0.002, 0.125)`` is the band of the
spectrum CNN without the lowest frequencies."""


class RadialPower:
    """Log of the radially averaged power on a uniform ``f^2`` axis, without standardizing.

    Same window and mean subtraction as the ``spectrum`` representation, power divided by the
    sum of the squared window, so white noise of variance ``v`` gives ``log v``.
    """

    def __init__(
        self, shape: tuple[int, int], bins: int = SPECTRUM_BINS, f2_max: float = SPECTRUM_F2_MAX
    ) -> None:
        h, w = shape
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        index = np.floor((fx**2 + fy**2) / f2_max * bins).astype(np.int64).ravel()
        self.valid = index < bins
        self.bin_of = index[self.valid]
        self.counts = np.bincount(self.bin_of, minlength=bins).astype(float)
        if self.counts.min() == 0:
            raise ValueError(f"{bins} bins up to f^2 = {f2_max} leave bins empty on {shape}")
        self.window = np.outer(np.hanning(h), np.hanning(w))
        self.norm = float((self.window**2).sum())
        self.f2 = (np.arange(bins) + 0.5) * f2_max / bins
        self.shape = (h, w)

    def __call__(self, hologram: np.ndarray) -> np.ndarray:
        x = np.asarray(hologram, dtype=np.float64)
        if x.shape != self.shape:
            raise ValueError(f"expected a hologram of shape {self.shape}, got {x.shape}")
        mean = (x * self.window).sum() / self.window.sum()
        spectrum = np.fft.fft2((x - mean) * self.window)
        power = (spectrum.real**2 + spectrum.imag**2).ravel()[self.valid] / self.norm
        profile = np.bincount(self.bin_of, weights=power, minlength=len(self.counts))
        return np.log(np.maximum(profile / self.counts, 1e-30))


def _band(f2: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (f2 >= band[0]) & (f2 < band[1])


def _ctf_correlation(
    f2: np.ndarray, log_power: np.ndarray, fresnel_number: float, band: tuple[float, float]
) -> float:
    """Correlation of the log power, minus its running mean over one CTF period, with
    ``sin^2(pi f^2 / Fr)`` in ``band``."""
    step = f2[1] - f2[0]
    period = max(3, int(round(fresnel_number / step)))
    trend = np.convolve(log_power, np.ones(period) / period, mode="same")
    inner = _band(f2, band)
    inner[: period // 2 + 1] = False
    inner[len(f2) - period // 2 - 1 :] = False
    if inner.sum() < 3:
        return float("nan")
    detrended = (log_power - trend)[inner]
    ctf = np.sin(np.pi * f2[inner] / fresnel_number) ** 2
    if detrended.std() == 0:
        return float("nan")
    return float(np.corrcoef(detrended, ctf)[0, 1])


def spectral_features(radial: RadialPower, hologram: np.ndarray, fresnel_number: float) -> dict:
    """Levels and CTF modulation of the radial power spectrum of one hologram.

    ``floor``: mean log power at the highest frequencies; ``upper_snr``: the upper band of the
    CNN above the floor; ``low_excess``: the lowest frequencies above the mid band;
    ``slope``: d log P / d f^2 over the CNN band (blur makes it steeper); ``ctf``: CTF
    correlation over the CNN band and per band of :data:`CTF_BANDS`. Logs are natural.
    """
    p = radial(hologram)
    f2 = radial.f2
    level = {k: float(p[_band(f2, b)].mean()) for k, b in FEATURE_BANDS.items()}
    fit = _band(f2, (0.03, 0.125))
    slope = float(np.polyfit(f2[fit], p[fit], 1)[0])
    return {
        "floor": level["floor"],
        "upper_snr": level["upper"] - level["floor"],
        "low_excess": level["low"] - level["mid"],
        "slope": slope,
        "ctf": _ctf_correlation(f2, p, fresnel_number, (0.002, 0.125)),
        "ctf_bands": [_ctf_correlation(f2, p, fresnel_number, b) for b in CTF_BANDS],
    }


FEATURE_NAMES = ("floor", "upper_snr", "low_excess", "slope", "ctf")


def feature_distribution(features: Sequence[dict]) -> dict:
    """Median and 5/95 % quantiles of each scalar feature over a set of holograms."""
    out = {}
    for name in FEATURE_NAMES:
        values = np.array([f[name] for f in features], dtype=float)
        values = values[np.isfinite(values)]
        out[name] = {
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    return out


def percentile_rank(values: Sequence[float], value: float) -> float:
    """Percentage of ``values`` below ``value`` (ties count half)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(100.0 * (np.mean(v < value) + 0.5 * np.mean(v == value)))


def binning_phases(image: np.ndarray, binning: int) -> list[np.ndarray]:
    """The image binned by the mean on every shift ``(dy, dx)`` of the block grid (wrapped)."""
    return [
        bin_mean(np.roll(image, (-dy, -dx), axis=(0, 1)), binning)
        for dy in range(binning)
        for dx in range(binning)
    ]


def measured_estimate(
    autofocus: LearnedAutofocus,
    hologram: torch.Tensor,
    sample: MeasuredSample,
    references: Mapping[str, float],
    repeats: int = 20,
    phases: Sequence[np.ndarray] | None = None,
) -> dict:
    """The learned estimate of ``z01`` for a measured hologram (binned like the training data).

    ``references`` maps names to values of ``z01``; for each the distance (um), the z-score and
    whether it lies in the 68 and 95 % intervals are reported. ``phases``: the hologram binned
    on the other shifts of the block grid (:func:`binning_phases`), for the spread of the
    estimate under a resampling of the same measurement.
    """
    if tuple(hologram.shape) != autofocus.fov_shape:
        raise ValueError(
            f"hologram of shape {tuple(hologram.shape)}, the model expects {autofocus.fov_shape}"
        )
    autofocus.estimate(hologram)
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        dz01, sigma, sigma_raw = (float(v[0]) for v in autofocus.estimate(hologram))
        seconds.append(time.perf_counter() - start)
    z01 = sample.z01_guess + dz01
    lower, upper = sample.simplex
    intervals = {}
    for level in LEVELS:
        half = interval_quantile(level) * sigma
        intervals[f"{round(100 * level)}"] = {
            "lower": z01 - half,
            "upper": z01 + half,
            "width_um": 2e6 * half,
        }
    record = {
        "z01_guess": sample.z01_guess,
        "dz01_est": dz01,
        "z01_est": z01,
        "sigma": sigma,
        "sigma_raw": sigma_raw,
        "inside_omega": bool(lower <= z01 <= upper),
        "intervals": intervals,
        "references": {},
        "seconds_per_estimate": {
            "median": float(np.median(seconds)),
            "min": float(np.min(seconds)),
            "repeats": repeats,
        },
    }
    for name, value in references.items():
        distance = z01 - value
        record["references"][name] = {
            "z01": value,
            "distance_um": distance * 1e6,
            "z_score": distance / sigma,
            "inside": {
                key: bool(i["lower"] <= value <= i["upper"]) for key, i in intervals.items()
            },
        }
    if phases is not None:
        batch = torch.from_numpy(np.stack([np.asarray(p, dtype=np.float32) for p in phases]))
        dz, sig, _ = autofocus.estimate(batch)
        z = sample.z01_guess + dz
        record["binning_phases"] = {
            "n": len(phases),
            "z01_est": [float(v) for v in z],
            "sigma": [float(v) for v in sig],
            "std_um": float(np.std(z, ddof=1) * 1e6) if len(z) > 1 else 0.0,
            "range_um": float((z.max() - z.min()) * 1e6),
        }
    return record
