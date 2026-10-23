"""Object model of the measured holograms (Stage 6): the texture behind their CTF modulation.

The smooth fibers of Stage 5 give spectra without the CTF modulation that three of the four
measured holograms of [1] show (Laborbuch 05, 5.6). A weak-phase texture
(:func:`holofocus.phantoms.texture_field`) inside the object or as a film over the field of
view produces it. This module fits the texture of each measured hologram and builds training
sets that never saw the hologram they are tested on:

* :func:`region_weights` and :func:`region_ctf`: the CTF modulation of a region of a measured
  hologram (the object, its edge, the background around it), to find where it comes from
  (``scripts/run_stage6_study.py regions``);
* :data:`DETECTOR_FITS` and :func:`phase_per_pixel`: the detector and the bulk object of each
  measured hologram, from its background corners and the phase reconstructions of Stage 3;
* :func:`fit_changes`: a split with the detector and bulk object of one sample and a fixed
  :class:`Texture`;
* :func:`texture_grid`, :func:`refined_grid` and :func:`fit_objective`: the textures tried for
  each sample and the distance of the spectral features of a simulated set from those of the
  measured hologram (``scripts/run_stage6_study.py fit``); :data:`TEXTURE_FITS` holds the
  result, :data:`TEXTURE_FITS_LEVELS` that of the objective with the levels of the spectrum
  (Stage 10);
* :func:`loo_ranges` and :func:`loo_changes`: the training set of one sample from the other
  three (leave one out). Every property estimated from a measured hologram (detector, bulk
  object, texture) becomes a range over the other three, widened by :data:`LOO_FACTOR`; the
  setup and ``delta / beta`` (composition) stay those of the sample
  (``configs/dataset/loo-<sample>.yaml``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..dataset.config import PhantomConfig
from ..geometry import P05_DETECTOR_SHAPE
from ..measured import MEASURED_SAMPLES
from .sim2real import (
    CTF_BANDS,
    FEATURE_NAMES,
    FLAT_FIELD_CUTOFF,
    MEASURED_MODELS,
    PSF_SIGMA,
    SPECTRUM_BINS,
    SPECTRUM_F2_MAX,
    _ctf_correlation,
)

BINNING = 4
"""Binning of the measured holograms and of the matched datasets (Stages 3 to 6)."""

TEXTURE_CUTOFF = 0.01
"""Corner frequency of the texture power law in cycles per binned pixel. It lies below the band
of the spectrum CNN (``f^2 >= 0.002``, ``f >= 0.045``), so there only the slope and the RMS of
the texture matter."""


@dataclass(frozen=True)
class DetectorFit:
    """Detector of a measured hologram from the noise spectra of its four background corners
    (``scripts/run_stage5_study.py noise``, ``results/sim2real/noise.json``): medians of the
    Gaussian PSF (unbinned pixels) and of the corner frequency of the flat-field residual
    (cycles per binned pixel). The residual RMS and the photons are those of
    :data:`MEASURED_MODELS`."""

    psf_sigma: float
    flat_field_cutoff: float


DETECTOR_FITS: dict[str, DetectorFit | None] = {
    "spider_hair": DetectorFit(1.71, 0.179),
    "tooth": DetectorFit(1.64, 0.111),
    "cactus_needle": DetectorFit(1.505, 0.192),
    "magnesium_wire": None,
}
"""``None``: the corners of the Mg wire hold the object, so its fits are not those of the
detector; its texture fit uses :data:`PSF_SIGMA`, :data:`FLAT_FIELD_CUTOFF` and the residual
RMS set in :data:`MEASURED_MODELS`."""


def region_ctf(
    hologram: np.ndarray,
    weight: np.ndarray,
    fresnel_number: float,
    bins: int = SPECTRUM_BINS,
    f2_max: float = SPECTRUM_F2_MAX,
) -> dict:
    """CTF correlation of one region of a hologram over the band of the spectrum CNN and per
    band of :data:`CTF_BANDS`, as in :func:`holofocus.learned.sim2real.spectral_features`.

    ``weight`` (1 inside the region, smooth at its border) multiplies the Hann window, so a
    large irregular region keeps the frequency resolution of the whole image; the hologram
    minus its weighted mean is transformed and radially averaged on the ``f^2`` axis of
    :class:`~holofocus.learned.sim2real.RadialPower`.
    """
    x = np.asarray(hologram, dtype=np.float64)
    h, w = x.shape
    window = np.outer(np.hanning(h), np.hanning(w)) * weight
    mean = (x * window).sum() / window.sum()
    power = np.abs(np.fft.fft2((x - mean) * window)) ** 2
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.fftfreq(w)[None, :]
    index = np.floor((fx**2 + fy**2) / f2_max * bins).astype(np.int64).ravel()
    valid = index < bins
    total = np.bincount(index[valid], weights=power.ravel()[valid], minlength=bins)
    counts = np.bincount(index[valid], minlength=bins)
    log_power = np.log(np.maximum(total / np.maximum(counts, 1), 1e-30))
    f2 = (np.arange(bins) + 0.5) * f2_max / bins
    return {
        "ctf": _ctf_correlation(f2, log_power, fresnel_number, (0.002, 0.125)),
        "ctf_bands": [_ctf_correlation(f2, log_power, fresnel_number, b) for b in CTF_BANDS],
    }


def _smoothed(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing with mirrored borders, so nothing wraps around the image."""
    pad = int(math.ceil(4 * sigma))
    x = np.pad(np.asarray(mask, dtype=np.float64), pad, mode="symmetric")
    fy = np.fft.fftfreq(x.shape[0])[:, None]
    fx = np.fft.rfftfreq(x.shape[1])[None, :]
    transfer = np.exp(-2.0 * math.pi**2 * sigma**2 * (fx**2 + fy**2))
    return np.fft.irfft2(np.fft.rfft2(x) * transfer, s=x.shape)[pad:-pad, pad:-pad]


def _dilated(mask: np.ndarray, radius: int) -> np.ndarray:
    """``mask`` grown by a disk of ``radius`` pixels; nothing outside the image."""
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = (xx**2 + yy**2 <= radius**2).astype(np.float64)
    shape = (mask.shape[0] + 2 * radius, mask.shape[1] + 2 * radius)
    full = np.fft.irfft2(
        np.fft.rfft2(np.asarray(mask, dtype=np.float64), shape) * np.fft.rfft2(disk, shape), shape
    )
    return full[radius:-radius, radius:-radius] > 0.5


REGION_FRACTIONS = (0.15, 0.05)
"""Thresholds of the object core in :func:`region_weights`: the phase below this fraction of
its minimum. The lower one takes in the thin and weak parts of the object as well."""


def region_weights(phase: np.ndarray, fraction: float) -> dict[str, np.ndarray]:
    """Smooth weights (0 to 1) of three regions of a measured hologram from the phase
    reconstruction at its distance (Stage 3, negative inside the object).

    The core is where ``phase`` lies below ``fraction`` of its minimum. ``object``: the core
    grown by 8 pixels, smoothed with a Gaussian of 6 pixels; ``background``: the complement
    of the core grown by 16 pixels, smoothed with 6 pixels; ``edge``: the band from the core
    shrunk by 16 pixels to the core grown by 16 pixels, smoothed with 4 pixels.
    """
    core = np.asarray(phase) < fraction * float(np.min(phase))
    grown = _dilated(core, 16)
    shrunk = ~_dilated(~core, 16)
    return {
        "object": np.clip(_smoothed(_dilated(core, 8), 6.0), 0.0, 1.0),
        "edge": np.clip(_smoothed(grown & ~shrunk, 4.0), 0.0, 1.0),
        "background": 1.0 - np.clip(_smoothed(grown, 6.0), 0.0, 1.0),
    }


def field_of_view(sample: str) -> int:
    """Side of the binned field of view of ``sample`` in pixels (without the cutting band)."""
    return (P05_DETECTOR_SHAPE[0] - 2 * MEASURED_SAMPLES[sample].cutting_band) // BINNING


def phase_per_pixel(sample: str) -> float:
    """Bulk phase of the measured object per binned pixel of projected thickness (rad): the
    geometric mean of the phase shift of :data:`MEASURED_MODELS` over that of the fiber
    diameter, since a cylinder is as thick in projection as it is wide."""
    model = MEASURED_MODELS[sample]
    phase = math.sqrt(model.phase_shift_range[0] * model.phase_shift_range[1])
    width = math.sqrt(model.fiber_diameter_range[0] * model.fiber_diameter_range[1])
    return phase / (width * field_of_view(sample))


@dataclass(frozen=True)
class Texture:
    """Texture of the phase (:class:`holofocus.dataset.config.PhantomConfig`): RMS at the
    thickest point of the object and RMS of the film over the field of view (rad), slope of
    the power law."""

    rms: float
    background_rms: float
    slope: float

    @property
    def amplitude(self) -> float:
        """The larger of the two RMS values: the texture wherever it is."""
        return max(self.rms, self.background_rms)


def fit_changes(sample: str, texture: Texture) -> dict[str, dict]:
    """Config changes ``{section: {key: value}}`` of a split with the detector and the bulk
    object of ``sample`` and the texture ``texture``.

    Detector: :data:`DETECTOR_FITS` (Mg wire: :data:`PSF_SIGMA` and :data:`FLAT_FIELD_CUTOFF`)
    with the photons and residual RMS of :data:`MEASURED_MODELS`. Object: fibers with the
    diameters of :data:`MEASURED_MODELS`, :func:`phase_per_pixel` and ``delta / beta``.
    """
    model = MEASURED_MODELS[sample]
    detector = DETECTOR_FITS[sample]
    psf = PSF_SIGMA if detector is None else detector.psf_sigma
    cutoff = FLAT_FIELD_CUTOFF if detector is None else detector.flat_field_cutoff
    ppp = phase_per_pixel(sample)
    return {
        "simulation": {"psf_sigma_range": [psf, psf]},
        "noise": {
            "photons_range": [model.photons] * 2,
            "flat_field_range": [model.flat_field] * 2,
            "flat_field_cutoff": cutoff,
        },
        "phantoms": {
            "types": {"fiber": 1.0},
            "delta_beta": model.delta_beta,
            "fiber_diameter_range": list(model.fiber_diameter_range),
            "phase_per_pixel_range": [ppp, ppp],
            "texture_rms_range": [texture.rms] * 2,
            "background_texture_rms_range": [texture.background_rms] * 2,
            "texture_slope_range": [texture.slope] * 2,
            "texture_cutoff": TEXTURE_CUTOFF,
        },
    }


TEXTURE_GRID = {
    "rms": (0.0, 0.03, 0.1, 0.3, 1.0),
    "background_rms": (0.0, 0.01, 0.03, 0.1, 0.3),
    "slope": (2.0, 2.5, 3.0, 3.5),
}
"""Coarse grid of the texture fit: RMS values a factor of about 3 apart, and slopes."""

REFINE_FACTOR = 10**0.25
REFINE_SLOPE = 0.25


def texture_grid() -> list[Texture]:
    """The textures of :data:`TEXTURE_GRID`. Without texture the slope does not matter, so the
    untextured setting comes once, first."""
    out = [Texture(0.0, 0.0, TEXTURE_GRID["slope"][0])]
    for rms, background, slope in itertools.product(*TEXTURE_GRID.values()):
        if rms > 0 or background > 0:
            out.append(Texture(rms, background, slope))
    return out


def refined_grid(best: Texture) -> list[Texture]:
    """Textures around ``best``: each RMS times and divided by :data:`REFINE_FACTOR` (a zero
    RMS stays zero), the slope +- :data:`REFINE_SLOPE`; ``best`` itself excluded. Empty for
    ``best`` without texture."""
    if best.amplitude == 0:
        return []

    def around(value: float) -> list[float]:
        if value == 0:
            return [0.0]
        return [float(f"{value / REFINE_FACTOR:.3g}"), value, float(f"{value * REFINE_FACTOR:.3g}")]

    slopes = [best.slope - REFINE_SLOPE, best.slope, best.slope + REFINE_SLOPE]
    out = []
    for rms, background, slope in itertools.product(
        around(best.rms), around(best.background_rms), slopes
    ):
        texture = Texture(rms, background, slope)
        if texture != best:
            out.append(texture)
    return out


FIT_SCALES = {"ctf": 0.1, "upper_snr": 0.2, "slope": 2.0}
"""Scale of each feature in :func:`fit_objective`; every band of ``ctf_bands`` counts once more
with the scale of ``ctf``. The floor and ``low_excess`` are left out: the flat-field residual
and the bulk object set them more than the texture, and the residual of the corner fits puts
more power at the highest frequencies of spider hair and tooth than their holograms have
(Laborbuch 05, 5.6)."""


LEVEL_NAMES = ("level_mid", "level_upper", "floor")
"""Absolute levels of :func:`holofocus.learned.sim2real.spectral_features` (mean log power of
the mid band, the upper band of the CNN and the highest frequencies) in the objective with
levels (Stage 10)."""

LEVEL_SCALE = 0.5
"""Scale of each level difference (natural log of the power) in :func:`fit_objective` with
levels: half a unit of log power counts as much as 0.1 of CTF correlation. Without levels the
objective cannot see how much power a texture adds, and the fits of Stage 6 put 18 (cactus
needle) and 30 times (Mg wire) the measured power into the texture from ``f^2 = 0.02`` up
(Laborbuch 10)."""


def median_features(features: Sequence[Mapping]) -> dict:
    """Median of every feature of :func:`holofocus.learned.sim2real.spectral_features` over a
    set of holograms, ``ctf_bands`` band by band; the levels only if every set member has
    them (records of Stage 6 do not)."""
    out = {name: float(np.nanmedian([f[name] for f in features])) for name in FEATURE_NAMES}
    out["ctf_bands"] = [
        float(np.nanmedian([f["ctf_bands"][k] for f in features])) for k in range(len(CTF_BANDS))
    ]
    for name in LEVEL_NAMES:
        if name not in out and all(name in f for f in features):
            out[name] = float(np.nanmedian([f[name] for f in features]))
    return out


def fit_objective(median: Mapping, measured: Mapping, levels: bool = False) -> float:
    """RMS of the scaled differences between the median features of a simulated set and the
    features of the measured hologram (:data:`FIT_SCALES`); ``inf`` if one is undefined.

    ``levels`` adds the differences of :data:`LEVEL_NAMES` over :data:`LEVEL_SCALE` (Stage 10);
    without them the objective is that of Stage 6.
    """
    scale = FIT_SCALES["ctf"]
    terms = [(median["ctf"] - measured["ctf"]) / scale]
    terms += [
        (a - b) / scale for a, b in zip(median["ctf_bands"], measured["ctf_bands"], strict=True)
    ]
    terms += [(median[k] - measured[k]) / FIT_SCALES[k] for k in ("upper_snr", "slope")]
    if levels:
        terms += [(median[k] - measured[k]) / LEVEL_SCALE for k in LEVEL_NAMES]
    value = float(np.sqrt(np.mean(np.square(terms))))
    return value if math.isfinite(value) else math.inf


TEXTURE_FITS: dict[str, Texture] = {
    "spider_hair": Texture(0.0, 0.1, 3.0),
    "tooth": Texture(0.0, 0.0, 2.0),
    "cactus_needle": Texture(0.3, 0.03, 1.75),
    "magnesium_wire": Texture(0.3, 0.533, 1.75),
}
"""The texture of each measured hologram with the smallest :func:`fit_objective` over
16 test samples (``results/objectmodel/fit-<sample>.json``): a film around the spider hair,
none on the tooth, mostly inside the cactus needle, inside and around the Mg wire. The fit sees
the whole hologram only; the Mg wire fills most of the field of view, so there texture and
film are nearly interchangeable."""

TEXTURE_FITS_LEVELS: dict[str, Texture] = {
    "spider_hair": Texture(0.0, 0.1, 3.0),
    "tooth": Texture(0.0, 0.0, 2.0),
    "cactus_needle": Texture(0.0, 0.1, 2.75),
    "magnesium_wire": Texture(0.3, 0.0533, 2.0),
}
"""The same fit with the levels of the spectrum in the objective (``levels=True``, Stage 10,
``results/objectmodel/levels/fit-<sample>.json``). The spider hair and the tooth keep their
fits; the cactus needle has a film instead of a texture inside, the Mg wire a tenth of the film
and a steeper slope: less power at high frequencies on both."""


LOO_FACTOR = 2.0
"""A range of a leave-one-out set reaches from the smallest value of the other three samples
divided by this factor to the largest times it."""

TEXTURE_FLOOR = 0.01
"""Texture RMS (rad) up to which a fitted texture leaves the spectrum as it is without texture:
such a sample counts as untextured. Every texture range of the leave-one-out sets starts at
this value divided by :data:`LOO_FACTOR`, since a sample may have no texture."""

SLOPE_MARGIN = 0.5
"""Added on both sides of the range of the fitted slopes of the textured samples."""

PSF_MARGIN = 0.3
"""Unbinned pixels added on both sides of the range of the PSF fits."""


@dataclass(frozen=True)
class LooRanges:
    """Ranges of the training set of ``sample`` from the measured holograms ``others``
    (:func:`loo_ranges`); the keys of :class:`holofocus.dataset.config.DatasetConfig`."""

    sample: str
    others: tuple[str, ...]
    psf_sigma: tuple[float, float]
    photons: tuple[float, float]
    flat_field: tuple[float, float]
    flat_field_cutoff: float
    phase_per_pixel: tuple[float, float]
    fiber_diameter: tuple[float, float]
    texture_rms: tuple[float, float]
    texture_slope: tuple[float, float]


def _round(value: float) -> float:
    return float(f"{value:.3g}")


def _widened(values: Sequence[float], cap: float = math.inf) -> tuple[float, float]:
    return (_round(min(values) / LOO_FACTOR), _round(min(max(values) * LOO_FACTOR, cap)))


def loo_ranges(sample: str, fits: Mapping[str, Texture] | None = None) -> LooRanges:
    """What the three measured holograms other than ``sample`` say, as ranges for a training
    set of ``sample`` (leave one out); values rounded to three significant digits.

    Detector: the PSF over the range of :data:`DETECTOR_FITS` +- :data:`PSF_MARGIN`, the corner
    frequency of the residual their median, the photons and residual RMS of
    :data:`MEASURED_MODELS` widened by :data:`LOO_FACTOR` (the Mg wire, whose corners hold the
    object, only with its photons). Bulk object: :func:`phase_per_pixel` and the fiber
    diameters widened, the diameters at most the field of view. Texture (``fits``, default
    :data:`TEXTURE_FITS`): it sits inside the object in one sample and around it in another,
    so the object texture and the film both range from :data:`TEXTURE_FLOOR` divided by
    :data:`LOO_FACTOR` to the largest amplitude of the others times it, and the slope over the
    fitted slopes of the textured ones +- :data:`SLOPE_MARGIN`.
    """
    if sample not in MEASURED_SAMPLES:
        raise ValueError(f"unknown sample {sample!r}; choose from {list(MEASURED_SAMPLES)}")
    fits = TEXTURE_FITS if fits is None else fits
    others = tuple(s for s in MEASURED_SAMPLES if s != sample)
    with_corners = [s for s in others if DETECTOR_FITS[s] is not None]
    psf = [DETECTOR_FITS[s].psf_sigma for s in with_corners]
    textured = [fits[s] for s in others if fits[s].amplitude > TEXTURE_FLOOR]
    slopes = [t.slope for t in textured] or [PhantomConfig().texture_slope_range[0]]
    amplitude = max([TEXTURE_FLOOR] + [t.amplitude for t in textured])
    diameters = [d for s in others for d in MEASURED_MODELS[s].fiber_diameter_range]
    return LooRanges(
        sample=sample,
        others=others,
        psf_sigma=(_round(min(psf) - PSF_MARGIN), _round(max(psf) + PSF_MARGIN)),
        photons=_widened([MEASURED_MODELS[s].photons for s in others]),
        flat_field=_widened([MEASURED_MODELS[s].flat_field for s in with_corners]),
        flat_field_cutoff=_round(
            float(np.median([DETECTOR_FITS[s].flat_field_cutoff for s in with_corners]))
        ),
        phase_per_pixel=_widened([phase_per_pixel(s) for s in others]),
        fiber_diameter=_widened(diameters, cap=1.0),
        texture_rms=(_round(TEXTURE_FLOOR / LOO_FACTOR), _round(amplitude * LOO_FACTOR)),
        texture_slope=(min(slopes) - SLOPE_MARGIN, max(slopes) + SLOPE_MARGIN),
    )


def loo_changes(sample: str, fits: Mapping[str, Texture] | None = None) -> dict[str, dict]:
    """Config changes of the leave-one-out training set of ``sample`` (:func:`loo_ranges`):
    fibers with the ``delta / beta`` of its composition, the same range for the object texture
    and the film."""
    r = loo_ranges(sample, fits)
    return {
        "simulation": {"psf_sigma_range": list(r.psf_sigma)},
        "noise": {
            "photons_range": list(r.photons),
            "flat_field_range": list(r.flat_field),
            "flat_field_cutoff": r.flat_field_cutoff,
        },
        "phantoms": {
            "types": {"fiber": 1.0},
            "delta_beta": MEASURED_MODELS[sample].delta_beta,
            "fiber_diameter_range": list(r.fiber_diameter),
            "phase_per_pixel_range": list(r.phase_per_pixel),
            "texture_rms_range": list(r.texture_rms),
            "background_texture_rms_range": list(r.texture_rms),
            "texture_slope_range": list(r.texture_slope),
            "texture_cutoff": TEXTURE_CUTOFF,
        },
    }
