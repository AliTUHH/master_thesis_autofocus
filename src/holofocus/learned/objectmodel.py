"""Object model of the measured holograms (Stage 6): the texture behind their CTF modulation.

The smooth fibers of Stage 5 give spectra without the CTF modulation that three of the four
measured holograms of [1] show (Laborbuch 05, 5.6). A weak-phase texture
(:func:`holofocus.phantoms.texture_field`) inside the object or as a film over the field of
view produces it. This module fits the texture of each measured hologram:

* :data:`DETECTOR_FITS` and :func:`phase_per_pixel`: the detector and the bulk object of each
  measured hologram, from its background corners and the phase reconstructions of Stage 3;
* :func:`fit_changes`: a split with the detector and bulk object of one sample and a fixed
  :class:`Texture`;
* :func:`texture_grid`, :func:`refined_grid` and :func:`fit_objective`: the textures tried for
  each sample and the distance of the spectral features of a simulated set from those of the
  measured hologram (``scripts/run_stage6_study.py fit``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..geometry import P05_DETECTOR_SHAPE
from ..measured import MEASURED_SAMPLES
from .sim2real import (
    CTF_BANDS,
    FEATURE_NAMES,
    FLAT_FIELD_CUTOFF,
    MEASURED_MODELS,
    PSF_SIGMA,
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
with the scale of ``ctf``. The floor and ``low_excess`` are left out: they follow the
flat-field residual and the bulk object, which the texture does not change."""


def median_features(features: Sequence[Mapping]) -> dict:
    """Median of every feature of :func:`holofocus.learned.sim2real.spectral_features` over a
    set of holograms, ``ctf_bands`` band by band."""
    out = {name: float(np.nanmedian([f[name] for f in features])) for name in FEATURE_NAMES}
    out["ctf_bands"] = [
        float(np.nanmedian([f["ctf_bands"][k] for f in features])) for k in range(len(CTF_BANDS))
    ]
    return out


def fit_objective(median: Mapping, measured: Mapping) -> float:
    """RMS of the scaled differences between the median features of a simulated set and the
    features of the measured hologram (:data:`FIT_SCALES`); ``inf`` if one is undefined."""
    scale = FIT_SCALES["ctf"]
    terms = [(median["ctf"] - measured["ctf"]) / scale]
    terms += [
        (a - b) / scale for a, b in zip(median["ctf_bands"], measured["ctf_bands"], strict=True)
    ]
    terms += [(median[k] - measured[k]) / FIT_SCALES[k] for k in ("upper_snr", "slope")]
    value = float(np.sqrt(np.mean(np.square(terms))))
    return value if math.isfinite(value) else math.inf
