"""YAML configuration of the synthetic dataset (one file describes all splits).

Unknown keys are rejected so that typos do not silently fall back to defaults. Lengths are in
metres, energies in keV, phases in radians. See ``configs/dataset/*.yaml`` for examples.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..geometry import P05_DETECTOR_SHAPE, P05_PIXEL_SIZE, ConeBeamGeometry
from ..noise import NOISE_MODELS
from ..phantoms import PHANTOM_TYPES
from ..propagation import next_fast_len
from ..yamlconfig import build, plain


@dataclass(frozen=True)
class GeometryConfig:
    """P05-like cone beam ([1] Sec. 5.1, [2] Table 2); ``z02`` is known ([2] Sec. 2)."""

    energy_kev: float = 11.0
    z02: float = 20.0
    z01_guess: float = 0.100
    z01_guess_range: tuple[float, float] | None = None
    detector_pixel_size: float = P05_PIXEL_SIZE
    detector_shape: tuple[int, int] = P05_DETECTOR_SHAPE
    binning: int = 4

    @property
    def fov_shape(self) -> tuple[int, int]:
        return (self.detector_shape[0] // self.binning, self.detector_shape[1] // self.binning)

    @property
    def pixel_size(self) -> float:
        return self.detector_pixel_size * self.binning

    def at(self, z01: float) -> ConeBeamGeometry:
        """Binned-detector geometry for a sample at ``z01``."""
        return ConeBeamGeometry(self.energy_kev, z01, self.z02, self.pixel_size)

    @property
    def guess_bounds(self) -> tuple[float, float]:
        if self.z01_guess_range is None:
            return (self.z01_guess, self.z01_guess)
        return tuple(self.z01_guess_range)


@dataclass(frozen=True)
class SamplingConfig:
    """``dz01 = z01_true - z01_guess`` is drawn uniformly from ``dz01_range`` ([2] Eq. (17))."""

    dz01_range: tuple[float, float] = (-5.0e-3, 5.0e-3)


@dataclass(frozen=True)
class SimulationConfig:
    """Numerics of the forward model.

    ``padding_margin``: free space around the field of view in units of ``1/Fr`` (at the
    smallest ``Fr`` of the dataset). ``oversample``: simulate on an ``s`` times finer grid and
    average ``s x s`` blocks of the intensity (pixel integration). ``a0_range``: uniform range
    of the illumination amplitude ``A0`` of [1] Eq. (4). ``psf_sigma_range``: uniform range of
    the standard deviation of a Gaussian detector PSF in unbinned detector pixels (0: none).
    """

    padding_margin: float = 1.0
    oversample: int = 1
    a0_range: tuple[float, float] = (1.0, 1.0)
    psf_sigma_range: tuple[float, float] = (0.0, 0.0)
    device: str = "auto"
    precision: str = "single"


@dataclass(frozen=True)
class PhantomConfig:
    """Single-material phantoms in the spirit of [2] Sec. 3.5 (fixed ``delta/beta``).

    ``size_range`` and ``margin`` apply to the compact types, ``fiber_diameter_range``
    (relative to the field of view, log-uniform) to fibers, which cross the field of view.
    ``phase_shift_range`` (uniform) is the largest phase shift in the field of view. With
    ``phase_per_pixel_range`` (fibers only, log-uniform) the phase follows the projected
    thickness instead: that many radians per binned pixel of path, so thin fibers shift less.

    Texture (Stage 6, :func:`holofocus.phantoms.texture_field`): one Gaussian random field per
    sample with the power ``(1 + (f / texture_cutoff)^2)^(-slope / 2)`` (``f`` in cycles per
    binned pixel, ``slope`` uniform in ``texture_slope_range``) is added to the phase, with
    the RMS ``texture_rms_range`` (rad, log-uniform) at the thickest point of the object and
    proportional to the square root of the thickness, as for a random density of the material
    integrated along the beam, plus ``background_texture_rms_range`` (rad, log-uniform) over
    the whole field, as of a film around the object. ``[0, 0]`` switches a part off.
    """

    types: dict[str, float] = field(
        default_factory=lambda: {"triangle": 1.0, "cell": 1.0, "ball": 1.0, "mixed": 1.0}
    )
    delta_beta: float = 200.0
    phase_shift_range: tuple[float, float] = (0.5, 2.0)
    size_range: tuple[float, float] = (0.15, 0.6)
    margin: float = 0.05
    supersample: int = 4
    fiber_diameter_range: tuple[float, float] = (0.03, 1.0)
    phase_per_pixel_range: tuple[float, float] | None = None
    texture_rms_range: tuple[float, float] = (0.0, 0.0)
    background_texture_rms_range: tuple[float, float] = (0.0, 0.0)
    texture_slope_range: tuple[float, float] = (3.0, 3.0)
    texture_cutoff: float = 0.01

    @property
    def textured(self) -> bool:
        return self.texture_rms_range[1] > 0 or self.background_texture_rms_range[1] > 0


@dataclass(frozen=True)
class NoiseConfig:
    """Poisson (+ Gaussian read-out) noise, see :mod:`holofocus.noise`.

    ``photons_range``: mean flat-field counts per binned pixel, drawn log-uniformly per sample.
    ``flat_field_range``: uniform range of the RMS of a correlated, multiplicative flat-field
    residual with the corner frequency ``flat_field_cutoff`` in cycles per binned pixel
    (:func:`holofocus.noise.flat_field_residual`; 0: none).
    """

    model: str = "poisson_gaussian"
    photons_range: tuple[float, float] = (1.0e3, 1.0e5)
    read_noise: float = 2.0
    flat_field_range: tuple[float, float] = (0.0, 0.0)
    flat_field_cutoff: float = 0.15


@dataclass(frozen=True)
class StorageConfig:
    """HDF5 layout options: ground-truth phase and noise-free hologram next to the noisy one."""

    store_phase: bool = True
    store_clean: bool = False
    compression: str | None = "gzip"
    compression_level: int = 4


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "dataset"
    seed: int = 0
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    phantoms: PhantomConfig = field(default_factory=PhantomConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    splits: dict[str, int] = field(default_factory=lambda: {"train": 1000, "val": 100, "test": 100})

    def __post_init__(self) -> None:
        g, s = self.geometry, self.sampling
        if self.geometry.binning < 1 or any(n % g.binning for n in g.detector_shape):
            raise ValueError(f"binning {g.binning} must divide detector shape {g.detector_shape}")
        if not s.dz01_range[0] < s.dz01_range[1]:
            raise ValueError(f"dz01_range must be increasing, got {s.dz01_range}")
        lo, hi = g.guess_bounds
        if lo > hi:
            raise ValueError(f"z01_guess_range must be increasing, got {g.z01_guess_range}")
        if not (0 < lo + s.dz01_range[0] and hi + s.dz01_range[1] < g.z02):
            raise ValueError("z01 range must lie between the focus and the detector")
        sim, ph = self.simulation, self.phantoms
        if sim.oversample < 1 or ph.supersample < 1:
            raise ValueError("oversample and supersample must be >= 1")
        if sim.padding_margin < 0:
            raise ValueError("padding_margin must be >= 0")
        if not 0 < sim.a0_range[0] <= sim.a0_range[1]:
            raise ValueError("a0_range must be positive and increasing")
        if not 0 <= sim.psf_sigma_range[0] <= sim.psf_sigma_range[1]:
            raise ValueError("psf_sigma_range must be >= 0 and increasing")
        if sim.precision not in ("single", "double"):
            raise ValueError(f"precision must be 'single' or 'double', got {sim.precision!r}")
        if not 0 < ph.phase_shift_range[0] <= ph.phase_shift_range[1]:
            raise ValueError("phase_shift_range must be positive and increasing")
        if ph.delta_beta <= 0:
            raise ValueError("delta_beta must be positive")
        if not 0 < ph.size_range[0] <= ph.size_range[1] <= 1 - 2 * ph.margin:
            raise ValueError("need 0 < size_range[0] <= size_range[1] <= 1 - 2 * margin")
        unknown = sorted(set(ph.types) - set(PHANTOM_TYPES))
        if unknown:
            raise ValueError(f"unknown phantom types {unknown}, choose from {PHANTOM_TYPES}")
        weights = list(ph.types.values())
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("phantom type weights must be >= 0 with a positive sum")
        if not 0 < ph.fiber_diameter_range[0] <= ph.fiber_diameter_range[1]:
            raise ValueError("fiber_diameter_range must be positive and increasing")
        if ph.phase_per_pixel_range is not None:
            if not 0 < ph.phase_per_pixel_range[0] <= ph.phase_per_pixel_range[1]:
                raise ValueError("phase_per_pixel_range must be positive and increasing")
            if any(w > 0 for kind, w in ph.types.items() if kind != "fiber"):
                raise ValueError("phase_per_pixel_range needs fibers only, types {fiber: 1}")
        for key in ("texture_rms_range", "background_texture_rms_range"):
            lo, hi = getattr(ph, key)
            if not (lo == hi == 0 or 0 < lo <= hi):
                raise ValueError(f"{key} must be [0, 0] (off) or positive and increasing")
        if not 0 <= ph.texture_slope_range[0] <= ph.texture_slope_range[1]:
            raise ValueError("texture_slope_range must be >= 0 and increasing")
        if ph.texture_cutoff <= 0:
            raise ValueError("texture_cutoff must be positive")
        if self.noise.model not in NOISE_MODELS:
            raise ValueError(f"unknown noise model {self.noise.model!r}")
        if not 0 < self.noise.photons_range[0] <= self.noise.photons_range[1]:
            raise ValueError("photons_range must be positive and increasing")
        if self.noise.read_noise < 0:
            raise ValueError("read_noise must be >= 0")
        if not 0 <= self.noise.flat_field_range[0] <= self.noise.flat_field_range[1]:
            raise ValueError("flat_field_range must be >= 0 and increasing")
        if self.noise.flat_field_cutoff <= 0:
            raise ValueError("flat_field_cutoff must be positive")
        if self.storage.compression not in (None, "gzip", "lzf"):
            raise ValueError("compression must be null, 'gzip' or 'lzf'")
        if any(n < 0 for n in self.splits.values()):
            raise ValueError("split sizes must be >= 0")

    @property
    def z01_bounds(self) -> tuple[float, float]:
        lo, hi = self.geometry.guess_bounds
        return lo + self.sampling.dz01_range[0], hi + self.sampling.dz01_range[1]

    @property
    def simulation_shape(self) -> tuple[int, int]:
        """Free-space array for the forward simulation (on the oversampled grid).

        ``Fr`` grows with ``z01``, so the smallest ``Fr`` (largest kernel) occurs at the lower
        ``z01`` bound; the array then also satisfies ``N >= 1/Fr`` ([1] Sec. 3.2.2).
        """
        s = self.simulation.oversample
        fr_min = self.geometry.at(self.z01_bounds[0]).fresnel_number / s**2
        margin = math.ceil(self.simulation.padding_margin / fr_min)
        return tuple(
            next_fast_len(max(n * s + margin, math.ceil(1 / fr_min)))
            for n in self.geometry.fov_shape
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        return build(cls, data, "config", _SECTIONS)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DatasetConfig:
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return plain(dataclasses.asdict(self))

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)


_SECTIONS = {
    "geometry": GeometryConfig,
    "sampling": SamplingConfig,
    "simulation": SimulationConfig,
    "phantoms": PhantomConfig,
    "noise": NoiseConfig,
    "storage": StorageConfig,
}
