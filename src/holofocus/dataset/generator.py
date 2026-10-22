"""Labelled synthetic holograms for learning ``dz01`` (Stage 2).

For each sample: draw ``z01_guess`` (fixed or from a range) and ``dz01`` from
``dz01_range`` ([2] Eq. (17), ``Omega_z``); simulate the hologram with the true
``z01 = z01_guess + dz01``; add detector noise. Labels are ``dz01`` (primary target), ``z01``,
``z01_guess`` and the Fresnel numbers of the true and the guessed geometry.

Phantoms are defined on the detector grid of the true geometry (sample pixel ``dx / M``).
Their apparent size therefore carries no information about ``z01``; in an experiment the
physical object size is unknown as well.

Every sample has its own random stream ``SeedSequence(seed, spawn_key=(crc32(split), i))``,
so a sample can be regenerated on its own and the result does not depend on the order or the
number of samples generated. The detector effects of Stage 5 (PSF width, flat-field residual)
draw from a second stream with the spawn key ``(crc32(split), i, 1)``: switching them on
leaves the setup, phantom and photon number of a sample unchanged, and configs without them
give the same samples as before. The texture of Stage 6 (its RMS values, slope and field) draws
from a third stream ``(crc32(split), i, 2)`` in the same way: with or without texture, and
with any texture ranges, a sample keeps its setup, bulk object and noise draws, and configs
without texture give the same samples as before.

Fibers cross the field of view, so they are rendered on the whole simulation grid instead of
being embedded in free space; so is every object under a background texture.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np
import torch

from ..arrays import bin_sum, crop_center
from ..device import complex_dtype, get_device, real_dtype
from ..forward import object_from_phase, simulate_hologram
from ..noise import add_detector_noise, flat_field_residual
from ..phantoms import PHANTOM_TYPES, random_phantom, texture_field
from .config import DatasetConfig

LABEL_KEYS = ("dz01", "z01", "z01_guess", "fresnel_number", "fresnel_number_guess")
META_KEYS = (
    "phantom_type",
    "phase_shift",
    "photons",
    "a0",
    "effective_pixel_size",
    "psf_sigma",
    "flat_field_rms",
    "texture_rms",
    "background_texture_rms",
    "texture_slope",
)


def _log_uniform(u: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return 0.0 if hi == 0 else float(lo * (hi / lo) ** u)


def sample_seed(seed: int, split: str, index: int, stream: int = 0) -> np.random.SeedSequence:
    """Seed sequence of sample ``index`` in ``split`` (``stream=1``: detector effects,
    ``stream=2``: object texture)."""
    key = (zlib.crc32(split.encode()), index) + ((stream,) if stream else ())
    return np.random.SeedSequence(entropy=seed, spawn_key=key)


@dataclass
class Sample:
    hologram: np.ndarray
    clean: np.ndarray
    phase: np.ndarray
    labels: dict[str, float]
    meta: dict[str, float]


class HologramGenerator:
    """Draws samples of a :class:`DatasetConfig` on CPU or GPU."""

    def __init__(self, config: DatasetConfig, device: str | None = None) -> None:
        self.config = config
        self.device = get_device(device or config.simulation.device)
        self.real_dtype = real_dtype(config.simulation.precision)
        self.complex_dtype = complex_dtype(config.simulation.precision)
        self.fov_shape = config.geometry.fov_shape
        self.simulation_shape = config.simulation_shape
        names, weights = zip(*config.phantoms.types.items(), strict=True)
        self._types = names
        self._p = np.asarray(weights, dtype=np.float64) / sum(weights)

    def sample(self, split: str, index: int) -> Sample:
        rng = np.random.default_rng(sample_seed(self.config.seed, split, index))
        detector_rng = np.random.default_rng(sample_seed(self.config.seed, split, index, 1))
        texture_rng = np.random.default_rng(sample_seed(self.config.seed, split, index, 2))
        return self.draw(rng, detector_rng, texture_rng)

    def draw(
        self,
        rng: np.random.Generator,
        detector_rng: np.random.Generator,
        texture_rng: np.random.Generator | None = None,
    ) -> Sample:
        cfg = self.config
        ph = cfg.phantoms
        if ph.textured and texture_rng is None:
            raise ValueError(f"config {cfg.name!r} has a texture and needs texture_rng")
        lo, hi = cfg.geometry.guess_bounds
        z01_guess = float(rng.uniform(lo, hi)) if hi > lo else lo
        dz01 = float(rng.uniform(*cfg.sampling.dz01_range))
        kind = self._types[int(rng.choice(len(self._types), p=self._p))]
        if ph.phase_per_pixel_range is None:
            phase_shift = float(rng.uniform(*ph.phase_shift_range))
        else:
            log_lo, log_hi = (math.log(v) for v in ph.phase_per_pixel_range)
            phase_per_pixel = float(math.exp(rng.uniform(log_lo, log_hi)))
        a0 = float(rng.uniform(*cfg.simulation.a0_range))
        log_lo, log_hi = (math.log(v) for v in cfg.noise.photons_range)
        photons = float(math.exp(rng.uniform(log_lo, log_hi)))
        psf_sigma = float(detector_rng.uniform(*cfg.simulation.psf_sigma_range))
        flat_field_rms = float(detector_rng.uniform(*cfg.noise.flat_field_range))
        texture_rms = background_texture_rms = texture_slope = 0.0
        if ph.textured:
            u = texture_rng.uniform(size=3)
            texture_rms = _log_uniform(u[0], ph.texture_rms_range)
            background_texture_rms = _log_uniform(u[1], ph.background_texture_rms_range)
            slope_lo, slope_hi = ph.texture_slope_range
            texture_slope = float(slope_lo + (slope_hi - slope_lo) * u[2])

        s = cfg.simulation.oversample
        fine_shape = (self.fov_shape[0] * s, self.fov_shape[1] * s)
        extended = kind == "fiber" or ph.background_texture_rms_range[1] > 0
        thickness = random_phantom(
            rng,
            kind,
            fine_shape,
            size_range=ph.size_range,
            margin=ph.margin,
            supersample=ph.supersample,
            canvas=self.simulation_shape if extended else None,
            fiber_diameter_range=ph.fiber_diameter_range,
            normalize=ph.phase_per_pixel_range is None,
        )
        if ph.phase_per_pixel_range is not None:
            peak = float(crop_center(thickness, fine_shape).max())
            phase_shift = phase_per_pixel * peak / s
            thickness = thickness / peak if peak > 0 else thickness
        geometry = cfg.geometry.at(z01_guess + dz01)
        phase = torch.from_numpy(-phase_shift * thickness).to(self.device, self.real_dtype)
        if ph.textured:
            field = texture_field(
                thickness.shape,
                texture_rng,
                texture_slope,
                ph.texture_cutoff,
                s,
                self.device,
                self.real_dtype,
            )
            root = torch.from_numpy(np.sqrt(np.clip(thickness, 0.0, None)))
            amplitude = texture_rms * root.to(self.device, self.real_dtype) + background_texture_rms
            phase = phase - amplitude * field
        obj = object_from_phase(phase, ph.delta_beta)
        fresnel_number = geometry.fresnel_number / s**2
        psf = psf_sigma * s / cfg.geometry.binning
        if extended:
            clean = simulate_hologram(obj, fresnel_number, a0=a0, psf_sigma=psf)
            clean, phase = crop_center(clean, fine_shape), crop_center(phase, fine_shape)
        else:
            clean = simulate_hologram(
                obj, fresnel_number, padded_shape=self.simulation_shape, a0=a0, psf_sigma=psf
            )
        if s > 1:
            clean = bin_sum(clean, s) / s**2
            phase = bin_sum(phase, s) / s**2
        clean_np = clean.double().cpu().numpy()
        noisy = add_detector_noise(
            clean_np, rng, cfg.noise.model, photons=photons, read_noise=cfg.noise.read_noise
        )
        if flat_field_rms > 0:
            noisy = noisy * (
                1.0
                + flat_field_residual(
                    self.fov_shape, detector_rng, flat_field_rms, cfg.noise.flat_field_cutoff
                )
            )
        guess = cfg.geometry.at(z01_guess)
        return Sample(
            hologram=noisy.astype(np.float32),
            clean=clean_np.astype(np.float32),
            phase=phase.cpu().numpy().astype(np.float32),
            labels={
                "dz01": dz01,
                "z01": geometry.z01,
                "z01_guess": z01_guess,
                "fresnel_number": geometry.fresnel_number,
                "fresnel_number_guess": guess.fresnel_number,
            },
            meta={
                "phantom_type": PHANTOM_TYPES.index(kind),
                "phase_shift": phase_shift,
                "photons": photons if cfg.noise.model != "none" else math.inf,
                "a0": a0,
                "effective_pixel_size": geometry.effective_pixel_size,
                "psf_sigma": psf_sigma,
                "flat_field_rms": flat_field_rms,
                "texture_rms": texture_rms,
                "background_texture_rms": background_texture_rms,
                "texture_slope": texture_slope,
            },
        )
