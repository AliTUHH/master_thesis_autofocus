"""HoloForge experiment setup with randomised propagation geometry.

HoloForge resolves the ``"type"`` string of the ``setup`` config block via ``getattr`` on the
module ``holowizard.forge.experiment.setup``. :func:`register_forge_setup` injects
:class:`NFHRandomDistSetup` into that module so that the regular Forge machinery
(``ConfigParser`` -> generators -> ``HDF5Labeller``) can use it like a built-in setup.

Units follow HoloWizard: distances in mm, detector pixel size in mm, energy in keV.
"""

from __future__ import annotations

from typing import Any

import holowizard.forge.experiment as forge_experiment
import holowizard.forge.experiment.setup as forge_setup_module
import holowizard.forge.utils.torch_settings as torch_settings
import numpy as np
import torch
from holowizard.forge.experiment.setup import NFHSetup
from holowizard.forge.utils import calc_Fr
from torch import fft

from src.utils.physics import DISTANCE_DECIMALS_MM

__all__ = ["NFHRandomDistSetup", "register_forge_setup", "reset_global_forge_setup", "DISTRIBUTIONS"]

DISTRIBUTIONS: tuple[str, ...] = ("uniform", "loguniform")


class NFHRandomDistSetup(NFHSetup):
    """NFH setup whose focus-to-object distance z01 (and optionally z02) is redrawn for every hologram.

    HoloForge calls ``setup.kernel`` exactly once per simulated hologram; the inherited ``kernel``
    property calls :meth:`get_distances`, which is where a new geometry is drawn. The labeller
    afterwards reads :meth:`as_dict`, so the stored ``z01``/``z02``/``Fr`` labels are exactly the
    values used for the propagation of that sample.

    Args:
        detector_size: Detector width/height in pixels (square detector).
        detector_px_size: Detector pixel size in mm (e.g. 0.0065 mm = 6.5 µm).
        padding_factor: Padding factor of the simulation grid (probe size = detector_size * padding_factor).
        downsample_factor: Integer downsampling of the detector grid (1 = none).
        energy: Photon energy in keV.
        z01_min: Lower bound of the focus-to-object distance in mm.
        z01_max: Upper bound of the focus-to-object distance in mm.
        z02_min: Focus-to-detector distance in mm (fixed value when ``z02_max`` is ``None``).
        z02_max: Optional upper bound in mm; z02 is then drawn uniformly from ``[z02_min, z02_max]``.
        z01_distribution: ``"uniform"`` or ``"loguniform"`` sampling of z01 within its bounds.
        seed: Seed of the private NumPy generator used for the geometry draws.
    """

    def __init__(
        self,
        detector_size: int,
        detector_px_size: float,
        padding_factor: int,
        downsample_factor: int,
        energy: float,
        z01_min: float,
        z01_max: float,
        z02_min: float,
        z02_max: float | None = None,
        z01_distribution: str = "uniform",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            detector_size=detector_size,
            detector_px_size=detector_px_size,
            padding_factor=padding_factor,
            downsample_factor=downsample_factor,
            energy=energy,
        )
        if z01_distribution not in DISTRIBUTIONS:
            raise ValueError(f"z01_distribution must be one of {DISTRIBUTIONS}, got {z01_distribution!r}")
        if not 0.0 < z01_min <= z01_max:
            raise ValueError(f"require 0 < z01_min <= z01_max, got {z01_min=}, {z01_max=}")
        z02_hi = z02_min if z02_max is None else z02_max
        if not z02_min <= z02_hi:
            raise ValueError(f"require z02_min <= z02_max, got {z02_min=}, {z02_max=}")
        if z01_max >= z02_min:
            raise ValueError(f"require z01_max < z02_min, got {z01_max=}, {z02_min=}")

        self._z01_bounds = (float(z01_min), float(z01_max))
        self._z02_bounds = (float(z02_min), float(z02_hi))
        self._z01_distribution = z01_distribution
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._num_draws = 0
        self._freq_sq: torch.Tensor | None = None
        self._z01 = 0.0
        self._z02 = 0.0
        self._Fr = 0.0
        self.get_distances()

    def get_distances(self) -> tuple[float, float]:
        """Draw a new (z01, z02) pair in mm, update the label state and return it.

        Distances are rounded to 1 µm, the resolution HoloWizard uses internally for Fr.
        """
        lo, hi = self._z01_bounds
        if self._z01_distribution == "loguniform":
            z01 = float(np.exp(self._rng.uniform(np.log(lo), np.log(hi))))
        else:
            z01 = float(self._rng.uniform(lo, hi))
        z02_lo, z02_hi = self._z02_bounds
        z02 = z02_lo if z02_lo == z02_hi else float(self._rng.uniform(z02_lo, z02_hi))

        self._z01 = round(z01, DISTANCE_DECIMALS_MM)
        self._z02 = round(z02, DISTANCE_DECIMALS_MM)
        self._Fr = float(calc_Fr(self.energy, self._z01, self._z02, self.detector_px_size))
        self._num_draws += 1
        return self._z01, self._z02

    def create_kernel(self, Fr: float) -> torch.Tensor:
        """Fresnel propagator ``exp(-i*pi/Fr * (xi^2 + eta^2))`` on the padded grid (cycles per pixel).

        The squared frequency grid is cached because a new kernel is needed for every sample.
        """
        if self._freq_sq is None:
            freqs = fft.fftfreq(self.probe_size, device=torch_settings.get_torch_device(), dtype=torch.float)
            xi, eta = torch.meshgrid(freqs, freqs, indexing="ij")
            self._freq_sq = xi * xi + eta * eta
        return torch.exp((-1j * np.pi) / Fr * self._freq_sq).type(torch.cfloat)

    def as_dict(self) -> dict[str, Any]:
        """Setup values of the most recently drawn sample; consumed per sample by ``HDF5Labeller``."""
        return super().as_dict() | dict(z01=self.z01, z02=self.z02, Fr=self.Fr)

    @property
    def z01(self) -> float:
        """Focus-to-object distance of the current sample in mm."""
        return self._z01

    @property
    def z02(self) -> float:
        """Focus-to-detector distance of the current sample in mm."""
        return self._z02

    @property
    def Fr(self) -> float:
        """Pixel Fresnel number of the current sample (dimensionless)."""
        return self._Fr

    @property
    def z01_bounds(self) -> tuple[float, float]:
        """Sampling interval of z01 in mm."""
        return self._z01_bounds

    @property
    def z02_bounds(self) -> tuple[float, float]:
        """Sampling interval of z02 in mm (equal bounds for a fixed z02)."""
        return self._z02_bounds

    @property
    def z01_distribution(self) -> str:
        return self._z01_distribution

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def num_draws(self) -> int:
        """Number of geometry draws so far (includes the initial draw in ``__init__``)."""
        return self._num_draws


def register_forge_setup() -> None:
    """Make ``"NFHRandomDistSetup"`` resolvable as ``setup.type`` in HoloForge configs (idempotent)."""
    setattr(forge_setup_module, NFHRandomDistSetup.__name__, NFHRandomDistSetup)


def reset_global_forge_setup() -> None:
    """Clear HoloForge's process-wide setup singleton so a new ``ConfigParser`` can install its own setup."""
    forge_experiment.GLOBAL_EXP_SETUP = None
