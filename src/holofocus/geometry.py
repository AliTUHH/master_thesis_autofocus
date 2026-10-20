"""Cone-beam geometry and Fresnel numbers for X-ray near-field holography (NFH).

Conventions
-----------
All lengths are in metres, energies in keV. Distances are measured from the focus of the
optics, as in Fig. 1 of [1] and [2]:

* ``z01``: focus to sample,
* ``z12``: sample to detector,
* ``z02 = z01 + z12``: focus to detector (treated as known, [2] Sec. 2).

``pixel_size`` is the physical detector pixel pitch (6.5 um at P05, [1] Sec. 5.1),
multiplied by the binning factor if the detector is down-sampled.

References
----------
[1] J. Dora et al., "Artifact-suppressing reconstruction of strongly interacting objects in
    X-ray near-field holography without a spatial support constraint", Opt. Express 32(7),
    10801 (2024). Eq. (7)-(9).
[2] J. Dora et al., "Model-based autofocus for near-field phase retrieval", Opt. Express
    33(4), 6641 (2025). Eq. (4)-(10).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

PLANCK_CONSTANT = 6.62607015e-34  # J s, exact (SI 2019)
SPEED_OF_LIGHT = 299_792_458.0  # m / s, exact
ELEMENTARY_CHARGE = 1.602176634e-19  # C, exact
HC_KEV_M = PLANCK_CONSTANT * SPEED_OF_LIGHT / ELEMENTARY_CHARGE * 1e-3  # 1.23984198e-9 keV m

P05_PIXEL_SIZE = 6.5e-6  # Hamamatsu C12849-101U at P05, [1] Sec. 5.1
P05_DETECTOR_SHAPE = (2048, 2048)


def wavelength_from_energy(energy_kev: float) -> float:
    """Photon wavelength in metres, ``lambda = h c / E``."""
    if energy_kev <= 0:
        raise ValueError(f"energy must be positive, got {energy_kev} keV")
    return HC_KEV_M / energy_kev


def energy_from_wavelength(wavelength: float) -> float:
    """Photon energy in keV for a wavelength in metres."""
    if wavelength <= 0:
        raise ValueError(f"wavelength must be positive, got {wavelength} m")
    return HC_KEV_M / wavelength


def fresnel_number_parallel(pixel_size: float, wavelength: float, distance: float) -> float:
    """Fresnel number of a parallel-beam setup, ``Fr = dx^2 / (lambda z)``.

    [1] Eq. (7), [2] Eq. (4). A negative ``distance`` gives a negative Fresnel number, which
    the propagator interprets as back-propagation. ``distance == 0`` returns ``inf``
    (identity propagation).
    """
    if pixel_size <= 0 or wavelength <= 0:
        raise ValueError("pixel_size and wavelength must be positive")
    if distance == 0:
        return math.inf
    return pixel_size**2 / (wavelength * distance)


@dataclass(frozen=True)
class ConeBeamGeometry:
    """Cone-beam NFH setup described by energy, ``z01``, ``z02`` and detector pixel size.

    The Fresnel scaling theorem maps the cone beam onto an equivalent parallel beam with
    the effective pixel size ``dx / M`` and the effective propagation distance ``z12 / M``,
    which yields the Fresnel number of [1] Eq. (9).
    """

    energy_kev: float
    z01: float
    z02: float
    pixel_size: float = P05_PIXEL_SIZE

    def __post_init__(self) -> None:
        if self.energy_kev <= 0:
            raise ValueError(f"energy must be positive, got {self.energy_kev} keV")
        if self.pixel_size <= 0:
            raise ValueError(f"pixel_size must be positive, got {self.pixel_size} m")
        if not 0 < self.z01 < self.z02:
            raise ValueError(f"need 0 < z01 < z02, got z01={self.z01} m, z02={self.z02} m")

    @property
    def wavelength(self) -> float:
        return wavelength_from_energy(self.energy_kev)

    @property
    def z12(self) -> float:
        return self.z02 - self.z01

    @property
    def magnification(self) -> float:
        """Geometric magnification ``M = (z01 + z12) / z01``, [1] Eq. (8)."""
        return self.z02 / self.z01

    @property
    def effective_pixel_size(self) -> float:
        """Pixel size in the sample plane, ``dx / M``."""
        return self.pixel_size / self.magnification

    @property
    def effective_distance(self) -> float:
        """Propagation distance of the equivalent parallel beam, ``z12 / M = z01 z12 / z02``."""
        return self.z12 / self.magnification

    @property
    def fresnel_number(self) -> float:
        """Fresnel number sample -> detector, ``Fr = dx^2 / (lambda M z12)``, [1] Eq. (9)."""
        return self.pixel_size**2 / (self.wavelength * self.magnification * self.z12)

    def fresnel_number_at(self, z: float) -> float:
        """Fresnel number for propagating the sample-plane field by a distance ``z``.

        [2] Eq. (5)-(7): ``Fr(z) = (dx / M1)^2 / (lambda z / M2)`` with ``M1 = z02 / z01``
        (sample -> detector) and ``M2 = (z01 + z) / z01`` (sample -> plane at ``z``). The
        result refers to the sample-plane pixel grid ``dx / M1``. ``z = z12`` reproduces
        :attr:`fresnel_number`; ``z = dz01`` is the defocus propagation of [2] Eq. (11).
        """
        if z == 0:
            return math.inf
        m1 = self.magnification
        m2 = (self.z01 + z) / self.z01
        if m2 <= 0:
            raise ValueError(f"plane at z={z} m lies behind the focus (z01={self.z01} m)")
        return (self.pixel_size / m1) ** 2 / (self.wavelength * (z / m2))

    def with_z01(self, z01: float) -> ConeBeamGeometry:
        """Same setup with another focus-sample distance (``z02`` stays fixed)."""
        return replace(self, z01=z01)

    def defocused(self, dz01: float) -> ConeBeamGeometry:
        """Geometry assumed for a de-focus ``dz01``: ``z01 + dz01`` and ``z12 - dz01``.

        [2] Eq. (9)-(10); ``z02`` is unchanged.
        """
        return self.with_z01(self.z01 + dz01)

    def binned(self, factor: float) -> ConeBeamGeometry:
        """Same setup with the detector down-sampled by ``factor`` (Fr grows by factor^2)."""
        if factor <= 0:
            raise ValueError("binning factor must be positive")
        return replace(self, pixel_size=self.pixel_size * factor)

    def min_grid_size(self) -> int:
        """Smallest array size ``N >= 1 / Fr`` that samples the Fresnel kernel, [1] Sec. 3.2.2."""
        return math.ceil(1.0 / abs(self.fresnel_number))

    def as_dict(self) -> dict[str, float]:
        """Inputs plus derived quantities, convenient for metadata and logging."""
        return {
            **asdict(self),
            "wavelength": self.wavelength,
            "z12": self.z12,
            "magnification": self.magnification,
            "effective_pixel_size": self.effective_pixel_size,
            "effective_distance": self.effective_distance,
            "fresnel_number": self.fresnel_number,
        }


def z01_from_fresnel_number(
    fresnel_number: float, energy_kev: float, z02: float, pixel_size: float = P05_PIXEL_SIZE
) -> float:
    """Invert [1] Eq. (9) for ``z01`` at fixed ``z02``.

    ``Fr = dx^2 z01 / (lambda z02 (z02 - z01))`` gives
    ``z01 = Fr lambda z02^2 / (dx^2 + Fr lambda z02)``.
    """
    if fresnel_number <= 0:
        raise ValueError("fresnel_number must be positive")
    wavelength = wavelength_from_energy(energy_kev)
    numerator = fresnel_number * wavelength * z02**2
    return numerator / (pixel_size**2 + fresnel_number * wavelength * z02)


def relative_fresnel_error(
    geometry: ConeBeamGeometry, dz01: float = 0.0, dz02: float = 0.0
) -> float:
    """Relative change of ``Fr`` for errors in ``z01`` and ``z02``, [2] Eq. (8).

    ``ERR = |Fr_hat / Fr - 1|`` where ``Fr_hat`` uses ``z01 + dz01`` and ``z02 + dz02``.
    """
    perturbed = replace(geometry, z01=geometry.z01 + dz01, z02=geometry.z02 + dz02)
    return abs(perturbed.fresnel_number / geometry.fresnel_number - 1.0)
