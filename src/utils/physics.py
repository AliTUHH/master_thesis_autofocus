"""Geometry and Fresnel-number relations for cone-beam near-field holography (NFH).

All functions follow the unit conventions of HoloWizard / HoloForge:

* distances ``z01`` (source/focus to object) and ``z02`` (source/focus to detector) in **mm**,
* detector pixel size ``px`` in **mm** (6.5 µm = 0.0065 mm),
* photon energy in **keV**,
* wavelengths in **nm**,
* the Fresnel number ``Fr`` is dimensionless.

The cone-beam geometry is mapped onto an equivalent parallel-beam problem via the Fresnel
scaling theorem with magnification ``M = z02 / z01``: effective pixel size ``px / M`` and
effective propagation distance ``(z02 - z01) / M``. The resulting *pixel* Fresnel number is

    Fr = (px / M)^2 / (lambda * (z02 - z01) / M) = px^2 * z01 / (lambda * z02 * (z02 - z01)).

Functions accept Python floats or NumPy arrays (broadcasting rules apply).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "HC_KEV_NM",
    "DISTANCE_DECIMALS_MM",
    "wavelength_nm",
    "magnification",
    "effective_pixel_size_mm",
    "effective_distance_mm",
    "fresnel_number",
    "z01_from_fresnel_number",
    "target_to_fr",
    "target_to_z01_mm",
    "TARGET_MODES",
]

HC_KEV_NM: float = 1.2398
"""Product h*c in keV*nm as used by HoloWizard (``lambda[nm] = 1.2398 / E[keV]``)."""

DISTANCE_DECIMALS_MM: int = 3
"""HoloWizard rounds z01 and z02 to this many decimals in mm (1 µm) before computing Fr."""

TARGET_MODES: tuple[str, ...] = ("log_fr", "fr", "z01_mm")
"""Supported regression targets: natural log of Fr, raw Fr, or z01 in mm."""


def wavelength_nm(energy_kev: ArrayLike) -> NDArray[np.float64] | float:
    """Photon wavelength in nm for a given energy in keV (``1.2398 / E``)."""
    energy = np.asarray(energy_kev, dtype=np.float64)
    if np.any(energy <= 0):
        raise ValueError("energy_kev must be positive")
    return HC_KEV_NM / energy


def magnification(z01_mm: ArrayLike, z02_mm: ArrayLike) -> NDArray[np.float64] | float:
    """Geometric magnification ``M = z02 / z01`` of the cone-beam setup (both in mm)."""
    z01 = np.asarray(z01_mm, dtype=np.float64)
    z02 = np.asarray(z02_mm, dtype=np.float64)
    if np.any(z01 <= 0) or np.any(z02 <= z01):
        raise ValueError("require 0 < z01_mm < z02_mm")
    return z02 / z01


def effective_pixel_size_mm(z01_mm: ArrayLike, z02_mm: ArrayLike, px_mm: ArrayLike) -> NDArray[np.float64] | float:
    """Effective (demagnified) pixel size ``px / M`` in mm in the object plane."""
    return np.asarray(px_mm, dtype=np.float64) / magnification(z01_mm, z02_mm)


def effective_distance_mm(z01_mm: ArrayLike, z02_mm: ArrayLike) -> NDArray[np.float64] | float:
    """Effective parallel-beam propagation distance ``(z02 - z01) / M`` in mm."""
    z01 = np.asarray(z01_mm, dtype=np.float64)
    z02 = np.asarray(z02_mm, dtype=np.float64)
    return (z02 - z01) / magnification(z01, z02)


def fresnel_number(z01_mm: ArrayLike, z02_mm: ArrayLike, energy_kev: ArrayLike, px_mm: ArrayLike) -> NDArray[np.float64] | float:
    """Pixel Fresnel number of the cone-beam NFH setup, identical to ``holowizard.forge.utils.calc_Fr``.

    Mirrors HoloWizard bit-for-bit: distances are rounded to 1e-3 mm (1 µm) and the arithmetic is
    carried out in nm, i.e. ``Fr = px_nm^2 / (lambda_nm * (z02_nm - z01_nm) * M)``.

    Args:
        z01_mm: Focus-to-object distance in mm.
        z02_mm: Focus-to-detector distance in mm.
        energy_kev: Photon energy in keV.
        px_mm: Detector pixel size in mm (after any downsampling).

    Returns:
        Dimensionless Fresnel number (scalar or array following NumPy broadcasting).
    """
    z01_nm = np.round(np.asarray(z01_mm, dtype=np.float64), DISTANCE_DECIMALS_MM) * 1e6
    z02_nm = np.round(np.asarray(z02_mm, dtype=np.float64), DISTANCE_DECIMALS_MM) * 1e6
    if np.any(z01_nm <= 0) or np.any(z02_nm <= z01_nm):
        raise ValueError("require 0 < z01_mm < z02_mm")
    px_nm = np.asarray(px_mm, dtype=np.float64) * 1e6
    lam_nm = wavelength_nm(energy_kev)
    mag = z02_nm / z01_nm
    fr = px_nm**2 / (lam_nm * (z02_nm - z01_nm) * mag)
    return fr if fr.ndim else float(fr)


def z01_from_fresnel_number(fr: ArrayLike, z02_mm: ArrayLike, energy_kev: ArrayLike, px_mm: ArrayLike) -> NDArray[np.float64] | float:
    """Invert :func:`fresnel_number` for ``z01`` (closed form, no rounding).

    ``z01 = Fr * lambda * z02^2 / (px^2 + Fr * lambda * z02)`` with all lengths in mm.

    Args:
        fr: Dimensionless Fresnel number.
        z02_mm: Focus-to-detector distance in mm.
        energy_kev: Photon energy in keV.
        px_mm: Detector pixel size in mm.

    Returns:
        Focus-to-object distance z01 in mm.
    """
    fr_arr = np.asarray(fr, dtype=np.float64)
    z02 = np.asarray(z02_mm, dtype=np.float64)
    px = np.asarray(px_mm, dtype=np.float64)
    lam_mm = wavelength_nm(energy_kev) * 1e-6
    numerator = fr_arr * lam_mm * z02**2
    z01 = numerator / (px**2 + fr_arr * lam_mm * z02)
    return z01 if z01.ndim else float(z01)


def _check_target_mode(target_mode: str) -> None:
    if target_mode not in TARGET_MODES:
        raise ValueError(f"unknown target_mode {target_mode!r}; expected one of {TARGET_MODES}")


def target_to_fr(values: ArrayLike, target_mode: str, z02_mm: ArrayLike, energy_kev: ArrayLike, px_mm: ArrayLike) -> NDArray[np.float64]:
    """Convert regression targets/predictions of any supported mode to Fresnel numbers."""
    _check_target_mode(target_mode)
    vals = np.asarray(values, dtype=np.float64)
    if target_mode == "log_fr":
        return np.exp(vals)
    if target_mode == "fr":
        return vals
    return np.asarray(fresnel_number(vals, z02_mm, energy_kev, px_mm), dtype=np.float64)


def target_to_z01_mm(
    values: ArrayLike, target_mode: str, z02_mm: ArrayLike, energy_kev: ArrayLike, px_mm: ArrayLike
) -> NDArray[np.float64]:
    """Convert regression targets/predictions of any supported mode to z01 in mm."""
    _check_target_mode(target_mode)
    vals = np.asarray(values, dtype=np.float64)
    if target_mode == "z01_mm":
        return vals
    fr = np.exp(vals) if target_mode == "log_fr" else vals
    return np.asarray(z01_from_fresnel_number(fr, z02_mm, energy_kev, px_mm), dtype=np.float64)
