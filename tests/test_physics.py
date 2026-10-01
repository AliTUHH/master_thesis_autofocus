"""Fresnel-number relations must agree with HoloWizard's ``calc_Fr`` and invert consistently."""

from __future__ import annotations

import numpy as np
import pytest
from holowizard.forge.utils import calc_Fr

from src.utils import physics

PARAMETER_SETS = [
    # (z01_mm, z02_mm, energy_kev, px_mm)
    (250.0, 20000.0, 11.0, 0.0065),
    (50.0, 20000.0, 11.0, 0.0065),
    (123.456, 19661.0, 11.0, 0.0065),
    (470.51, 19661.0, 17.0, 0.0065),
    (77.7777, 20000.0, 8.0, 0.052),
    (299.999, 20000.0, 11.0, 0.208),
]


@pytest.mark.parametrize("z01, z02, energy, px", PARAMETER_SETS)
def test_fresnel_number_matches_holowizard(z01: float, z02: float, energy: float, px: float) -> None:
    expected = calc_Fr(energy, z01, z02, px)
    assert physics.fresnel_number(z01, z02, energy, px) == pytest.approx(expected, rel=1e-6)


def test_fresnel_number_reference_value() -> None:
    assert physics.fresnel_number(250.0, 20000.0, 11.0, 0.0065) == pytest.approx(2.3725e-4, rel=1e-4)


def test_fresnel_number_vectorised_matches_scalar() -> None:
    z01 = np.array([50.0, 150.0, 250.0])
    fr = physics.fresnel_number(z01, 20000.0, 11.0, 0.0065)
    assert fr.shape == (3,)
    for value, z in zip(fr, z01):
        assert value == pytest.approx(calc_Fr(11.0, float(z), 20000.0, 0.0065), rel=1e-6)


@pytest.mark.parametrize("z01, z02, energy, px", PARAMETER_SETS)
def test_z01_roundtrip(z01: float, z02: float, energy: float, px: float) -> None:
    fr = physics.fresnel_number(z01, z02, energy, px)
    recovered = physics.z01_from_fresnel_number(fr, z02, energy, px)
    # fresnel_number rounds distances to 1 µm like HoloWizard, so the roundtrip is exact to that resolution
    assert recovered == pytest.approx(round(z01, 3), abs=1e-6)


def test_wavelength_and_effective_geometry() -> None:
    assert physics.wavelength_nm(11.0) == pytest.approx(0.112709, rel=1e-5)
    assert physics.magnification(250.0, 20000.0) == pytest.approx(80.0)
    assert physics.effective_pixel_size_mm(250.0, 20000.0, 0.0065) == pytest.approx(0.0065 / 80.0)
    assert physics.effective_distance_mm(250.0, 20000.0) == pytest.approx(19750.0 / 80.0)
    fr_direct = physics.fresnel_number(250.0, 20000.0, 11.0, 0.0065)
    dx_eff = physics.effective_pixel_size_mm(250.0, 20000.0, 0.0065)
    z_eff = physics.effective_distance_mm(250.0, 20000.0)
    lam_mm = physics.wavelength_nm(11.0) * 1e-6
    assert fr_direct == pytest.approx(dx_eff**2 / (lam_mm * z_eff), rel=1e-9)


def test_invalid_geometry_raises() -> None:
    with pytest.raises(ValueError):
        physics.fresnel_number(0.0, 20000.0, 11.0, 0.0065)
    with pytest.raises(ValueError):
        physics.fresnel_number(250.0, 200.0, 11.0, 0.0065)
    with pytest.raises(ValueError):
        physics.wavelength_nm(0.0)


def test_target_conversions_are_consistent() -> None:
    z01 = np.array([60.0, 180.0, 290.0])
    fr = physics.fresnel_number(z01, 20000.0, 11.0, 0.052)
    log_fr = np.log(fr)
    np.testing.assert_allclose(physics.target_to_fr(log_fr, "log_fr", 20000.0, 11.0, 0.052), fr, rtol=1e-12)
    np.testing.assert_allclose(physics.target_to_fr(fr, "fr", 20000.0, 11.0, 0.052), fr, rtol=1e-12)
    np.testing.assert_allclose(physics.target_to_fr(z01, "z01_mm", 20000.0, 11.0, 0.052), fr, rtol=1e-12)
    np.testing.assert_allclose(physics.target_to_z01_mm(log_fr, "log_fr", 20000.0, 11.0, 0.052), z01, atol=1e-6)
    np.testing.assert_allclose(physics.target_to_z01_mm(fr, "fr", 20000.0, 11.0, 0.052), z01, atol=1e-6)
    np.testing.assert_allclose(physics.target_to_z01_mm(z01, "z01_mm", 20000.0, 11.0, 0.052), z01)
    with pytest.raises(ValueError):
        physics.target_to_fr(fr, "bogus", 20000.0, 11.0, 0.052)
