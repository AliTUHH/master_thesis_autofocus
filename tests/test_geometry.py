import math

import pytest

from holofocus.geometry import (
    HC_KEV_M,
    ConeBeamGeometry,
    energy_from_wavelength,
    fresnel_number_parallel,
    relative_fresnel_error,
    wavelength_from_energy,
    z01_from_fresnel_number,
)

PIXEL = 6.5e-6

# [1] Table 3: sample, energy / keV, z01 / m, z02 / m, Fr, array size of [1] Table 1
PAPER1_TABLE3 = [
    ("spider hair", 11.0, 79.95e-3, 19.661, 7.790e-5, 14336),
    ("cactus needle", 17.0, 285.41e-3, 19.652, 4.347e-4, 14336),
    ("tooth", 17.0, 81.708e-3, 19.652, 1.231e-4, 8192),
    ("magnesium wire", 11.0, 470.51e-3, 19.661, 4.678e-4, 8192),
]

# [2] Table 2 (Appendix A)
PAPER2_TABLE2 = [
    ("spider hair", 11.0, 79.4e-3, 19.661, 7.790e-5),
    ("cactus needle", 17.0, 284.1e-3, 19.652, 4.347e-4),
    ("tooth", 17.0, 81.0e-3, 19.652, 1.231e-4),
    ("magnesium-based wire", 11.0, 470.8e-3, 19.661, 4.678e-4),
    ("wire in flow-cell", 11.0, 329.3e-3, 19.914, 3.165e-4),
    ("simulation", 11.0, 100.0e-3, 20.000, 9.331e-5),
]

# hc = 1.2389 keV nm as hard-coded in models/cone_beam.py of the paper code (Zenodo v1.0.0)
HC_PAPER_CODE_KEV_M = 1.2389e-9


def test_wavelength_from_energy():
    assert HC_KEV_M == pytest.approx(1.2398419843e-9, rel=1e-9)
    assert wavelength_from_energy(11.0) == pytest.approx(1.127129e-10, rel=1e-6)
    assert wavelength_from_energy(17.0) == pytest.approx(7.293188e-11, rel=1e-6)
    assert energy_from_wavelength(wavelength_from_energy(12.3)) == pytest.approx(12.3)
    with pytest.raises(ValueError):
        wavelength_from_energy(0.0)


@pytest.mark.parametrize("name, energy, z01, z02, fr_table, array_size", PAPER1_TABLE3)
def test_paper1_table3_fresnel_numbers(name, energy, z01, z02, fr_table, array_size):
    geometry = ConeBeamGeometry(energy_kev=energy, z01=z01, z02=z02, pixel_size=PIXEL)
    # Energies are printed with three digits (+-0.45 %); the table is matched to < 0.1 %.
    assert geometry.fresnel_number == pytest.approx(fr_table, rel=1e-3), name
    # With the paper code's hc the residual is at the table's rounding level.
    fr_paper_code = geometry.fresnel_number * HC_KEV_M / HC_PAPER_CODE_KEV_M
    assert fr_paper_code == pytest.approx(fr_table, rel=6e-4), name
    # The arrays used in [1] Table 1 satisfy the sampling condition N >= 1/Fr.
    assert array_size >= geometry.min_grid_size()


@pytest.mark.parametrize("name, energy, z01, z02, fr_table", PAPER2_TABLE2)
def test_paper2_table2_fresnel_numbers(name, energy, z01, z02, fr_table):
    geometry = ConeBeamGeometry(energy_kev=energy, z01=z01, z02=z02, pixel_size=PIXEL)
    # Magnesium wire and flow-cell match to < 0.02 %. Spider hair, cactus needle and tooth
    # repeat the Fr of [1] Table 3 although z01 was refined (-0.5 to -0.9 %); Eq. (9) with
    # 6.5 um pixels gives +0.9 % more than the simulation row. See docs/laborbuch/01.
    assert geometry.fresnel_number == pytest.approx(fr_table, rel=1e-2), name


def test_magnification_and_effective_parallel_beam():
    g = ConeBeamGeometry(energy_kev=11.0, z01=0.1, z02=20.0)
    assert g.z12 == pytest.approx(19.9)
    assert g.magnification == pytest.approx((g.z01 + g.z12) / g.z01)
    assert g.effective_distance == pytest.approx(g.z01 * g.z12 / g.z02)
    via_parallel = fresnel_number_parallel(
        g.effective_pixel_size, g.wavelength, g.effective_distance
    )
    eq9 = g.pixel_size**2 / (g.wavelength * g.magnification * g.z12)
    assert via_parallel == pytest.approx(eq9, rel=1e-12)
    assert g.fresnel_number == pytest.approx(eq9, rel=1e-12)


def test_paper2_eq5_reduces_to_eq9_at_detector():
    g = ConeBeamGeometry(energy_kev=11.0, z01=79.95e-3, z02=19.661)
    assert g.fresnel_number_at(g.z12) == pytest.approx(g.fresnel_number, rel=1e-12)
    assert math.isinf(g.fresnel_number_at(0.0))
    assert g.fresnel_number_at(-1e-3) < 0


@pytest.mark.parametrize("dz01", [-5e-3, -1e-3, -1e-4, 1e-4, 1e-3, 5e-3])
def test_defocus_composition_identity(dz01):
    """Sample -> de-focus plane ([2] Eq. (11)) -> detector equals sample -> detector.

    In sample-plane pixel units: 1/Fr(z12) = 1/Fr(dz01) + 1/Fr_hat(z12_hat), where Fr_hat is
    the Fresnel number of the de-focused geometry ([2] Eq. (9)-(10)).
    """
    g = ConeBeamGeometry(energy_kev=11.0, z01=0.1, z02=20.0)
    lhs = 1.0 / g.fresnel_number
    rhs = 1.0 / g.fresnel_number_at(dz01) + 1.0 / g.defocused(dz01).fresnel_number
    assert rhs == pytest.approx(lhs, rel=1e-12)


def test_z01_from_fresnel_number_inverts_eq9():
    for z01 in (79.95e-3, 0.1, 0.47051):
        g = ConeBeamGeometry(energy_kev=11.0, z01=z01, z02=19.661)
        assert z01_from_fresnel_number(g.fresnel_number, 11.0, 19.661, PIXEL) == pytest.approx(
            z01, rel=1e-12
        )


def test_relative_fresnel_error_reproduces_paper2_fig3():
    """[2] Fig. 3: +-5 mm in z01 changes Fr by ~5 %, the same error in z02 by < 0.1 %."""
    g = ConeBeamGeometry(energy_kev=11.0, z01=0.1, z02=20.0)
    for dz in (-5e-3, 5e-3):
        assert 0.049 < relative_fresnel_error(g, dz01=dz) < 0.051
        assert relative_fresnel_error(g, dz02=dz) < 1e-3
    assert relative_fresnel_error(g, dz01=10e-3) == pytest.approx(0.1005, abs=5e-4)


def test_binning_scales_fresnel_number_quadratically():
    g = ConeBeamGeometry(energy_kev=11.0, z01=0.1, z02=20.0)
    assert g.binned(4).fresnel_number == pytest.approx(16 * g.fresnel_number, rel=1e-12)
    assert relative_fresnel_error(g.binned(4), dz01=5e-3) == pytest.approx(
        relative_fresnel_error(g, dz01=5e-3), rel=1e-12
    )


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        ConeBeamGeometry(energy_kev=11.0, z01=0.0, z02=20.0)
    with pytest.raises(ValueError):
        ConeBeamGeometry(energy_kev=11.0, z01=21.0, z02=20.0)
    with pytest.raises(ValueError):
        ConeBeamGeometry(energy_kev=-1.0, z01=0.1, z02=20.0)
