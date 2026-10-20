import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holofocus.arrays import center_slices, pad_center
from holofocus.asrm import ASRMSettings, ASRMStage
from holofocus.autofocus import (
    FOCUS_EXTREMUM,
    IMAGE_CRITERIA,
    extremum_offset,
    find_focus,
    image_criteria,
    model_fit_error,
    model_fit_error_eq14,
    nelder_mead,
    normalize_curve,
    prefilter,
    reconstruction_focus_curve,
    simulated_focus_curve,
    symmetric_offsets,
)
from holofocus.autofocus import search as search_module
from holofocus.autofocus.criteria import (
    gini_of_gradient,
    gradient_sum,
    laplacian,
    spectral,
    tenengrad_of_gradient,
    variance,
)
from holofocus.constraints import project_wavefield
from holofocus.forward import exit_wave, object_from_phase, simulate_hologram
from holofocus.geometry import ConeBeamGeometry, z01_from_fresnel_number
from holofocus.phantoms import demo_object
from holofocus.propagation import FresnelPropagator

GEOMETRY = ConeBeamGeometry(energy_kev=11.0, z01=0.1, z02=20.0).binned(16)
PADDED = (512, 512)


@pytest.fixture(scope="module")
def demo():
    t = torch.from_numpy(demo_object((128, 128)))
    return object_from_phase(-0.5 * t, delta_beta_ratio=200.0)


def _penalized(func):
    """Objective with the repeated-point penalty of the reference ``find_focus``."""
    seen = set()

    def wrapped(x):
        key = tuple(np.asarray(x, dtype=float).ravel())
        if key in seen:
            return sys.float_info.max
        seen.add(key)
        return func(x)

    return wrapped


def _scipy_points(func, simplex, bounds, **options):
    optimize = pytest.importorskip("scipy.optimize")
    points = []

    def recorded(x):
        points.append(np.array(x, dtype=float))
        return func(x)

    result = optimize.minimize(
        recorded,
        np.asarray(simplex[0], dtype=float),
        method="Nelder-Mead",
        bounds=bounds,
        options={"initial_simplex": simplex, **options},
    )
    return result, points


@pytest.mark.parametrize("target", [0.3, 0.02, 0.97, 0.0, 1.0, 0.5001])
@pytest.mark.parametrize("shape", ["quadratic", "kink"])
def test_nelder_mead_follows_scipy_in_one_dimension(target, shape):
    def f(x):
        d = float(x[0]) - target
        return d * d if shape == "quadratic" else abs(d) + 0.1 * d

    simplex, bounds = [[0.0], [1.0]], [(0.0, 1.0)]
    ref, ref_points = _scipy_points(_penalized(f), simplex, bounds, xatol=1e-3, fatol=1e6)
    ours = nelder_mead(_penalized(f), simplex, bounds=bounds, xatol=1e-3, fatol=1e6)
    assert ours.evaluations == len(ref_points) == ref.nfev
    np.testing.assert_array_equal(np.array(ours.points), np.array(ref_points))
    np.testing.assert_array_equal(ours.x, ref.x)
    assert ours.fun == ref.fun
    assert (ours.iterations, ours.converged) == (ref.nit, ref.success)
    assert all(0.0 <= p[0] <= 1.0 for p in ours.points)
    assert abs(ours.x[0] - target) <= 1e-3


@pytest.mark.parametrize("bounds", [None, [(-1.5, 1.2), (-0.5, 0.9)]])
def test_nelder_mead_follows_scipy_in_two_dimensions(bounds):
    def rosenbrock(x):
        return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

    simplex = [[-1.0, 0.5], [-0.8, 0.5], [-1.0, 0.7]]
    ref, ref_points = _scipy_points(rosenbrock, simplex, bounds, xatol=1e-6, fatol=1e-8)
    ours = nelder_mead(rosenbrock, simplex, bounds=bounds, xatol=1e-6, fatol=1e-8)
    assert (ours.evaluations, ours.iterations) == (ref.nfev, ref.nit)
    np.testing.assert_array_equal(np.array(ours.points), np.array(ref_points))
    np.testing.assert_array_equal(ours.x, ref.x)


def test_nelder_mead_budget_and_validation():
    result = nelder_mead(lambda x: float(x[0] ** 2), [[-1.0], [2.0]], max_evaluations=5)
    assert result.evaluations == 5 and not result.converged
    with pytest.raises(ValueError):
        nelder_mead(lambda x: 0.0, [[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        nelder_mead(lambda x: 0.0, [[0.0], [1.0]], bounds=[(1.0, 0.0)])


def _exit_wave_and_amplitude(obj):
    psi_exit = exit_wave(pad_center(obj, PADDED, 0.0))
    detector = FresnelPropagator(PADDED, GEOMETRY.fresnel_number, dtype=psi_exit.dtype)
    return psi_exit, detector(psi_exit).abs(), center_slices(PADDED, tuple(obj.shape))


def test_model_fit_error_eq14_equals_eq15_and_vanishes_in_focus(demo):
    psi_exit, sqrt_hologram, fov = _exit_wave_and_amplitude(demo)
    in_focus = model_fit_error(
        project_wavefield(psi_exit), sqrt_hologram, GEOMETRY.fresnel_number, fov
    )
    assert in_focus < 1e-25
    for dz in (-3e-3, 4e-3):
        fr_dz = GEOMETRY.fresnel_number_at(dz)
        projected = project_wavefield(
            FresnelPropagator(PADDED, fr_dz, dtype=psi_exit.dtype)(psi_exit)
        )
        eq15 = model_fit_error(projected, sqrt_hologram, GEOMETRY.defocused(dz).fresnel_number, fov)
        eq14 = model_fit_error_eq14(psi_exit, sqrt_hologram, GEOMETRY.fresnel_number, fr_dz, fov)
        assert eq15 > 1e-8
        assert eq14 == pytest.approx(eq15, rel=1e-9)


def test_image_criteria_invariances():
    g = torch.Generator().manual_seed(5)
    x = torch.rand((48, 40), generator=g, dtype=torch.float64)
    shifted, scaled = x + 3.0, 2.5 * x
    for name, criterion in IMAGE_CRITERIA.items():
        assert criterion(shifted) == pytest.approx(criterion(x), rel=1e-9), name
    assert variance(scaled) == pytest.approx(2.5**2 * variance(x), rel=1e-12)
    assert gradient_sum(scaled) == pytest.approx(2.5 * gradient_sum(x), rel=1e-12)
    assert laplacian(scaled) == pytest.approx(2.5**2 * laplacian(x), rel=1e-12)
    assert tenengrad_of_gradient(scaled) == pytest.approx(
        2.5**0.5 * tenengrad_of_gradient(x), rel=1e-12
    )
    assert gini_of_gradient(scaled) == pytest.approx(gini_of_gradient(x), rel=1e-12)
    assert spectral(torch.full((8, 8), 4.0, dtype=torch.float64)) == pytest.approx(0.0, abs=1e-9)


def test_gradient_statistics_of_a_ramp_and_a_spike():
    axis = torch.arange(32, dtype=torch.float64)
    i, j = torch.meshgrid(axis, axis, indexing="ij")
    ramp = 0.3 * i + 0.4 * j
    assert gini_of_gradient(ramp) == pytest.approx(0.0, abs=1e-6)
    assert tenengrad_of_gradient(ramp) == pytest.approx(0.0, abs=1e-6)
    spike = torch.zeros((32, 32))
    spike[16, 16] = 1.0
    assert gini_of_gradient(spike) > 0.99


def test_prefilter_and_criteria_selection():
    g = torch.Generator().manual_seed(2)
    x = torch.rand((32, 32), generator=g, dtype=torch.float64)
    assert prefilter(x, 0.0) is x
    smooth = prefilter(x, 4.0)
    assert smooth.mean().item() == pytest.approx(x.mean().item(), rel=1e-12)
    assert smooth.std() < 0.5 * x.std()
    assert set(image_criteria(x)) == set(IMAGE_CRITERIA)
    assert set(image_criteria(x, names=("GoG",))) == {"GoG"}
    assert set(FOCUS_EXTREMUM) == {"MFE", *IMAGE_CRITERIA}


def test_curve_helpers():
    np.testing.assert_allclose(normalize_curve([3.0, 1.0, 2.0]), [1.0, 0.0, 0.5])
    np.testing.assert_array_equal(normalize_curve([2.0, 2.0]), [0.0, 0.0])
    assert extremum_offset([-1.0, 0.0, 1.0], [2.0, 1.0, 3.0], "min") == 0.0
    assert extremum_offset([-1.0, 0.0, 1.0], [2.0, 1.0, 3.0], "max") == 1.0
    paper = symmetric_offsets(201, 10e-3)
    assert len(paper) == 201 and paper[100] == 0.0
    np.testing.assert_allclose(paper[[0, 1, -1]], [-10e-3, -9.9e-3, 10e-3], rtol=0, atol=1e-15)
    np.testing.assert_allclose(paper, -paper[::-1], rtol=0, atol=0)
    for points in (2, 1, 40):
        with pytest.raises(ValueError):
            symmetric_offsets(points, 1e-3)


def test_simulated_focus_curve_has_its_mfe_minimum_at_the_focus(demo):
    offsets = np.arange(-3, 4) * 2e-3
    curve = simulated_focus_curve(demo, GEOMETRY, offsets, PADDED)
    assert set(curve) == {"dz01", "MFE", *IMAGE_CRITERIA}
    assert all(len(v) == len(offsets) for v in curve.values())
    mfe = curve["MFE"]
    assert mfe[3] < 1e-25
    assert np.all(np.diff(mfe[:4]) < 0) and np.all(np.diff(mfe[3:]) > 0)


def test_reconstruction_focus_curve(demo):
    hologram = simulate_hologram(demo, GEOMETRY.fresnel_number, padded_shape=PADDED)
    settings = ASRMSettings(stages=(ASRMStage(20, 0.9, 1, 1.0, 4.0, 0.0, 0.0, 1.0),), fade_width=5)
    offsets = [-2e-3, 0.0, 2e-3]
    curve = reconstruction_focus_curve(hologram, GEOMETRY, offsets, settings, phase_truth=demo.real)
    assert set(curve) == {"dz01", "MFE", "seconds", "phase_rmse", *IMAGE_CRITERIA}
    assert all(len(v) == 3 for v in curve.values())
    np.testing.assert_array_equal(curve["dz01"], offsets)
    with pytest.raises(ValueError):
        reconstruction_focus_curve(hologram, GEOMETRY, [0.0], settings, phase_truth=demo.real[:64])


def test_find_focus_on_a_model_objective(monkeypatch):
    z_true = 0.10137
    probed = []

    def fake_asrm(hologram, fresnel_number, settings):
        z01 = z01_from_fresnel_number(
            fresnel_number, GEOMETRY.energy_kev, GEOMETRY.z02, GEOMETRY.pixel_size
        )
        probed.append(z01)
        return SimpleNamespace(final_loss=(z01 - z_true) ** 2)

    monkeypatch.setattr(search_module, "asrm_reconstruct", fake_asrm)
    messages = []
    result = find_focus(torch.zeros((8, 8)), GEOMETRY, log=messages.append)
    assert result.bounds == pytest.approx((0.095, 0.105))
    assert result.z01_guess == GEOMETRY.z01
    assert result.converged
    assert abs(result.z01 - z_true) < 1e-4
    assert result.reconstruction.final_loss == result.mfe == min(e.mfe for e in result.evaluations)
    assert result.n_reconstructions == len(probed) == len(messages)
    assert all(0.095 - 1e-12 <= z <= 0.105 + 1e-12 for z in probed)
    first = result.evaluations[:2]
    assert [e.z01 for e in first] == pytest.approx([0.095, 0.105])
    repeated = [e for e in result.evaluations if not e.reconstructed]
    assert all(e.mfe == search_module.PENALTY for e in repeated)
    assert result.n_evaluations == result.n_reconstructions + len(repeated)
    record = result.to_dict()
    assert record["n_reconstructions"] == result.n_reconstructions
    assert record["bounds"] == list(result.bounds)
    assert len(record["evaluations"]) == result.n_evaluations


@pytest.mark.parametrize("z_true_mm", [97.17899794176402, 101.34605, 104.6])
def test_find_focus_probes_the_points_of_the_reference_search(monkeypatch, z_true_mm):
    """Same points as scipy's Nelder-Mead in millimetres with the reference penalty."""

    def mfe(z_mm):
        return abs(z_mm - z_true_mm) + 0.05 * (z_mm - z_true_mm) ** 2

    probed = []

    def fake_asrm(hologram, fresnel_number, settings):
        z01 = z01_from_fresnel_number(
            fresnel_number, GEOMETRY.energy_kev, GEOMETRY.z02, GEOMETRY.pixel_size
        )
        probed.append(z01)
        return SimpleNamespace(final_loss=mfe(z01 * 1e3))

    monkeypatch.setattr(search_module, "asrm_reconstruct", fake_asrm)
    result = find_focus(torch.zeros((8, 8)), GEOMETRY)
    _, reference = _scipy_points(
        _penalized(lambda x: mfe(float(x[0]))),
        [[95.0], [105.0]],
        [(95.0, 105.0)],
        xatol=0.1,
        fatol=1e6,
    )
    unique = [p[0] for i, p in enumerate(reference) if not any(p[0] == q[0] for q in reference[:i])]
    np.testing.assert_allclose(np.array(probed) * 1e3, unique, rtol=0, atol=1e-9)
    assert result.n_evaluations == len(reference)
    assert min(np.diff(np.sort(probed))) > 1e-9


def test_find_focus_rejects_an_unphysical_search_interval():
    with pytest.raises(ValueError):
        find_focus(torch.zeros((8, 8)), GEOMETRY.with_z01(0.004), confidence=5e-3)
