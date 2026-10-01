"""Tests for the model-based autofocus baseline and the classical sharpness-metric fallback."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from src.baseline.model_based_autofocus import (
    DEFAULT_FOCUS_PRESET,
    DEFAULT_QUALITY_PRESET,
    PRESETS,
    SMOKE_PRESET,
    AutofocusResult,
    Geometry,
    RecoPreset,
    classical_autofocus,
    default_fading_width_px,
    fr_search_bounds,
    get_preset,
)
from src.utils.fresnel import hologram_amplitude
from src.utils.physics import fresnel_number, z01_from_fresnel_number

torch.set_num_threads(2)


# --------------------------------------------------------------------------------------------------
# presets and geometry
# --------------------------------------------------------------------------------------------------
def test_presets_are_consistent() -> None:
    assert DEFAULT_FOCUS_PRESET in PRESETS and DEFAULT_QUALITY_PRESET in PRESETS and SMOKE_PRESET in PRESETS
    assert PRESETS[DEFAULT_FOCUS_PRESET].purpose == "focus"
    assert PRESETS[DEFAULT_QUALITY_PRESET].purpose == "quality"
    for name, preset in PRESETS.items():
        assert preset.name == name
        assert preset.purpose in ("focus", "quality")
        assert preset.padding_factor >= 1
        assert all(s.iterations >= 1 and s.down_sampling_factor >= 1 for s in preset.stages)
        assert all(np.isfinite(s.update_rate) and s.l2_absorption >= 0 for s in preset.stages)
    p05 = PRESETS["p05_default_2048"]
    assert p05.iterations == [700, 300, 500] and p05.padding_factor == 4.0
    assert [s.down_sampling_factor for s in p05.stages] == [16, 4, 4]


def test_get_preset_validation_and_scaling() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        get_preset("does_not_exist")
    with pytest.raises(ValueError):
        get_preset(SMOKE_PRESET, 0.0)
    base = get_preset(DEFAULT_FOCUS_PRESET)
    assert base is PRESETS[DEFAULT_FOCUS_PRESET]
    half = get_preset(DEFAULT_FOCUS_PRESET, 0.5)
    assert isinstance(half, RecoPreset)
    assert half.iterations == [150, 150]
    assert half.name == base.name and half.stages[0].fwhm_object == base.stages[0].fwhm_object
    tiny = get_preset(SMOKE_PRESET, 1e-3)
    assert tiny.iterations == [1, 1]
    assert get_preset(half, 2.0).iterations == [300, 300]
    assert base.grid_sizes(256) == [256, 512]
    assert isinstance(base.to_dict()["stages"][0]["fwhm_object"], str)


def test_geometry_delegates_to_physics() -> None:
    geom = Geometry(energy_kev=11.0, px_mm=0.052, z02_mm=20000.0)
    for z01 in (50.0, 100.0, 250.0, 400.0):
        fr = geom.fresnel(z01)
        assert fr == pytest.approx(fresnel_number(z01, 20000.0, 11.0, 0.052))
        assert geom.z01_from_fresnel(fr) == pytest.approx(z01_from_fresnel_number(fr, 20000.0, 11.0, 0.052))
        assert geom.z01_from_fresnel(fr) == pytest.approx(z01, rel=1e-6)
    lo, hi, centre, half = geom.z01_interval((0.01, 0.02))
    assert lo < centre < hi and half == pytest.approx(0.5 * (hi - lo))
    assert geom.fresnel(lo) == pytest.approx(0.01, rel=1e-6) and geom.fresnel(hi) == pytest.approx(0.02, rel=1e-6)
    with pytest.raises(ValueError):
        geom.z01_interval((0.02, 0.01))


def test_fr_search_bounds_and_fading_width() -> None:
    assert fr_search_bounds(0.01, 50.0) == pytest.approx((0.005, 0.015))
    with pytest.raises(ValueError):
        fr_search_bounds(0.01, 100.0)
    assert default_fading_width_px(2048) == 80 and default_fading_width_px(256) == 80
    assert default_fading_width_px(64) == 20 and default_fading_width_px(8) == 4


# --------------------------------------------------------------------------------------------------
# classical fallback on a synthetic weak object
# --------------------------------------------------------------------------------------------------
def _broadband_phase_object(size: int, seed: int = 0, phi_max: float = 0.3, sigma_px: float = 1.0, taper: int = 8) -> np.ndarray:
    """Weak phase object with fine structure (spectrum well beyond the first CTF zero), tapered to zero at the edges."""
    rng = np.random.default_rng(seed)
    phi = gaussian_filter(rng.standard_normal((size, size)), sigma_px)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(taper) / taper))
    window = np.ones(size)
    window[:taper] = ramp
    window[-taper:] = ramp[::-1]
    phi = -np.abs(phi) * window[:, None] * window[None, :]
    return (phi * phi_max / np.abs(phi).max()).astype(np.float32)


def _synthetic_hologram(size: int, fr: float, noise: float = 0.01, seed: int = 1) -> np.ndarray:
    obj = torch.as_tensor(_broadband_phase_object(size)).to(torch.complex64)
    amplitude = hologram_amplitude(obj, fr, pad_factor=2).numpy()
    rng = np.random.default_rng(seed)
    return (amplitude + noise * rng.standard_normal(amplitude.shape)).astype(np.float32)


@pytest.mark.parametrize("fr", [5e-3, 2e-2])
@pytest.mark.parametrize("metric", ["tv", "var"])
def test_classical_autofocus_on_synthetic_weak_object(fr: float, metric: str) -> None:
    hologram = _synthetic_hologram(128, fr)
    geom = Geometry(energy_kev=11.0, px_mm=0.052, z02_mm=20000.0)
    result = classical_autofocus(hologram, fr_search_bounds(fr, 50.0), metric=metric, n_grid=21, geom=geom)
    assert isinstance(result, AutofocusResult)
    assert result.method == f"classical_{metric}"
    assert abs(result.fr_est / fr - 1.0) < 0.02, f"{metric}: fr_est {result.fr_est:.4e} vs {fr:.4e}"
    assert result.n_evals >= 21 and result.n_evals == len(result.loss_history) == len(result.fr_history) == len(result.z01_history)
    assert result.z01_est_mm == pytest.approx(geom.z01_from_fresnel(result.fr_est))
    assert result.fr_bounds == pytest.approx(fr_search_bounds(fr, 50.0))
    assert result.runtime_s > 0 and result.preset is None


def test_classical_autofocus_validation() -> None:
    hologram = _synthetic_hologram(64, 2e-2)
    with pytest.raises(ValueError, match="metric"):
        classical_autofocus(hologram, (0.01, 0.03), metric="nope")
    with pytest.raises(ValueError, match="fr_bounds"):
        classical_autofocus(hologram, (0.03, 0.01))
    without_geom = classical_autofocus(hologram, (0.01, 0.03), n_grid=5)
    assert np.isnan(without_geom.z01_est_mm) and all(np.isnan(z) for z in without_geom.z01_history)


# --------------------------------------------------------------------------------------------------
# HoloWizard end-to-end (needs holowizard.core)
# --------------------------------------------------------------------------------------------------
def _first_test_sample(tiny_dataset_dir):
    from src.data.forge_samples import read_forge_samples

    sample = read_forge_samples([tiny_dataset_dir / "test.hdf5"], n_per_file=1)[0]
    assert sample.phantom is not None, "the tiny dataset must store images/phantoms"
    return sample


@pytest.mark.slow
def test_model_based_autofocus_smoke(tiny_dataset_dir) -> None:
    pytest.importorskip("holowizard")
    from src.baseline.model_based_autofocus import METHOD_HOLOWIZARD, model_based_autofocus

    sample = _first_test_sample(tiny_dataset_dir)
    geom = Geometry.from_sample(sample)
    bounds = fr_search_bounds(sample.fr, 50.0)
    result = model_based_autofocus(sample.hologram_amplitude, geom, bounds, preset=SMOKE_PRESET, threads=2)
    assert result.method == METHOD_HOLOWIZARD and result.preset == SMOKE_PRESET
    assert bounds[0] <= result.fr_est <= bounds[1]
    assert result.n_evals >= 3
    assert result.n_evals == len(result.loss_history) == len(result.z01_history) == len(result.fr_history)
    assert all(np.isfinite(result.loss_history)) and all(v >= 0 for v in result.loss_history)
    z_lo, z_hi, _, _ = geom.z01_interval(bounds)
    assert all(z_lo - 1e-6 <= z <= z_hi + 1e-6 for z in result.z01_history)
    assert result.fr_est == pytest.approx(geom.fresnel(result.z01_est_mm))
    assert result.fr_history == pytest.approx([geom.fresnel(z) for z in result.z01_history])
    assert result.runtime_s > 0


@pytest.mark.slow
def test_reconstruction_is_rotated_back_into_phantom_orientation(tiny_dataset_dir) -> None:
    """The core rotates the measurement by 90 degrees; ``reconstruct`` must undo it (compare with images/phantoms)."""
    pytest.importorskip("holowizard")
    from src.eval.downstream import reconstruct

    sample = _first_test_sample(tiny_dataset_dir)
    geom = Geometry.from_sample(sample)
    reco = reconstruct(sample.hologram_amplitude, geom, sample.fr, preset=DEFAULT_QUALITY_PRESET, iterations_scale=0.3, threads=2)
    assert reco.phase.shape == sample.shape and reco.preset == DEFAULT_QUALITY_PRESET
    assert reco.losses.size == sum(get_preset(DEFAULT_QUALITY_PRESET, 0.3).iterations)
    gt = sample.gt_phase.astype(np.float64)
    corr_back_rotated = np.corrcoef(reco.phase.ravel(), gt.ravel())[0, 1]
    corr_raw = np.corrcoef(np.rot90(reco.phase, k=1).ravel(), gt.ravel())[0, 1]  # orientation as returned by the core
    assert corr_back_rotated > 0.4, corr_back_rotated
    assert corr_back_rotated > corr_raw + 0.3, (corr_back_rotated, corr_raw)
    assert reco.phase.min() < 0 and reco.phase.max() <= 1e-6  # phase constraint: <= 0
