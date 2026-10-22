import json
import math
from pathlib import Path

import pytest

from holofocus.dataset import DatasetConfig
from holofocus.learned.objectmodel import (
    DETECTOR_FITS,
    FIT_SCALES,
    REFINE_FACTOR,
    TEXTURE_GRID,
    Texture,
    field_of_view,
    fit_changes,
    fit_objective,
    median_features,
    phase_per_pixel,
    refined_grid,
    texture_grid,
)
from holofocus.learned.sim2real import MEASURED_MODELS, perturbed_config
from holofocus.measured import MEASURED_SAMPLES

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "dataset"


def test_detector_fits_are_the_medians_of_the_corner_fits():
    noise = json.loads((ROOT / "results" / "sim2real" / "noise.json").read_text())["samples"]
    assert set(DETECTOR_FITS) == set(MEASURED_SAMPLES) and DETECTOR_FITS["magnesium_wire"] is None
    for name, fit in DETECTOR_FITS.items():
        if fit is not None:
            assert fit.psf_sigma == pytest.approx(noise[name]["psf_sigma_median"], abs=5e-3)
            assert fit.flat_field_cutoff == pytest.approx(
                noise[name]["flat_field_cutoff_median"], abs=5e-4
            )


@pytest.mark.parametrize("name", list(MEASURED_SAMPLES))
def test_phase_per_pixel_gives_the_measured_phase_at_the_mean_width(name):
    matched = DatasetConfig.from_yaml(CONFIGS / f"matched-{name}.yaml")
    assert field_of_view(name) == matched.geometry.fov_shape[0] == matched.geometry.fov_shape[1]
    model = MEASURED_MODELS[name]
    (p0, p1), (d0, d1) = model.phase_shift_range, model.fiber_diameter_range
    thickness = math.sqrt(d0 * d1) * field_of_view(name)
    assert phase_per_pixel(name) * thickness == pytest.approx(math.sqrt(p0 * p1))
    # the widths of the measured object then span about its phase range
    assert p0 / 2 < phase_per_pixel(name) * d0 * field_of_view(name) < p0 * 2
    assert p1 / 2 < phase_per_pixel(name) * d1 * field_of_view(name) < p1 * 2


@pytest.mark.parametrize("name", list(MEASURED_SAMPLES))
def test_fit_configs_have_the_measured_detector_object_and_the_texture(name):
    matched = DatasetConfig.from_yaml(CONFIGS / f"matched-{name}.yaml")
    texture = Texture(0.1, 0.03, 2.5)
    config = perturbed_config(matched, fit_changes(name, texture), f"fit-{name}")
    ph, model = config.phantoms, MEASURED_MODELS[name]
    assert config.seed == matched.seed and config.geometry == matched.geometry
    assert ph.types == {"fiber": 1.0} and ph.delta_beta == model.delta_beta
    assert ph.phase_per_pixel_range == (phase_per_pixel(name),) * 2
    assert ph.texture_rms_range == (0.1, 0.1) and ph.background_texture_rms_range == (0.03, 0.03)
    assert ph.texture_slope_range == (2.5, 2.5)
    assert config.noise.photons_range == (model.photons,) * 2
    assert config.noise.flat_field_range == (model.flat_field,) * 2
    detector = DETECTOR_FITS[name]
    if detector is not None:
        assert config.simulation.psf_sigma_range == (detector.psf_sigma,) * 2
        assert config.noise.flat_field_cutoff == detector.flat_field_cutoff
    untextured = perturbed_config(matched, fit_changes(name, Texture(0, 0, 3)), "fit")
    assert not untextured.phantoms.textured


def test_texture_grid_has_the_untextured_setting_once_and_first():
    grid = texture_grid()
    n_rms, n_background, n_slope = (len(v) for v in TEXTURE_GRID.values())
    assert len(grid) == len(set(grid)) == 1 + (n_rms * n_background - 1) * n_slope
    assert grid[0].amplitude == 0 and all(t.amplitude > 0 for t in grid[1:])


def test_refined_grid_brackets_the_best_texture():
    best = Texture(0.1, 0.0, 3.0)
    refined = refined_grid(best)
    assert best not in refined and len(refined) == 8
    assert {t.background_rms for t in refined} == {0.0}
    assert {t.slope for t in refined} == {2.75, 3.0, 3.25}
    rms = sorted({t.rms for t in refined})
    assert rms[0] == pytest.approx(0.1 / REFINE_FACTOR, rel=1e-2)
    assert rms[2] == pytest.approx(0.1 * REFINE_FACTOR, rel=1e-2)
    assert len(refined_grid(Texture(0.3, 0.1, 2.0))) == 26
    assert refined_grid(Texture(0.0, 0.0, 2.0)) == []


def _features(ctf=0.3, bands=(0.4, 0.3, 0.2, 0.1), upper_snr=1.0, slope=-18.0):
    return {
        "floor": -8.0,
        "upper_snr": upper_snr,
        "low_excess": 4.0,
        "slope": slope,
        "ctf": ctf,
        "ctf_bands": list(bands),
    }


def test_fit_objective_scales_each_feature():
    measured = _features()
    assert fit_objective(measured, measured) == 0
    # one of seven terms off by one scale
    off_band = _features(bands=(0.4, 0.3 + FIT_SCALES["ctf"], 0.2, 0.1))
    assert fit_objective(off_band, measured) == pytest.approx(math.sqrt(1 / 7))
    off_slope = _features(slope=-18.0 + 2 * FIT_SCALES["slope"])
    assert fit_objective(off_slope, measured) == pytest.approx(2 * math.sqrt(1 / 7))
    # floor and low_excess do not count
    assert fit_objective({**measured, "floor": 0.0, "low_excess": 0.0}, measured) == 0
    assert fit_objective(_features(ctf=math.nan), measured) == math.inf


def test_median_features_take_the_bands_one_by_one():
    features = [_features(ctf=c, bands=(c, 2 * c, 0.0, -c)) for c in (0.1, 0.5, 0.2)]
    median = median_features(features)
    assert median["ctf"] == pytest.approx(0.2)
    assert median["ctf_bands"] == pytest.approx([0.2, 0.4, 0.0, -0.2])
