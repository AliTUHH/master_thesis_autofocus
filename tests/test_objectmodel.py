import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from holofocus.dataset import DatasetConfig
from holofocus.learned.objectmodel import (
    DETECTOR_FITS,
    FIT_SCALES,
    LEVEL_NAMES,
    LEVEL_SCALE,
    LOO_FACTOR,
    REFINE_FACTOR,
    TEXTURE_FITS,
    TEXTURE_FLOOR,
    TEXTURE_GRID,
    DetectorFit,
    Texture,
    field_of_view,
    fit_changes,
    fit_objective,
    loo_changes,
    loo_ranges,
    median_features,
    phase_per_pixel,
    refined_grid,
    region_ctf,
    region_weights,
    texture_grid,
)
from holofocus.learned.sim2real import (
    MEASURED_MODELS,
    RadialPower,
    perturbed_config,
    spectral_features,
)
from holofocus.measured import MEASURED_SAMPLES

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "dataset"
FRESNEL_NUMBER = 0.02


def _modulated(shape, seed):
    """White noise with the power ``1 + 3 sin^2(pi f^2 / Fr)``: CTF modulation and nothing
    else."""
    fy, fx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
    gain = np.sqrt(1 + 3 * np.sin(np.pi * (fx**2 + fy**2) / FRESNEL_NUMBER) ** 2)
    noise = np.random.default_rng(seed).standard_normal(shape)
    return np.fft.ifft2(np.fft.fft2(noise) * gain).real


def test_region_ctf_of_the_whole_image_is_that_of_the_spectral_features():
    hologram = _modulated((512, 512), 0)
    features = spectral_features(RadialPower(hologram.shape), hologram, FRESNEL_NUMBER)
    whole = region_ctf(hologram, np.ones_like(hologram), FRESNEL_NUMBER)
    assert features["ctf"] > 0.8
    assert whole["ctf"] == pytest.approx(features["ctf"], abs=1e-9)
    assert whole["ctf_bands"] == pytest.approx(features["ctf_bands"], abs=1e-9)


def test_region_weights_split_object_edge_and_background():
    phase = np.zeros((256, 256))
    phase[:, 100:160] = -2.0  # a fiber, 60 pixels wide
    phase[:, :40] = -1.0  # an object cut by the left border of the image
    w = region_weights(phase, 0.15)
    centre, boundary, far, border = (128, 130), (128, 100), (128, 220), (128, 5)
    assert w["object"][centre] == pytest.approx(1) and w["object"][far] == pytest.approx(0)
    assert w["background"][far] == pytest.approx(1)
    assert w["background"][centre] == pytest.approx(0) and w["background"][boundary] < 0.01
    assert w["edge"][boundary] == pytest.approx(1, abs=1e-3)
    assert w["edge"][centre] == pytest.approx(0, abs=1e-3) == w["edge"][far]
    # the border of the image is no edge of the object
    assert w["edge"][border] == pytest.approx(0, abs=1e-3)
    assert w["object"][border] == pytest.approx(1)


def test_region_ctf_finds_the_modulation_where_it_is():
    phase = np.zeros((512, 512))
    phase[:, :256] = -1.0
    white = np.random.default_rng(2).standard_normal(phase.shape)
    hologram = np.where(phase < 0, _modulated(phase.shape, 1), white)
    w = region_weights(phase, 0.15)
    inside = region_ctf(hologram, w["object"], FRESNEL_NUMBER)
    outside = region_ctf(hologram, w["background"], FRESNEL_NUMBER)
    assert inside["ctf"] > 0.8 and min(inside["ctf_bands"]) > 0.5
    assert abs(outside["ctf"]) < 0.3


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


def test_fit_objective_with_levels():
    measured = {**_features(), "level_mid": -2.0, "level_upper": -5.0}
    assert fit_objective(measured, measured, levels=True) == 0
    # ten terms: the seven of Stage 6 and the three levels; one level off by its scale
    louder = {**measured, "level_upper": -5.0 + LEVEL_SCALE}
    assert fit_objective(louder, measured) == 0
    assert fit_objective(louder, measured, levels=True) == pytest.approx(math.sqrt(1 / 10))
    floor = {**measured, "floor": measured["floor"] - 2 * LEVEL_SCALE}
    assert fit_objective(floor, measured, levels=True) == pytest.approx(2 * math.sqrt(1 / 10))
    assert set(LEVEL_NAMES) == {"level_mid", "level_upper", "floor"}
    with pytest.raises(KeyError):
        fit_objective(_features(), measured, levels=True)


def test_median_features_take_the_bands_one_by_one():
    features = [_features(ctf=c, bands=(c, 2 * c, 0.0, -c)) for c in (0.1, 0.5, 0.2)]
    median = median_features(features)
    assert median["ctf"] == pytest.approx(0.2)
    assert median["ctf_bands"] == pytest.approx([0.2, 0.4, 0.0, -0.2])
    assert "level_mid" not in median
    levels = [
        {**f, "level_mid": v, "level_upper": -v} for f, v in zip(features, (1, 3, 2), strict=True)
    ]
    median = median_features(levels)
    assert (median["level_mid"], median["level_upper"]) == (2.0, -2.0)
    assert "level_mid" not in median_features([*levels, features[0]])


FITS = {
    "spider_hair": Texture(0.0, 0.1, 3.0),
    "tooth": Texture(0.0, 0.0, 2.0),
    "cactus_needle": Texture(0.3, 0.0, 2.5),
    "magnesium_wire": Texture(0.5, 0.0, 2.0),
}
"""Synthetic texture fits with the pattern of the measured holograms: a film around the spider
hair, nothing on the tooth, texture inside needle and wire."""


@pytest.mark.parametrize("name", list(MEASURED_SAMPLES))
def test_loo_sets_know_nothing_of_the_left_out_hologram(name, monkeypatch):
    changes = loo_changes(name, FITS)
    model = MEASURED_MODELS[name]
    monkeypatch.setitem(
        MEASURED_MODELS,
        name,
        dataclasses.replace(
            model,
            photons=7.0,
            flat_field=0.7,
            phase_shift_range=(70.0, 80.0),
            fiber_diameter_range=(0.7, 0.8),
        ),
    )
    monkeypatch.setitem(DETECTOR_FITS, name, DetectorFit(7.0, 0.7))
    assert loo_changes(name, {**FITS, name: Texture(7.0, 7.0, 0.7)}) == changes
    # only delta / beta, known from the composition, is the sample's own
    assert changes["phantoms"]["delta_beta"] == model.delta_beta


@pytest.mark.parametrize("name", list(MEASURED_SAMPLES))
def test_loo_ranges_contain_the_other_holograms(name):
    r = loo_ranges(name, FITS)
    assert r.others == tuple(s for s in MEASURED_SAMPLES if s != name)
    for other in r.others:
        m = MEASURED_MODELS[other]
        assert r.photons[0] < m.photons < r.photons[1]
        assert r.phase_per_pixel[0] < phase_per_pixel(other) < r.phase_per_pixel[1]
        assert r.fiber_diameter[0] < m.fiber_diameter_range[0]
        assert m.fiber_diameter_range[1] <= r.fiber_diameter[1] <= 1.0
        if DETECTOR_FITS[other] is not None:
            assert r.psf_sigma[0] < DETECTOR_FITS[other].psf_sigma < r.psf_sigma[1]
            assert r.flat_field[0] < m.flat_field < r.flat_field[1]
        texture = FITS[other]
        assert r.texture_rms[0] < TEXTURE_FLOOR and texture.amplitude < r.texture_rms[1]
        if texture.amplitude > TEXTURE_FLOOR:
            assert r.texture_slope[0] < texture.slope < r.texture_slope[1]


def test_loo_textures_pool_object_and_film():
    # none of the other three has a film, but the spider hair may get one as strong as the
    # texture inside the Mg wire; the untextured tooth adds no slope
    spider = loo_changes("spider_hair", FITS)["phantoms"]
    assert spider["background_texture_rms_range"] == spider["texture_rms_range"]
    assert spider["texture_rms_range"] == [TEXTURE_FLOOR / LOO_FACTOR, 0.5 * LOO_FACTOR]
    assert spider["texture_slope_range"] == [1.5, 3.0]
    assert loo_ranges("magnesium_wire", FITS).texture_rms[1] == 0.3 * LOO_FACTOR
    # the Mg wire has no detector fit: three PSF fits for it, two for the others
    assert loo_ranges("magnesium_wire", FITS).psf_sigma == (1.2, 2.01)
    assert loo_ranges("tooth", FITS).flat_field_cutoff == pytest.approx((0.179 + 0.192) / 2, 1e-2)


@pytest.mark.parametrize("name", list(MEASURED_SAMPLES))
def test_texture_fits_are_the_best_textures_of_the_fit_records(name):
    record = json.loads((ROOT / "results" / "objectmodel" / f"fit-{name}.json").read_text())
    assert TEXTURE_FITS[name] == Texture(**record["best"]["texture"])
    assert record["best"]["objective"] == min(e["objective"] for e in record["settings"])
    refined = refined_grid(Texture(**record["best_coarse"]))
    assert len(record["settings"]) == len(texture_grid()) + len(refined)


@pytest.mark.parametrize("name", list(MEASURED_SAMPLES))
def test_loo_configs_are_the_matched_ones_with_the_loo_changes(name):
    matched = DatasetConfig.from_yaml(CONFIGS / f"matched-{name}.yaml")
    small = DatasetConfig.from_yaml(CONFIGS / f"loo-{name}.yaml")
    large = DatasetConfig.from_yaml(CONFIGS / f"loo-{name}-large.yaml")
    expected = perturbed_config(matched, loo_changes(name), f"loo-{name}")
    assert small == dataclasses.replace(expected, seed=small.seed) and small.seed != matched.seed
    assert small.phantoms.textured and small.splits == matched.splits
    assert large.name == f"loo-{name}-large" and large.splits == {"train": 50000}
    for section in ("geometry", "sampling", "simulation", "phantoms", "noise", "storage"):
        assert getattr(large, section) == getattr(small, section), section
