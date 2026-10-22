import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from holofocus.dataset import LABEL_KEYS, META_KEYS, DatasetConfig, HologramGenerator
from holofocus.forward import object_from_phase, simulate_hologram
from holofocus.geometry import ConeBeamGeometry
from holofocus.phantoms import PHANTOM_TYPES

CI_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "dataset" / "ci.yaml"


@pytest.fixture(scope="module")
def config():
    return DatasetConfig.from_yaml(CI_CONFIG)


@pytest.fixture(scope="module")
def generator(config):
    return HologramGenerator(config, device="cpu")


def test_sample_shapes_and_keys(generator, config):
    sample = generator.sample("train", 0)
    for array in (sample.hologram, sample.clean, sample.phase):
        assert array.shape == config.geometry.fov_shape
        assert array.dtype == np.float32
    assert set(sample.labels) == set(LABEL_KEYS)
    assert set(sample.meta) == set(META_KEYS)
    assert sample.phase.max() <= 0


def test_reproducible_and_order_independent(config, generator):
    other = HologramGenerator(config, device="cpu")
    in_order = [generator.sample("train", i) for i in range(4)]
    alone = other.sample("train", 2)
    np.testing.assert_array_equal(alone.hologram, in_order[2].hologram)
    np.testing.assert_array_equal(alone.phase, in_order[2].phase)
    assert alone.labels == in_order[2].labels
    assert alone.meta == in_order[2].meta
    assert generator.sample("val", 2).labels != alone.labels


def test_labels_are_consistent(config, generator):
    lo, hi = config.sampling.dz01_range
    for i in range(16):
        s = generator.sample("train", i)
        lab = s.labels
        assert lo <= lab["dz01"] <= hi
        assert lab["z01_guess"] == config.geometry.z01_guess
        assert lab["z01"] == pytest.approx(lab["z01_guess"] + lab["dz01"], abs=1e-15)
        truth = ConeBeamGeometry(11.0, lab["z01"], 20.0, config.geometry.pixel_size)
        guess = ConeBeamGeometry(11.0, lab["z01_guess"], 20.0, config.geometry.pixel_size)
        assert lab["fresnel_number"] == pytest.approx(truth.fresnel_number, rel=1e-12)
        assert lab["fresnel_number_guess"] == pytest.approx(guess.fresnel_number, rel=1e-12)
        assert s.meta["effective_pixel_size"] == pytest.approx(truth.effective_pixel_size)
        assert np.sign(lab["fresnel_number"] - lab["fresnel_number_guess"]) == np.sign(lab["dz01"])
        assert PHANTOM_TYPES[s.meta["phantom_type"]] in config.phantoms.types
        p_lo, p_hi = config.noise.photons_range
        assert p_lo <= s.meta["photons"] <= p_hi
        ps_lo, ps_hi = config.phantoms.phase_shift_range
        assert -s.phase.min() == pytest.approx(s.meta["phase_shift"], rel=1e-6)
        assert ps_lo <= s.meta["phase_shift"] <= ps_hi


def test_clean_hologram_is_the_forward_model_of_the_labels(config, generator):
    s = generator.sample("test", 1)
    phase = torch.from_numpy(s.phase.astype(np.float64))
    obj = object_from_phase(phase, config.phantoms.delta_beta)
    expected = simulate_hologram(
        obj, s.labels["fresnel_number"], padded_shape=config.simulation_shape
    )
    np.testing.assert_allclose(s.clean, expected.numpy(), rtol=0, atol=2e-5)


def test_noise_free_model_returns_clean(config):
    quiet = dataclasses.replace(config, noise=dataclasses.replace(config.noise, model="none"))
    s = HologramGenerator(quiet, device="cpu").sample("train", 0)
    np.testing.assert_array_equal(s.hologram, s.clean)
    assert s.meta["photons"] == np.inf


def test_phantom_does_not_depend_on_z01(config):
    """No size cue: the same index gives the same object for any z01 sampling range."""
    shifted = dataclasses.replace(
        config, sampling=dataclasses.replace(config.sampling, dz01_range=(-1e-3, 1e-3))
    )
    a = HologramGenerator(config, device="cpu").sample("train", 5)
    b = HologramGenerator(shifted, device="cpu").sample("train", 5)
    assert a.labels["dz01"] != b.labels["dz01"]
    np.testing.assert_array_equal(a.phase, b.phase)


def test_guess_range(config):
    data = config.to_dict()
    data["geometry"]["z01_guess_range"] = [0.08, 0.3]
    gen = HologramGenerator(DatasetConfig.from_dict(data), device="cpu")
    guesses = [gen.sample("train", i).labels["z01_guess"] for i in range(8)]
    assert all(0.08 <= g <= 0.3 for g in guesses)
    assert len(set(guesses)) == len(guesses)


def test_oversampling_bins_the_fine_intensity(config):
    fine = dataclasses.replace(
        config,
        simulation=dataclasses.replace(config.simulation, oversample=2),
        noise=dataclasses.replace(config.noise, model="none"),
    )
    s = HologramGenerator(fine, device="cpu").sample("train", 3)
    assert s.hologram.shape == config.geometry.fov_shape
    transmission = np.exp(2 * s.phase / config.phantoms.delta_beta)
    # Energy conservation: the mean intensity equals the mean transmission (up to the part
    # of the wave that leaves the field of view).
    assert s.clean.mean() == pytest.approx(transmission.mean(), rel=5e-3)


def test_detector_effects_leave_the_sample_unchanged(config, generator):
    data = config.to_dict()
    data["simulation"]["psf_sigma_range"] = [4.0, 8.0]
    data["noise"]["flat_field_range"] = [0.01, 0.03]
    effects = HologramGenerator(DatasetConfig.from_dict(data), device="cpu")
    for i in range(3):
        base, other = generator.sample("val", i), effects.sample("val", i)
        assert base.meta["psf_sigma"] == base.meta["flat_field_rms"] == 0.0
        assert 4.0 <= other.meta["psf_sigma"] <= 8.0
        assert 0.01 <= other.meta["flat_field_rms"] <= 0.03
        assert other.labels == base.labels
        np.testing.assert_array_equal(other.phase, base.phase)
        for key in ("phantom_type", "phase_shift", "photons"):
            assert other.meta[key] == base.meta[key]
        # PSF of 4-8 detector pixels = 0.25-0.5 pixels of the 16x binned grid: smoother.
        assert np.abs(np.diff(other.clean)).mean() < np.abs(np.diff(base.clean)).mean()
        assert other.clean.mean() == pytest.approx(base.clean.mean(), rel=1e-3)


def test_flat_field_residual_multiplies_the_hologram(config):
    data = config.to_dict()
    data["noise"].update(model="none", flat_field_range=[0.02, 0.02])
    s = HologramGenerator(DatasetConfig.from_dict(data), device="cpu").sample("test", 0)
    residual = s.hologram.astype(np.float64) / s.clean - 1
    assert residual.std() == pytest.approx(0.02, rel=1e-4)
    assert residual.mean() == pytest.approx(0.0, abs=1e-5)


def _changed(config, **phantoms):
    data = config.to_dict()
    data["phantoms"].update(phantoms)
    return DatasetConfig.from_dict(data)


def test_texture_leaves_the_rest_of_the_sample_unchanged(config, generator):
    textured = HologramGenerator(
        _changed(
            config,
            texture_rms_range=[0.05, 0.2],
            texture_slope_range=[2.0, 3.0],
            texture_cutoff=0.1,
        ),
        device="cpu",
    )
    for i in range(3):
        base, other = generator.sample("test", i), textured.sample("test", i)
        assert base.meta["texture_rms"] == base.meta["background_texture_rms"] == 0.0
        assert base.meta["texture_slope"] == 0.0
        assert 0.05 <= other.meta["texture_rms"] <= 0.2
        assert 2.0 <= other.meta["texture_slope"] <= 3.0
        assert other.meta["background_texture_rms"] == 0.0
        assert other.labels == base.labels
        for key in ("phantom_type", "phase_shift", "photons", "psf_sigma", "flat_field_rms"):
            assert other.meta[key] == base.meta[key]
        texture = other.phase.astype(np.float64) - base.phase
        inside = base.phase < 0
        assert not texture[~inside].any()
        # RMS proportional to the square root of the thickness, unit field inside the object.
        root = np.sqrt(base.phase[inside] / -base.meta["phase_shift"])
        field = texture[inside] / (other.meta["texture_rms"] * root)
        assert field.std() == pytest.approx(1.0, rel=0.35)
        assert not np.array_equal(other.hologram, base.hologram)


def test_background_texture_covers_the_field_of_view(config, generator):
    film = HologramGenerator(
        _changed(config, background_texture_rms_range=[0.1, 0.1], texture_cutoff=0.1), "cpu"
    )
    faint = HologramGenerator(
        _changed(config, background_texture_rms_range=[1e-9, 1e-9], texture_cutoff=0.1), "cpu"
    )
    for i in range(2):
        base, other = generator.sample("val", i), film.sample("val", i)
        assert other.labels == base.labels and other.meta["phase_shift"] == base.meta["phase_shift"]
        outside = base.phase == 0
        assert other.phase[outside].std() == pytest.approx(0.1, rel=0.35)
        # A compact object under a film is rendered on the whole simulation grid; without the
        # film that gives the hologram of the object embedded in free space.
        np.testing.assert_allclose(faint.sample("val", i).clean, base.clean, rtol=0, atol=1e-5)


def test_phase_per_pixel_follows_the_fiber_thickness(config):
    uniform = HologramGenerator(_changed(config, types={"fiber": 1.0}), "cpu")
    coupled = HologramGenerator(
        _changed(config, types={"fiber": 1.0}, phase_per_pixel_range=[0.02, 0.02]), "cpu"
    )
    n = config.geometry.fov_shape[0]
    for i in range(3):
        a, b = uniform.sample("train", i), coupled.sample("train", i)
        ratio = b.meta["phase_shift"] / a.meta["phase_shift"]
        np.testing.assert_allclose(b.phase, ratio * a.phase, rtol=1e-5, atol=1e-6)
        assert -b.phase.min() == pytest.approx(b.meta["phase_shift"], rel=1e-6)
        # Largest projected thickness in the field of view, in binned pixels.
        assert 0 < b.meta["phase_shift"] / 0.02 <= 4 * n


def test_textured_configs_need_the_texture_stream(config):
    textured = HologramGenerator(_changed(config, texture_rms_range=[0.1, 0.1]), "cpu")
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="texture_rng"):
        textured.draw(rng, rng)


def test_fiber_samples_cross_the_field_of_view(config):
    data = config.to_dict()
    data["phantoms"]["types"] = {"fiber": 1.0}
    gen = HologramGenerator(DatasetConfig.from_dict(data), device="cpu")
    for i in range(3):
        s = gen.sample("train", i)
        assert PHANTOM_TYPES[s.meta["phantom_type"]] == "fiber"
        assert -s.phase.min() == pytest.approx(s.meta["phase_shift"], rel=1e-6)
        edges = np.concatenate([s.phase[0], s.phase[-1], s.phase[:, 0], s.phase[:, -1]])
        assert edges.min() < 0
        assert np.isfinite(s.clean).all() and 0.5 < s.clean.mean() < 1.0 + 1e-3
