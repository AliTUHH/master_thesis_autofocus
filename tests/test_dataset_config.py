import math
from pathlib import Path

import pytest
import yaml

from holofocus.dataset import DatasetConfig
from holofocus.geometry import P05_DETECTOR_SHAPE, P05_PIXEL_SIZE
from holofocus.measured import MEASURED_SAMPLES
from holofocus.propagation import next_fast_len

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "dataset"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_repository_configs_load(path):
    config = DatasetConfig.from_yaml(path)
    assert config.name == path.stem
    lo, hi = config.z01_bounds
    assert 0 < lo < hi < config.geometry.z02


def test_yaml_round_trip():
    config = DatasetConfig.from_yaml(CONFIG_DIR / "cpu-4x.yaml")
    assert DatasetConfig.from_dict(yaml.safe_load(config.to_yaml())) == config


@pytest.mark.parametrize("name", ["cpu-4x-extra", "cpu-4x-large"])
def test_extra_training_data_has_the_physics_of_cpu_4x(name):
    main = DatasetConfig.from_yaml(CONFIG_DIR / "cpu-4x.yaml")
    extra = DatasetConfig.from_yaml(CONFIG_DIR / f"{name}.yaml")
    for section in ("geometry", "sampling", "simulation", "phantoms", "noise"):
        assert getattr(extra, section) == getattr(main, section), section
    assert list(extra.splits) == ["train"] and not extra.storage.store_phase


def test_calibration_split_has_the_physics_of_cpu_4x():
    main = DatasetConfig.from_yaml(CONFIG_DIR / "cpu-4x.yaml")
    calibration = DatasetConfig.from_yaml(CONFIG_DIR / "cpu-4x-calibration.yaml")
    for section in ("geometry", "sampling", "simulation", "phantoms", "noise"):
        assert getattr(calibration, section) == getattr(main, section), section
    assert calibration.splits == {"calibration": 2000} and not calibration.storage.store_phase


def test_repository_datasets_have_their_own_seeds():
    seeds = [DatasetConfig.from_yaml(path).seed for path in sorted(CONFIG_DIR.glob("*.yaml"))]
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("name", list(MEASURED_SAMPLES))
def test_matched_datasets_follow_the_measured_setup(name):
    sample = MEASURED_SAMPLES[name]
    config = DatasetConfig.from_yaml(CONFIG_DIR / f"matched-{name}.yaml")
    g = config.geometry
    assert (g.energy_kev, g.z02) == (sample.energy_kev, sample.z02)
    assert g.z01_guess == pytest.approx(sample.z01_guess, abs=1e-12)
    assert g.z01_guess_range is None
    assert config.sampling.dz01_range == pytest.approx((-sample.confidence, sample.confidence))
    # Detector path of the measured runs: cutting band removed, 4x block mean (load_measured).
    n = P05_DETECTOR_SHAPE[0] - 2 * sample.cutting_band
    assert (g.detector_shape, g.binning, g.detector_pixel_size) == ((n, n), 4, P05_PIXEL_SIZE)
    guess = sample.geometry(binning=4)
    assert g.at(g.z01_guess).fresnel_number == pytest.approx(guess.fresnel_number, rel=1e-12)
    # Everything else as in the Stage 2/4 data.
    main = DatasetConfig.from_yaml(CONFIG_DIR / "cpu-4x.yaml")
    for section in ("simulation", "phantoms", "noise"):
        assert getattr(config, section) == getattr(main, section), section


@pytest.mark.parametrize("name", list(MEASURED_SAMPLES))
def test_large_matched_datasets_have_the_physics_of_the_cpu_sets(name):
    cpu = DatasetConfig.from_yaml(CONFIG_DIR / f"matched-{name}.yaml")
    large = DatasetConfig.from_yaml(CONFIG_DIR / f"matched-{name}-large.yaml")
    assert large.name == f"matched-{name}-large"
    for section in ("geometry", "sampling", "simulation", "phantoms", "noise"):
        assert getattr(large, section) == getattr(cpu, section), section
    assert large.splits == {"train": 50000} and not large.storage.store_phase


def test_exponent_without_dot_is_a_number():
    # YAML 1.1 (PyYAML) reads 1e5 as a string; the loader converts it.
    config = DatasetConfig.from_dict(yaml.safe_load("noise: {photons_range: [1e3, 1e5]}"))
    assert config.noise.photons_range == (1000.0, 100000.0)
    assert DatasetConfig.from_dict({"splits": {"train": 1e2}}).splits == {"train": 100}


def test_defaults_match_stage2_setup():
    config = DatasetConfig()
    assert config.geometry.fov_shape == (512, 512)
    assert config.geometry.pixel_size == pytest.approx(26e-6)
    assert config.geometry.at(0.100).fresnel_number == pytest.approx(1.5069e-3, rel=1e-4)
    assert config.z01_bounds == pytest.approx((0.095, 0.105))


@pytest.mark.parametrize("oversample", [1, 2])
def test_simulation_shape_covers_kernel_and_margin(oversample):
    config = DatasetConfig.from_dict({"simulation": {"oversample": oversample}})
    fr_min = config.geometry.at(config.z01_bounds[0]).fresnel_number / oversample**2
    for n_fov, n_sim in zip(config.geometry.fov_shape, config.simulation_shape, strict=True):
        assert n_sim >= max(n_fov * oversample + 1 / fr_min, 1 / fr_min)
        assert n_sim == next_fast_len(n_sim)


def test_next_fast_len_is_smallest_7_smooth_number():
    def smooth(n):
        for p in (2, 3, 5, 7):
            while n % p == 0:
                n //= p
        return n == 1

    for n in range(1, 400):
        m = next_fast_len(n)
        assert m >= n and smooth(m)
        assert not any(smooth(k) for k in range(n, m))
    assert next_fast_len(math.pi) == 4


@pytest.mark.parametrize(
    "data, message",
    [
        ({"colour": 1}, "unknown keys"),
        ({"noise": {"photons": 1}}, "unknown keys"),
        ({"geometry": {"binning": 3}}, "must divide"),
        ({"geometry": {"binning": "four"}}, "cannot read"),
        ({"geometry": {"z01_guess": 19.999}}, "between the focus and the detector"),
        ({"sampling": {"dz01_range": [5e-3, -5e-3]}}, "increasing"),
        ({"sampling": {"dz01_range": [1e-3]}}, "expected 2 values"),
        ({"phantoms": {"types": {"square": 1}}}, "unknown phantom types"),
        ({"phantoms": {"types": {"ball": 0}}}, "positive sum"),
        ({"phantoms": {"size_range": [0.5, 0.95]}}, "size_range"),
        ({"noise": {"model": "gamma"}}, "unknown noise model"),
        ({"storage": {"store_phase": "yes"}}, "cannot read"),
        ({"storage": {"compression": "zstd"}}, "compression"),
        ({"simulation": {"precision": "half"}}, "precision"),
        ({"simulation": {"psf_sigma_range": [1.0, 0.5]}}, "psf_sigma_range"),
        ({"simulation": {"psf_sigma_range": [-1.0, 0.5]}}, "psf_sigma_range"),
        ({"noise": {"flat_field_range": [0.02, 0.01]}}, "flat_field_range"),
        ({"noise": {"flat_field_cutoff": 0.0}}, "flat_field_cutoff"),
        ({"phantoms": {"fiber_diameter_range": [0.0, 0.5]}}, "fiber_diameter_range"),
        ({"phantoms": {"types": {"fibre": 1}}}, "unknown phantom types"),
        ({"splits": {"train": -1}}, "split sizes"),
    ],
)
def test_validation_errors(data, message):
    with pytest.raises(ValueError, match=message):
        DatasetConfig.from_dict(data)
