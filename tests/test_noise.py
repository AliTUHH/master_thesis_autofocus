import numpy as np
import pytest

from holofocus.noise import add_detector_noise


def test_none_returns_copy():
    intensity = np.full((8, 8), 0.7)
    out = add_detector_noise(intensity, np.random.default_rng(0), "none")
    np.testing.assert_array_equal(out, intensity)
    assert out is not intensity


@pytest.mark.parametrize("photons, read_noise", [(100.0, 0.0), (1e4, 0.0), (1e3, 5.0)])
def test_mean_and_variance(photons, read_noise):
    level = 0.8
    intensity = np.full((512, 512), level)
    model = "poisson_gaussian" if read_noise else "poisson"
    out = add_detector_noise(intensity, np.random.default_rng(1), model, photons, read_noise)
    expected_var = level / photons + (read_noise / photons) ** 2
    n = intensity.size
    assert out.mean() == pytest.approx(level, abs=5 * np.sqrt(expected_var / n))
    assert out.var() == pytest.approx(expected_var, rel=5 * np.sqrt(2 / n))


def test_reproducible_with_same_generator_seed():
    intensity = np.random.default_rng(0).uniform(0.5, 1.5, size=(32, 32))
    a = add_detector_noise(intensity, np.random.default_rng(5), "poisson_gaussian", 1e3, 2.0)
    b = add_detector_noise(intensity, np.random.default_rng(5), "poisson_gaussian", 1e3, 2.0)
    np.testing.assert_array_equal(a, b)


def test_validation():
    with pytest.raises(ValueError, match="unknown noise model"):
        add_detector_noise(np.ones(4), np.random.default_rng(0), "gamma")
    with pytest.raises(ValueError, match="photons"):
        add_detector_noise(np.ones(4), np.random.default_rng(0), "poisson", photons=0)
