import numpy as np
import pytest

from holofocus.phantoms import (
    PHANTOM_TYPES,
    Primitive,
    ball,
    ellipse,
    ellipsoid,
    random_phantom,
    rasterize,
    render,
)


def test_render_matches_rasterize_of_weighted_sum():
    shape = (64, 80)
    parts = [
        Primitive(ball(-10.0, 5.0, 12.0), -10.0, 5.0, 12.0, weight=1.0),
        Primitive(ellipse(15.0, -8.0, 9.0, 4.0, 0.4), 15.0, -8.0, 9.0, weight=0.5),
        Primitive(ellipsoid(3.0, 3.0, 7.0, 5.0, 1.0), 3.0, 3.0, 7.0, weight=0.3),
    ]

    def weighted(x, y):
        return sum(p.weight * p.fn(x, y) for p in parts)

    np.testing.assert_allclose(
        render(parts, shape, supersample=3), rasterize(weighted, shape, supersample=3), atol=1e-12
    )


def test_ellipsoid_peak_and_support():
    values = ellipsoid(0.0, 0.0, 4.0, 2.0)(np.array([0.0, 3.9, 0.0, 4.1]), np.zeros(4))
    assert values[0] == 1.0
    assert 0 < values[1] < 1
    assert values[3] == 0.0


@pytest.mark.parametrize("kind", PHANTOM_TYPES)
def test_random_phantom_range_and_border(kind):
    shape = (96, 96)
    margin = 0.05
    for seed in range(5):
        t = random_phantom(np.random.default_rng(seed), kind, shape, margin=margin)
        assert t.shape == shape
        assert t.min() >= 0
        assert t.max() == pytest.approx(1.0)
        border = int(margin * shape[0]) - 1
        assert np.all(t[:border] == 0) and np.all(t[-border:] == 0)
        assert np.all(t[:, :border] == 0) and np.all(t[:, -border:] == 0)


def test_random_phantom_reproducible_and_seed_dependent():
    a = random_phantom(np.random.default_rng(3), "cell", (64, 64))
    b = random_phantom(np.random.default_rng(3), "cell", (64, 64))
    c = random_phantom(np.random.default_rng(4), "cell", (64, 64))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_size_range_controls_object_extent():
    small = random_phantom(np.random.default_rng(0), "ball", (128, 128), size_range=(0.2, 0.2))
    large = random_phantom(np.random.default_rng(0), "ball", (128, 128), size_range=(0.6, 0.6))
    # Area of a disc of diameter 0.2 * 128 and 0.6 * 128 pixels.
    assert (small > 0).sum() == pytest.approx(np.pi * (0.1 * 128) ** 2, rel=0.1)
    assert (large > 0).sum() == pytest.approx(np.pi * (0.3 * 128) ** 2, rel=0.05)


def test_unknown_phantom_type():
    with pytest.raises(ValueError, match="unknown phantom type"):
        random_phantom(np.random.default_rng(0), "square", (32, 32))
