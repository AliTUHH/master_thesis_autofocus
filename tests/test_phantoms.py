import math

import numpy as np
import pytest
import torch

from holofocus.phantoms import (
    COMPACT_TYPES,
    PHANTOM_TYPES,
    Primitive,
    ball,
    ellipse,
    ellipsoid,
    fiber_thickness,
    pixel_coordinates,
    random_phantom,
    rasterize,
    render,
    texture_field,
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


@pytest.mark.parametrize("kind", COMPACT_TYPES)
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


def test_stored_type_indices_are_stable():
    assert PHANTOM_TYPES[:4] == COMPACT_TYPES == ("triangle", "cell", "ball", "mixed")
    assert PHANTOM_TYPES[4] == "fiber"


@pytest.mark.parametrize("angle, radius", [(0.0, 5.3), (0.37, 7.7), (math.pi / 2, 2.2), (1.1, 0.6)])
def test_fiber_is_the_pixel_average_of_a_cylinder(angle, radius):
    shape, center, s = (40, 52), (1.3, -2.1), 32
    xs = np.arange(shape[1]) + 0.5 - shape[1] / 2
    ys = np.arange(shape[0]) + 0.5 - shape[0] / 2
    x, y = pixel_coordinates(shape, 1.0, s)
    d = -(x - center[0]) * math.sin(angle) + (y - center[1]) * math.cos(angle)
    fine = 2 * np.sqrt(np.clip(radius**2 - d**2, 0, None))
    expected = fine.reshape(shape[0], s, shape[1], s).mean(axis=(1, 3))
    got = fiber_thickness(xs, ys, center, angle, radius)
    np.testing.assert_allclose(got, expected, atol=1e-3 * expected.max())
    if abs(math.cos(angle)) > 0.1:
        # The pixel average keeps the cross-section pi R^2 per unit length of the axis.
        column = got[:, shape[1] // 2].sum() * abs(math.cos(angle))
        assert column == pytest.approx(np.pi * radius**2, rel=1e-9)


def test_tapered_fiber_ends_in_a_tip():
    xs = np.arange(64) + 0.5 - 32
    t = fiber_thickness(xs, np.array([0.5]), (0.0, 0.5), 0.0, 4.0, taper=1 / 16)
    assert t[0, 0] == 0.0 and t[0, :10].max() == 0.0  # radius 0 for s <= -16
    assert t[0, -1] == pytest.approx(2 * 8.0, rel=1e-2)  # radius doubled (clipped) at s = 31.5


def test_fibers_cross_the_field_of_view_into_the_canvas():
    shape, canvas = (64, 64), (160, 160)
    for seed in range(6):
        t = random_phantom(np.random.default_rng(seed), "fiber", shape, canvas=canvas)
        assert t.shape == canvas and t.min() >= 0
        fov = t[48:112, 48:112]
        assert fov.max() == pytest.approx(1.0)
        border = np.concatenate([fov[0], fov[-1], fov[:, 0], fov[:, -1]])
        assert border.max() > 0  # a fiber leaves the field of view
        edge = np.concatenate([t[0], t[-1], t[:, 0], t[:, -1]])
        assert edge.max() < 1e-12  # faded out at the edge of the canvas
    a = random_phantom(np.random.default_rng(3), "fiber", shape, canvas=canvas)
    b = random_phantom(np.random.default_rng(3), "fiber", shape, canvas=canvas)
    np.testing.assert_array_equal(a, b)


def test_compact_types_on_a_canvas_are_zero_padded():
    plain = random_phantom(np.random.default_rng(7), "cell", (48, 48))
    padded = random_phantom(np.random.default_rng(7), "cell", (48, 48), canvas=(101, 100))
    assert padded.shape == (101, 100)
    np.testing.assert_array_equal(padded[26:74, 26:74], plain)
    assert padded.sum() == pytest.approx(plain.sum())


def test_fiber_thickness_in_pixels():
    shape, canvas = (64, 64), (160, 160)
    for seed in range(4):
        pixels = random_phantom(
            np.random.default_rng(seed), "fiber", shape, canvas=canvas, normalize=False
        )
        unit = random_phantom(np.random.default_rng(seed), "fiber", shape, canvas=canvas)
        peak = pixels[48:112, 48:112].max()
        np.testing.assert_allclose(pixels / peak, unit, rtol=1e-12, atol=1e-15)
        # Diameters of 3-100 % of the field of view; the taper at most doubles a radius.
        assert 0 < peak <= 2 * 2 * 64
    with pytest.raises(ValueError, match="only fibers"):
        random_phantom(np.random.default_rng(0), "ball", shape, normalize=False)


def _radial_power(field: np.ndarray, bins: np.ndarray) -> np.ndarray:
    f = np.hypot(*np.meshgrid(np.fft.fftfreq(field.shape[0]), np.fft.fftfreq(field.shape[1])))
    power = np.abs(np.fft.fft2(field)) ** 2
    index = np.digitize(f.T, bins)
    return np.array([power[index == i].mean() for i in range(1, len(bins))])


# Tolerance: 4 standard deviations of the mean variance of four realisations (the few modes
# below the cutoff carry most of the power of steep spectra).
@pytest.mark.parametrize("slope, tolerance", [(0.0, 0.02), (2.0, 0.05), (3.5, 0.15)])
def test_texture_field_has_unit_rms_and_the_power_law(slope, tolerance):
    rng = np.random.default_rng(11)
    fields = [texture_field((256, 256), rng, slope, 0.02).numpy() for _ in range(4)]
    assert all(f.dtype == np.float32 and f.shape == (256, 256) for f in fields)
    assert np.mean([f.mean() for f in fields]) == pytest.approx(0.0, abs=1e-6)
    assert np.mean([f.var() for f in fields]) == pytest.approx(1.0, rel=tolerance)
    bins = np.array([0.1, 0.13, 0.2, 0.26, 0.4, 0.45])
    power = np.mean([_radial_power(f, bins) for f in fields], axis=0)
    centre = np.sqrt(bins[:-1] * bins[1:])
    fitted = np.polyfit(np.log(centre[[0, 2, 4]]), np.log(power[[0, 2, 4]]), 1)[0]
    assert fitted == pytest.approx(-slope, abs=0.15)


def test_texture_field_is_band_limited_to_the_binned_grid():
    a = texture_field((128, 96), np.random.default_rng(3), 2.0, 0.05, oversample=2)
    b = texture_field((128, 96), np.random.default_rng(3), 2.0, 0.05, oversample=2)
    np.testing.assert_array_equal(a.numpy(), b.numpy())
    spectrum = np.abs(np.fft.fft2(a.numpy().astype(np.float64)))
    fy, fx = np.meshgrid(np.fft.fftfreq(128), np.fft.fftfreq(96), indexing="ij")
    outside = (np.abs(fy) >= 0.25) | (np.abs(fx) >= 0.25)
    assert spectrum[outside].max() < 1e-4 * spectrum[~outside].max()
    double = texture_field((64, 64), np.random.default_rng(0), 1.0, 0.1, dtype=torch.float64)
    assert double.dtype == torch.float64
    with pytest.raises(ValueError):
        texture_field((64, 64), np.random.default_rng(0), -1.0, 0.1)
