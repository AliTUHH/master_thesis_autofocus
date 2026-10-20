"""Analytic test objects as normalised projected thickness maps ``t(x, y) in [0, 1]``.

Shapes are evaluated on coordinate arrays, so they work in pixel units as well as in metres.
:func:`rasterize` averages a shape over ``supersample x supersample`` sub-pixels to avoid
staircase edges. A single-material object then follows from the maximal phase shift and the
fixed ``delta/beta`` ratio (:func:`holofocus.forward.object_from_phase`), as for the phantoms
of [2] Sec. 3.5 / Fig. 5 (``phi in [-2, 0] rad``, ``mu in [0, 0.01]``, i.e. ``delta/beta = 200``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

Shape = Callable[[np.ndarray, np.ndarray], np.ndarray]


def pixel_coordinates(
    shape: tuple[int, int], pixel_size: float = 1.0, supersample: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel centre coordinates ``(x, y)`` with the origin in the array centre.

    Returns arrays of shape ``(H * s, W * s)``; pixel ``(i, j)`` covers
    ``[(j - W/2) dx, (j + 1 - W/2) dx)`` horizontally.
    """
    h, w = shape
    s = int(supersample)
    ys = (np.arange(h * s) + 0.5) / s - h / 2
    xs = (np.arange(w * s) + 0.5) / s - w / 2
    x, y = np.meshgrid(xs * pixel_size, ys * pixel_size, indexing="xy")
    return x, y


def rasterize(
    shape_fn: Shape, shape: tuple[int, int], pixel_size: float = 1.0, supersample: int = 4
) -> np.ndarray:
    """Pixel-averaged values of ``shape_fn`` on a ``shape`` grid."""
    x, y = pixel_coordinates(shape, pixel_size, supersample)
    values = shape_fn(x, y)
    h, w = shape
    s = int(supersample)
    return values.reshape(h, s, w, s).mean(axis=(1, 3))


def ball(cx: float, cy: float, radius: float) -> Shape:
    """Projected thickness of a sphere, ``sqrt(1 - r^2/R^2)`` (normalised to 1 at the centre)."""

    def fn(x, y):
        r2 = ((x - cx) ** 2 + (y - cy) ** 2) / radius**2
        return np.sqrt(np.clip(1.0 - r2, 0.0, None))

    return fn


def ellipse(cx: float, cy: float, a: float, b: float, angle: float = 0.0) -> Shape:
    """Indicator of an ellipse with half axes ``a``, ``b`` rotated by ``angle`` (radians)."""
    cos, sin = np.cos(angle), np.sin(angle)

    def fn(x, y):
        u = (x - cx) * cos + (y - cy) * sin
        v = -(x - cx) * sin + (y - cy) * cos
        return ((u / a) ** 2 + (v / b) ** 2 <= 1.0).astype(np.float64)

    return fn


def polygon(vertices: Sequence[tuple[float, float]]) -> Shape:
    """Indicator of a simple polygon (even-odd rule), e.g. the triangle of [2] Fig. 5."""
    verts = np.asarray(vertices, dtype=np.float64)

    def fn(x, y):
        inside = np.zeros(x.shape, dtype=bool)
        n = len(verts)
        for i in range(n):
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % n]
            crosses = (y0 > y) != (y1 > y)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (x < x_cross)
        return inside.astype(np.float64)

    return fn


def union_max(*shapes: Shape) -> Shape:
    """Pointwise maximum of several shapes (overlaps do not add thickness)."""

    def fn(x, y):
        return np.maximum.reduce([s(x, y) for s in shapes])

    return fn


def demo_object(shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Fixed test object: a right triangle and a ball, normalised thickness in ``[0, 1]``."""
    h, w = shape
    s = min(h, w)
    tri = polygon([(-0.38 * s, -0.30 * s), (-0.38 * s, 0.22 * s), (0.02 * s, 0.22 * s)])
    sphere = ball(0.20 * s, -0.08 * s, 0.16 * s)
    return rasterize(union_max(tri, sphere), shape, supersample=4)
