"""Analytic test objects as normalised projected thickness maps ``t(x, y) in [0, 1]``.

Shapes are evaluated on coordinate arrays, so they work in pixel units as well as in metres.
:func:`rasterize` averages a shape over ``supersample x supersample`` sub-pixels to avoid
staircase edges. A single-material object then follows from the maximal phase shift and the
fixed ``delta/beta`` ratio (:func:`holofocus.forward.object_from_phase`), as for the phantoms
of [2] Sec. 3.5 / Fig. 5 (``phi in [-2, 0] rad``, ``mu in [0, 0.01]``, i.e. ``delta/beta = 200``).

:func:`random_phantom` draws randomised versions of the three phantom classes of [2] Sec. 3.5
(triangle: strong edges and a flat area; cell: many small sharp features; ball: a broad
gradient) plus ``mixed`` scenes of several objects. Thicknesses of overlapping parts add up,
as for a projection through one material.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

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


def ellipsoid(cx: float, cy: float, a: float, b: float, angle: float = 0.0) -> Shape:
    """Projected thickness of an ellipsoid, ``sqrt(1 - (u/a)^2 - (v/b)^2)``, maximum 1."""
    cos, sin = np.cos(angle), np.sin(angle)

    def fn(x, y):
        u = (x - cx) * cos + (y - cy) * sin
        v = -(x - cx) * sin + (y - cy) * cos
        return np.sqrt(np.clip(1.0 - (u / a) ** 2 - (v / b) ** 2, 0.0, None))

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


@dataclass(frozen=True)
class Primitive:
    """A shape in pixel coordinates with a weight and a bounding circle used for rendering."""

    fn: Shape
    cx: float
    cy: float
    radius: float
    weight: float = 1.0


def render(primitives: Sequence[Primitive], shape: tuple[int, int], supersample: int = 4):
    """Sum of the pixel-averaged, weighted primitives on a ``shape`` grid (pixel units).

    Each primitive is evaluated only inside its bounding box; the sub-pixel grid is the one of
    :func:`pixel_coordinates`, so the result equals :func:`rasterize` of the weighted sum.
    """
    h, w = shape
    s = int(supersample)
    out = np.zeros(shape)
    for p in primitives:
        r0 = max(math.floor(p.cy + h / 2 - p.radius) - 1, 0)
        r1 = min(math.ceil(p.cy + h / 2 + p.radius) + 1, h)
        c0 = max(math.floor(p.cx + w / 2 - p.radius) - 1, 0)
        c1 = min(math.ceil(p.cx + w / 2 + p.radius) + 1, w)
        if r0 >= r1 or c0 >= c1:
            continue
        ys = (np.arange(r0 * s, r1 * s) + 0.5) / s - h / 2
        xs = (np.arange(c0 * s, c1 * s) + 0.5) / s - w / 2
        x, y = np.meshgrid(xs, ys, indexing="xy")
        values = p.fn(x, y).reshape(r1 - r0, s, c1 - c0, s).mean(axis=(1, 3))
        out[r0:r1, c0:c1] += p.weight * values
    return out


def _random_centre(rng, shape, radius, margin):
    """Centre such that a circle of ``radius`` stays ``margin`` pixels inside the grid."""
    h, w = shape
    lim_x = max(w / 2 - radius - margin, 0.0)
    lim_y = max(h / 2 - radius - margin, 0.0)
    return rng.uniform(-lim_x, lim_x), rng.uniform(-lim_y, lim_y)


def _point_in_ellipse(rng, cx, cy, a, b, angle, rho_max):
    rho = rho_max * math.sqrt(rng.uniform())
    t = rng.uniform(0, 2 * math.pi)
    u, v = rho * a * math.cos(t), rho * b * math.sin(t)
    return (
        cx + u * math.cos(angle) - v * math.sin(angle),
        cy + u * math.sin(angle) + v * math.cos(angle),
    )


def _triangle(rng, shape, radius, margin):
    """Flat triangle with perturbed vertices on a circle (strong edges, smooth area)."""
    cx, cy = _random_centre(rng, shape, radius, margin)
    angles = rng.uniform(0, 2 * math.pi) + np.arange(3) * 2 * math.pi / 3
    angles = angles + rng.uniform(-0.45, 0.45, size=3)
    radii = radius * rng.uniform(0.6, 1.0, size=3)
    vertices = [
        (cx + r * math.cos(a), cy + r * math.sin(a)) for r, a in zip(radii, angles, strict=True)
    ]
    return [Primitive(polygon(vertices), cx, cy, radius)]


def _ball(rng, shape, radius, margin):
    """Sphere: projected thickness with a broad gradient towards the edge."""
    cx, cy = _random_centre(rng, shape, radius, margin)
    return [Primitive(ball(cx, cy, radius), cx, cy, radius)]


def _cell(rng, shape, radius, margin):
    """Cell-like object: ellipsoidal body, denser nucleus and small sharp organelles."""
    cx, cy = _random_centre(rng, shape, radius, margin)
    a, b = radius, radius * rng.uniform(0.6, 1.0)
    angle = rng.uniform(0, math.pi)
    primitives = [Primitive(ellipsoid(cx, cy, a, b, angle), cx, cy, a)]
    scale = rng.uniform(0.3, 0.5)
    nx, ny = _point_in_ellipse(rng, cx, cy, a, b, angle, 1.0 - scale)
    na, nb = scale * b, scale * b * rng.uniform(0.7, 1.0)
    primitives.append(
        Primitive(
            ellipsoid(nx, ny, na, nb, rng.uniform(0, math.pi)),
            nx,
            ny,
            na,
            weight=rng.uniform(0.4, 0.8),
        )
    )
    for _ in range(int(rng.integers(6, 21))):
        r = max(radius * rng.uniform(0.03, 0.10), 1.0)
        ox, oy = _point_in_ellipse(rng, cx, cy, a, b, angle, 0.8)
        primitives.append(
            Primitive(
                ellipse(ox, oy, r, r * rng.uniform(0.5, 1.0), rng.uniform(0, math.pi)),
                ox,
                oy,
                r,
                weight=rng.uniform(0.15, 0.5),
            )
        )
    return primitives


def _mixed(rng, shape, radius, margin):
    """Two to four smaller triangles, cells or balls; overlaps add thickness."""
    primitives = []
    for _ in range(int(rng.integers(2, 5))):
        kind = ("triangle", "cell", "ball")[int(rng.integers(3))]
        primitives += _GENERATORS[kind](rng, shape, radius * rng.uniform(0.35, 0.6), margin)
    return primitives


_GENERATORS = {"triangle": _triangle, "cell": _cell, "ball": _ball, "mixed": _mixed}
PHANTOM_TYPES = tuple(_GENERATORS)


def random_phantom(
    rng: np.random.Generator,
    kind: str,
    shape: tuple[int, int],
    size_range: tuple[float, float] = (0.15, 0.6),
    margin: float = 0.05,
    supersample: int = 4,
) -> np.ndarray:
    """Normalised thickness (maximum 1) of a random phantom of class ``kind``.

    ``size_range`` is the object diameter relative to the shorter side of the grid and
    ``margin`` the free border relative to it; objects never touch the border.
    """
    if kind not in _GENERATORS:
        raise ValueError(f"unknown phantom type {kind!r}, choose from {PHANTOM_TYPES}")
    n = min(shape)
    radius = 0.5 * n * rng.uniform(*size_range)
    thickness = render(_GENERATORS[kind](rng, shape, radius, margin * n), shape, supersample)
    peak = thickness.max()
    return thickness / peak if peak > 0 else thickness
