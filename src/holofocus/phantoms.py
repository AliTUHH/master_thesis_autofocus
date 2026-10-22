"""Analytic test objects as normalised projected thickness maps ``t(x, y) in [0, 1]``.

Shapes are evaluated on coordinate arrays, so they work in pixel units as well as in metres.
:func:`rasterize` averages a shape over ``supersample x supersample`` sub-pixels to avoid
staircase edges. A single-material object then follows from the maximal phase shift and the
fixed ``delta/beta`` ratio (:func:`holofocus.forward.object_from_phase`), as for the phantoms
of [2] Sec. 3.5 / Fig. 5 (``phi in [-2, 0] rad``, ``mu in [0, 0.01]``, i.e. ``delta/beta = 200``).

:func:`random_phantom` draws randomised versions of the three phantom classes of [2] Sec. 3.5
(triangle: strong edges and a flat area; cell: many small sharp features; ball: a broad
gradient) plus ``mixed`` scenes of several objects. Thicknesses of overlapping parts add up,
as for a projection through one material. These compact types stay inside the field of view.
The ``fiber`` type (Stage 5) models the measured samples of [1], hair, needle and wire: straight,
possibly tapered cylinders that cross the field of view and continue into the free-space
padding around it (:func:`fiber_thickness`).

:func:`texture_field` (Stage 6) is the weak-phase texture that the smooth shapes lack and the
measured holograms of [1] show: a Gaussian random field with a power-law spectrum, which the
dataset generator adds inside the object and as a uniform film over the field of view.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .arrays import center_slices

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
COMPACT_TYPES = tuple(_GENERATORS)
"""Phantom types that lie inside the field of view (the types of the Stage 2 datasets)."""
PHANTOM_TYPES = (*COMPACT_TYPES, "fiber")
"""All types; the index into this tuple is the stored ``phantom_type``."""


def _cylinder_h(x: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Second antiderivative of the projected thickness ``2 sqrt(R^2 - x^2)`` of a cylinder,
    zero for ``x <= -R`` and ``pi R^2 x`` for ``x >= R``."""
    r = np.maximum(radius, 1e-12)
    u = np.clip(x, -r, r)
    root = np.sqrt(np.maximum(r**2 - u**2, 0.0))
    inside = -(root**3) / 3 + r**2 * (u * np.arcsin(u / r) + root) + 0.5 * np.pi * r**2 * u
    return np.where(x >= r, np.pi * r**2 * x, np.where(x <= -r, 0.0, inside))


def fiber_thickness(
    xs: np.ndarray,
    ys: np.ndarray,
    center: tuple[float, float],
    angle: float,
    radius: float,
    taper: float = 0.0,
    chunk_rows: int = 256,
) -> np.ndarray:
    """Pixel-averaged projected thickness of a straight cylinder on the grid ``ys x xs``.

    ``xs`` and ``ys`` are pixel-centre coordinates in pixels (as in :func:`pixel_coordinates`).
    The axis passes through ``center = (x, y)`` in the direction ``angle``; at the distance
    ``s`` along the axis from ``center`` the radius is ``radius * clip(1 + taper * s, 0, 2)``,
    so a tapered fiber ends in a tip on one side and at most doubles on the other. The
    thickness ``2 sqrt(R^2 - d^2)`` at the distance ``d`` from the axis is averaged exactly over
    the pixel footprint (two box filters across the axis, the projections of the pixel sides)
    with the local radius.
    """
    cos, sin = math.cos(angle), math.sin(angle)
    a, b = abs(sin), abs(cos)
    a, b = max(a, 1e-3), max(b, 1e-3)
    dx = np.asarray(xs, dtype=np.float64) - center[0]
    out = np.empty((len(ys), len(xs)))
    for start in range(0, len(ys), chunk_rows):
        dy = np.asarray(ys[start : start + chunk_rows], dtype=np.float64)[:, None] - center[1]
        d = -dx[None, :] * sin + dy * cos
        r = radius * np.clip(1.0 + taper * (dx[None, :] * cos + dy * sin), 0.0, 2.0)
        near = (np.abs(d) < r + 0.5 * (a + b)) & (r > 0)
        values = np.zeros_like(d)
        dn, rn = d[near], r[near]
        values[near] = (
            _cylinder_h(dn + 0.5 * (a + b), rn)
            - _cylinder_h(dn + 0.5 * (a - b), rn)
            - _cylinder_h(dn - 0.5 * (a - b), rn)
            + _cylinder_h(dn - 0.5 * (a + b), rn)
        ) / (a * b)
        out[start : start + len(dy)] = values
    return out


def _edge_taper(n_outer: int, n_inner: int) -> np.ndarray:
    """1 on the inner block and the inner three quarters of the border, then a cos^2 decay to
    0 at the array edge (keeps fibers from wrapping around the periodic padded array)."""
    start = (n_outer - n_inner) // 2
    width = start // 4
    if width < 1:
        return np.ones(n_outer)
    i = np.arange(n_outer)
    dist = np.maximum(np.maximum(start - i, i - (start + n_inner - 1)), 0)
    edge = np.clip((dist - (start - width)) / width, 0.0, 1.0)
    return np.cos(0.5 * np.pi * edge) ** 2


def random_fibers(
    rng: np.random.Generator,
    shape: tuple[int, int],
    canvas: tuple[int, int] | None = None,
    diameter_range: tuple[float, float] = (0.03, 1.0),
) -> np.ndarray:
    """Projected thickness (pixels) of one or two random fibers crossing the field of view.

    The field of view ``shape`` lies centred in ``canvas`` (as in
    :func:`holofocus.arrays.center_slices`); the fibers continue over the whole canvas and fade
    out in the outer quarter of the border. Diameters are log-uniform in ``diameter_range``
    relative to the shorter side of the field of view; thick fibers may have their axis outside
    it and cover part of it, like the edge of a large object. Half of the fibers taper, some to
    a tip inside the field of view.
    """
    canvas = tuple(shape) if canvas is None else tuple(canvas)
    h, w = shape
    n = min(shape)
    r0 = (canvas[0] - h) // 2
    c0 = (canvas[1] - w) // 2
    xs = np.arange(canvas[1]) - c0 + 0.5 - w / 2
    ys = np.arange(canvas[0]) - r0 + 0.5 - h / 2
    thickness = np.zeros(canvas)
    log_lo, log_hi = (math.log(v) for v in diameter_range)
    for _ in range(1 if rng.uniform() < 0.7 else 2):
        angle = rng.uniform(0, math.pi)
        radius = 0.5 * n * math.exp(rng.uniform(log_lo, log_hi))
        offset = rng.uniform(-1.0, 1.0) * (0.45 * n + 0.5 * radius)
        taper = rng.uniform(-2.0, 2.0) / n if rng.uniform() < 0.5 else 0.0
        center = (-offset * math.sin(angle), offset * math.cos(angle))
        thickness += fiber_thickness(xs, ys, center, angle, radius, taper)
    return thickness * np.outer(_edge_taper(canvas[0], h), _edge_taper(canvas[1], w))


def random_phantom(
    rng: np.random.Generator,
    kind: str,
    shape: tuple[int, int],
    size_range: tuple[float, float] = (0.15, 0.6),
    margin: float = 0.05,
    supersample: int = 4,
    canvas: tuple[int, int] | None = None,
    fiber_diameter_range: tuple[float, float] = (0.03, 1.0),
    normalize: bool = True,
) -> np.ndarray:
    """Normalised thickness (maximum 1 inside the field of view) of a random phantom.

    ``shape`` is the field of view. For the compact types, ``size_range`` is the object
    diameter relative to its shorter side and ``margin`` the free border relative to it;
    these objects never touch the border. ``fiber`` uses ``fiber_diameter_range`` instead
    (:func:`random_fibers`). With ``canvas`` the result covers that larger array with the field
    of view in its centre (zero outside the field of view for the compact types). With
    ``normalize=False`` a fiber keeps its projected thickness in pixels of ``shape``; the
    compact shapes have no thickness in pixels.
    """
    if kind not in PHANTOM_TYPES:
        raise ValueError(f"unknown phantom type {kind!r}, choose from {PHANTOM_TYPES}")
    if not normalize and kind != "fiber":
        raise ValueError(f"only fibers have a thickness in pixels, not {kind!r}")
    if kind == "fiber":
        thickness = random_fibers(rng, shape, canvas, fiber_diameter_range)
        if not normalize:
            return thickness
        rows, cols = center_slices(thickness.shape, tuple(shape))
        peak = thickness[rows, cols].max()
        return thickness / peak if peak > 0 else thickness
    n = min(shape)
    radius = 0.5 * n * rng.uniform(*size_range)
    thickness = render(_GENERATORS[kind](rng, shape, radius, margin * n), shape, supersample)
    peak = thickness.max()
    thickness = thickness / peak if peak > 0 else thickness
    if canvas is None or tuple(canvas) == tuple(shape):
        return thickness
    out = np.zeros(canvas)
    rows, cols = center_slices(tuple(canvas), tuple(shape))
    out[rows, cols] = thickness
    return out


def texture_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    slope: float,
    cutoff: float,
    oversample: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Gaussian random field with the power spectrum ``(1 + (f / cutoff)^2)^(-slope / 2)``.

    ``f`` is in cycles per pixel of the binned grid and ``shape`` a grid ``oversample`` times
    finer. The field is periodic over ``shape``, has zero mean and is band-limited to the binned
    grid (``|f_x|, |f_y| < 1/2``), so the oversampling of the simulation does not change it. Its
    expected variance over that band is 1; one realisation deviates by the sampling of its
    lowest frequencies. The white noise comes from ``rng`` (single precision), so the field is
    the same on every device up to rounding.
    """
    if slope < 0 or cutoff <= 0 or oversample < 1:
        raise ValueError("need slope >= 0, cutoff > 0 and oversample >= 1")
    h, w = shape
    white = torch.from_numpy(rng.standard_normal((h, w), dtype=np.float32))
    white = white.to(device=device, dtype=dtype)
    real = torch.float64
    fy = torch.fft.fftfreq(h, device=white.device, dtype=real)[:, None] * oversample
    fx = torch.fft.rfftfreq(w, device=white.device, dtype=real)[None, :] * oversample
    band = (fy.abs() < 0.5) & (fx.abs() < 0.5)
    amplitude = torch.where(band, (1.0 + (fx**2 + fy**2) / cutoff**2) ** (-slope / 4), 0.0)
    amplitude[0, 0] = 0.0
    # Each column of the half spectrum stands for two frequencies of the full one, except the
    # zero column and, for even w, the Nyquist column.
    weight = torch.full_like(amplitude, 2.0)
    weight[:, 0] = 1.0
    if w % 2 == 0:
        weight[:, -1] = 1.0
    variance = float((weight * amplitude**2).sum()) / (h * w)
    if variance == 0:
        raise ValueError(f"no frequency of the band on a {shape} grid with oversample {oversample}")
    spectrum = torch.fft.rfft2(white) * (amplitude / math.sqrt(variance)).to(dtype)
    return torch.fft.irfft2(spectrum, s=(h, w))
