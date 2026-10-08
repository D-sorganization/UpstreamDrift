"""Engine-agnostic smooth body-segment meshes (pure numpy).

``lofted_segment`` builds a closed spindle (or a flat-capped garment band)
along a segment; ``ellipsoid_mesh`` builds an oriented ellipsoid (club head,
shoe toe). Meshes carry vertices and triangle faces in the caller's frame and
are visual only: they are never used for collision, mass or inertia.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Mesh:
    vertices: Array  # (N, 3) float
    faces: NDArray[np.int64]  # (M, 3) outward-wound triangles

    def volume(self) -> float:
        """Signed enclosed volume; positive for outward winding."""
        tri = self.vertices[self.faces]
        return float(
            np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
        )


def _frame(axis: Array, hint: Array | None = None) -> tuple[Array, Array, Array]:
    """Right-handed (u, v, a) with ``a`` the unit axis."""
    a = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if hint is None else np.asarray(hint, float)
    if abs(float(np.dot(ref, a))) > 0.9:
        ref = (
            np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
        )
    u = ref - np.dot(ref, a) * a
    u /= np.linalg.norm(u)
    return u, np.cross(a, u), a


def _check_finite(name: str, value: Array, size: int) -> Array:
    v = np.asarray(value, dtype=float)
    if v.shape != (size,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return v


def _orient(vertices: Array, faces: NDArray[np.int64]) -> Mesh:
    mesh = Mesh(vertices, faces)
    if mesh.volume() < 0:
        mesh = Mesh(vertices, faces[:, ::-1].copy())
    return mesh


def _ring_faces(n_rings: int, sides: int) -> list[tuple[int, int, int]]:
    faces = []
    for k in range(n_rings - 1):
        for j in range(sides):
            j2 = (j + 1) % sides
            a, b = k * sides + j, k * sides + j2
            c, d = (k + 1) * sides + j2, (k + 1) * sides + j
            faces += [(a, b, c), (a, c, d)]
    return faces


@dataclass(frozen=True)
class LoftOptions:
    """Shape options for :func:`lofted_segment`."""

    aspect: float = 1.0
    coverage: tuple[float, float] = (0.0, 1.0)
    thickness: float = 1.0
    rings: int = 14
    sides: int = 20
    width_hint: Array | None = None


def lofted_segment(
    start: Array,
    end: Array,
    radius: float,
    options: LoftOptions | None = None,
) -> Mesh:
    """Smooth lofted body between ``start`` and ``end``.

    Full coverage gives a closed spindle that overshoots both ends by half a
    radius so neighbouring segments overlap at joints. A partial ``coverage``
    (fractions of the length) gives a flat-capped band, used for sleeves,
    shorts and shoes. ``aspect`` is depth/width of the cross-section.
    Preconditions: finite points, positive length and radius.
    """
    opts = options or LoftOptions()
    aspect, coverage, thickness = opts.aspect, opts.coverage, opts.thickness
    rings, sides, width_hint = opts.rings, opts.sides, opts.width_hint
    p0, p1 = _check_finite("start", start, 3), _check_finite("end", end, 3)
    if not math.isfinite(radius) or radius <= 0 or thickness <= 0 or aspect <= 0:
        raise ValueError("radius, thickness and aspect must be positive")
    t0, t1 = coverage
    if not 0.0 <= t0 < t1 <= 1.0:
        raise ValueError("coverage must satisfy 0 <= start < end <= 1")
    length = float(np.linalg.norm(p1 - p0))
    if length <= 1e-9 or rings < 4 or sides < 6:
        raise ValueError("Segment needs positive length, rings >= 4, sides >= 6")
    u, v, a = _frame(p1 - p0, width_hint)
    over = 0.5 * radius
    half = length / 2.0 + over
    centre = length / 2.0

    def profile(s: float) -> float:
        x = min(1.0, abs(s - centre) / half)
        return math.sqrt(max(0.0, 1.0 - x * x)) ** 0.55

    closed = t0 == 0.0 and t1 == 1.0
    if closed:
        theta = np.linspace(0.0, math.pi, rings + 2)[1:-1]
        stations = centre - half * np.cos(theta)
    else:
        stations = np.linspace(t0 * length, t1 * length, rings)
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring_dirs = np.outer(np.cos(phi), u) + aspect * np.outer(np.sin(phi), v)
    verts: list[Array] = []
    for s in stations:
        r = radius * thickness * profile(float(s))
        verts.extend(p0 + a * s + r * ring_dirs)
    face_list = _ring_faces(len(stations), sides)
    n = len(verts)
    if closed:
        verts += [p0 - a * over, p1 + a * over]
        first = (len(stations) - 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            face_list.append((n, j2, j))
            face_list.append((n + 1, first + j, first + j2))
    else:
        verts += [p0 + a * stations[0], p0 + a * stations[-1]]
        first = (len(stations) - 1) * sides
        for j in range(sides):
            j2 = (j + 1) % sides
            face_list.append((n, j2, j))
            face_list.append((n + 1, first + j, first + j2))
    return _orient(
        np.asarray(verts, dtype=float), np.asarray(face_list, dtype=np.int64)
    )


def ellipsoid_mesh(
    center: Array,
    half_sizes: Array,
    axis: Array,
    *,
    rings: int = 12,
    sides: int = 20,
    width_hint: Array | None = None,
) -> Mesh:
    """Ellipsoid whose first half size lies along ``axis`` (then u, v)."""
    c = _check_finite("center", center, 3)
    h = _check_finite("half_sizes", half_sizes, 3)
    if (h <= 0).any():
        raise ValueError("Ellipsoid half sizes must be positive")
    if np.linalg.norm(np.asarray(axis, float)) <= 1e-12:
        raise ValueError("Ellipsoid axis must be nonzero")
    u, v, a = _frame(np.asarray(axis, float), width_hint)
    theta = np.linspace(0.0, math.pi, rings + 2)[1:-1]
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    verts = []
    for t in theta:
        ring = np.outer(math.sin(t) * np.cos(phi), h[1] * u) + np.outer(
            math.sin(t) * np.sin(phi), h[2] * v
        )
        verts.extend(c + a * (h[0] * -math.cos(t)) + ring)
    faces = _ring_faces(rings, sides)
    n = len(verts)
    verts += [c - a * h[0], c + a * h[0]]
    last = (rings - 1) * sides
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append((n, j2, j))
        faces.append((n + 1, last + j, last + j2))
    return _orient(np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64))


def closed_grid_mesh(grid: Array, top: Array, bottom: Array) -> Mesh:
    """Closed mesh from ``(rings, sides, 3)`` vertex rings between two poles.

    ``top`` is fanned to the first ring and ``bottom`` to the last, so a
    deformed sphere (skull) or a dome fanned to an interior point (hair, cap)
    is built the same way. Preconditions: finite grid with at least two rings
    and six sides. Postcondition: positive enclosed volume (winding repaired).
    """
    g = np.asarray(grid, dtype=float)
    if g.ndim != 3 or g.shape[2] != 3 or g.shape[0] < 2 or g.shape[1] < 6:
        raise ValueError("grid must be (rings >= 2, sides >= 6, 3)")
    if not np.isfinite(g).all():
        raise ValueError("grid must be finite")
    t = _check_finite("top", top, 3)
    b = _check_finite("bottom", bottom, 3)
    rings, sides = g.shape[:2]
    faces = _ring_faces(rings, sides)
    n = rings * sides
    verts = np.vstack([g.reshape(-1, 3), t, b])
    last = (rings - 1) * sides
    for j in range(sides):
        j2 = (j + 1) % sides
        faces.append((n, j2, j))
        faces.append((n + 1, last + j, last + j2))
    return _orient(verts, np.asarray(faces, dtype=np.int64))
