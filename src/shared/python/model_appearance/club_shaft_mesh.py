"""Tapered tubes for club shafts, grips and hosels (visual only, pure numpy)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh

Array = NDArray[np.float64]


def tapered_tube(
    start: Array, end: Array, r_start: float, r_end: float, sides: int = 24
) -> Mesh:
    """Closed flat-capped frustum from ``start`` to ``end`` with outward winding.

    Preconditions: finite distinct points, positive radii, ``sides >= 6``.
    """
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    if (
        p0.shape != (3,)
        or p1.shape != (3,)
        or not (np.isfinite(p0).all() and np.isfinite(p1).all())
    ):
        raise ValueError("start and end must be finite 3-vectors")
    if (
        not (math.isfinite(r_start) and math.isfinite(r_end))
        or min(r_start, r_end) <= 0
    ):
        raise ValueError("radii must be positive")
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 1e-9 or sides < 6:
        raise ValueError("tube needs positive length and sides >= 6")
    a = axis / length
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v)
    verts = np.vstack([p0 + r_start * ring, p1 + r_end * ring, p0, p1])
    faces = []
    for j in range(sides):
        j2 = (j + 1) % sides
        faces += [(j, j2, sides + j2), (j, sides + j2, sides + j)]
        faces += [(2 * sides, j2, j), (2 * sides + 1, sides + j, sides + j2)]
    mesh = Mesh(verts, np.asarray(faces, dtype=np.int64))
    if mesh.volume() < 0.0:
        mesh = Mesh(verts, mesh.faces[:, ::-1].copy())
    return mesh


def profiled_tube(
    start: Array, end: Array, stations: list[tuple[float, float]], sides: int = 24
) -> Mesh:
    """Closed flat-capped surface of revolution from ``start`` to ``end``.

    ``stations`` are ``(fraction, radius)`` pairs, fractions rising from 0 to 1
    along the axis. Preconditions: finite distinct points, two or more
    stations, positive radii, ``sides >= 6``. Postcondition: outward winding.
    """
    p0, p1 = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    frac, rad = np.array(stations, dtype=float).T
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if p0.shape != (3,) or not (np.isfinite(p0).all() and np.isfinite(p1).all()):
        raise ValueError("start and end must be finite 3-vectors")
    if length <= 1e-9 or sides < 6 or len(frac) < 2:
        raise ValueError("tube needs positive length, two stations and sides >= 6")
    if frac[0] != 0.0 or frac[-1] != 1.0 or (np.diff(frac) <= 0.0).any():
        raise ValueError("station fractions must rise from 0 to 1")
    if not np.isfinite(rad).all() or (rad <= 0.0).any():
        raise ValueError("radii must be positive")
    a = axis / length
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, ref)
    u /= np.linalg.norm(u)
    phi = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = np.outer(np.cos(phi), u) + np.outer(np.sin(phi), np.cross(a, u))
    rings = [p0 + f * axis + r * ring for f, r in zip(frac, rad, strict=True)]
    n = len(frac)
    verts = np.vstack([*rings, p0, p1])
    faces = []
    for i in range(n - 1):
        for j in range(sides):
            lo, hi, j2 = i * sides, (i + 1) * sides, (j + 1) % sides
            faces += [(lo + j, lo + j2, hi + j2), (lo + j, hi + j2, hi + j)]
    last = (n - 1) * sides
    for j in range(sides):
        j2 = (j + 1) % sides
        faces += [(n * sides, j2, j), (n * sides + 1, last + j, last + j2)]
    mesh = Mesh(verts, np.asarray(faces, dtype=np.int64))
    if mesh.volume() < 0.0:
        mesh = Mesh(verts, mesh.faces[:, ::-1].copy())
    return mesh
