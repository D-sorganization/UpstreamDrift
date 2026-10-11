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
