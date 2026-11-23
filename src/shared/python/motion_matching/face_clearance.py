"""Instantaneous sphere clearance to a finite triangle patch, not contact detection.

The point/triangle region calculation is extracted from the experimental
``contact_event._point_triangles`` implementation. No trajectory, front-side
test, approach direction, first-root search, or physical feasibility is implied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class FaceClearance:
    """One unsigned point-to-patch distance minus radius, in metres."""

    gap_m: float
    closest_point_m: tuple[float, float, float]
    triangle_index: int


def _closest_points(point: Array, triangles: Array) -> Array:
    """Closest point on each nondegenerate triangle by Voronoi region."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    bp, cp = point - b, point - c
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4
    if not all(np.isfinite(x).all() for x in (d1, d2, d3, d4, d5, d6, va, vb, vc)):
        raise ValueError("face geometry exceeds numerical resolution")
    count = len(triangles)
    closest = np.empty((count, 3), dtype=float)
    assigned = np.zeros(count, dtype=bool)

    def assign(mask: NDArray[np.bool_], weights: Array) -> None:
        closest[mask] = np.einsum("ij,ijk->ik", weights[mask], triangles[mask])
        assigned[mask] = True

    weights = np.zeros((count, 3))
    mask = (d1 <= 0) & (d2 <= 0)
    weights[mask, 0] = 1
    assign(mask, weights)
    mask = ~assigned & (d3 >= 0) & (d4 <= d3)
    weights[:] = 0
    weights[mask, 1] = 1
    assign(mask, weights)
    mask = ~assigned & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    v = np.divide(d1, d1 - d3, out=np.zeros(count), where=(d1 - d3) != 0)
    weights[:] = 0
    weights[:, 0], weights[:, 1] = 1 - v, v
    assign(mask, weights)
    mask = ~assigned & (d6 >= 0) & (d5 <= d6)
    weights[:] = 0
    weights[mask, 2] = 1
    assign(mask, weights)
    mask = ~assigned & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    w = np.divide(d2, d2 - d6, out=np.zeros(count), where=(d2 - d6) != 0)
    weights[:] = 0
    weights[:, 0], weights[:, 2] = 1 - w, w
    assign(mask, weights)
    mask = ~assigned & (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    den = (d4 - d3) + (d5 - d6)
    w = np.divide(d4 - d3, den, out=np.zeros(count), where=den != 0)
    weights[:] = 0
    weights[:, 1], weights[:, 2] = 1 - w, w
    assign(mask, weights)
    mask = ~assigned
    den = va + vb + vc
    if np.any(den[mask] == 0):
        raise ValueError("face triangles must be nondegenerate")
    v, w = vb[mask] / den[mask], vc[mask] / den[mask]
    weights[mask] = np.column_stack((1 - v - w, v, w))
    assign(mask, weights)
    return closest


def face_clearance(triangles: Any, centre_m: Any, radius_m: float) -> FaceClearance:
    """Measure a single sphere-to-open-patch gap; no contact is certified."""
    mesh = np.asarray(triangles, dtype=float)
    point = np.asarray(centre_m, dtype=float)
    if mesh.ndim != 3 or mesh.shape[1:] != (3, 3) or not len(mesh):
        raise ValueError("triangles must have shape (N, 3, 3), N >= 1")
    if (
        point.shape != (3,)
        or not np.isfinite(mesh).all()
        or not np.isfinite(point).all()
    ):
        raise ValueError("face and centre must be finite three-dimensional coordinates")
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius must be finite and positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        edges = mesh[:, 1:] - mesh[:, :1]
        lengths = np.linalg.norm(edges, axis=2)
        areas = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
        if (
            not np.isfinite(lengths).all()
            or not np.isfinite(areas).all()
            or np.any(areas <= np.finfo(float).eps * lengths.max(axis=1) ** 2)
        ):
            raise ValueError("face triangles must be numerically nondegenerate")
        scale = float(np.min(lengths.max(axis=1)))
        magnitude = float(max(np.max(np.abs(mesh)), np.max(np.abs(point))))
        if float(np.spacing(magnitude)) > 1e-8 * scale:
            raise ValueError("face geometry exceeds coordinate resolution")
        points = _closest_points(point, mesh)
        distance = np.linalg.norm(points - point, axis=1)
    if not np.isfinite(points).all() or not np.isfinite(distance).all():
        raise ValueError("face geometry exceeds numerical resolution")
    index = int(np.argmin(distance))
    closest = points[index]
    return FaceClearance(
        float(distance[index] - radius_m),
        (float(closest[0]), float(closest[1]), float(closest[2])),
        index,
    )


def require_initial_separation(
    triangles: Any, centre_m: Any, radius_m: float, *, margin_m: float = 0.0
) -> FaceClearance:
    """Reject initial contact/overlap or an insufficient caller-supplied margin.

    A positive returned gap is not a certificate of outside-solid position or
    a first root; future trajectory intervals must be independently audited.
    """
    if not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError("initial separation margin must be finite and nonnegative")
    sample = face_clearance(triangles, centre_m, radius_m)
    if sample.gap_m <= margin_m:
        raise ValueError("initial face separation is not established")
    return sample
