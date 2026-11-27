"""Triangle-surface distances for appearance diagnostics, in mesh length units.

Negative distances denote the interior of a closed, consistently wound surface.
Containment uses the absolute solid-angle winding number; distance uses the
nearest triangle face or edge, never the nearest vertex or a convex hull.
This diagnoses visual meshes, not the separate physical contact capsules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.geometry import Mesh


def signed_mesh_distances(
    points: NDArray[np.float64], mesh: Mesh
) -> NDArray[np.float64]:
    """Return one signed distance per finite point; refuse open/degenerate meshes.

    Work is bounded to one point against the triangles at a time. A global
    reversal of all faces preserves the result. Mixed face winding is refused.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (N, 3) array")
    vertices, faces = (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces),
    )
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError("mesh must contain finite vertices and indexed triangles")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    ordered = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(
        ordered, axis=0, return_inverse=True, return_counts=True
    )
    balance = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if np.any(counts != 2) or np.any(balance):
        raise ValueError("mesh must be closed with consistent triangle winding")
    triangles = vertices[faces]
    ab, ac = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    normal = np.asarray(np.cross(ab, ac), dtype=np.float64)
    norm2 = np.einsum("ij,ij->i", normal, normal)
    if np.any(norm2 <= 0):
        raise ValueError("mesh must not contain degenerate triangles")
    result = np.empty(len(points))
    for index in range(len(points)):
        point = np.asarray(points[index], dtype=np.float64)
        result[index] = _signed_distance(point, triangles, normal, norm2)
    return result


def _signed_distance(
    point: NDArray[np.float64],
    triangles: NDArray[np.float64],
    normal: NDArray[np.float64],
    norm2: NDArray[np.float64],
) -> float:
    """Return the signed distance from a point to the mesh."""
    relative = point - triangles[:, 0]
    height = np.einsum("ij,ij->i", relative, normal)
    projected = point - normal * (height / norm2)[:, None]
    on_face = np.ones(len(triangles), dtype=bool)
    edge_distance2 = np.full(len(triangles), np.inf)
    for start_index, end_index in ((0, 1), (1, 2), (2, 0)):
        start, end = triangles[:, start_index], triangles[:, end_index]
        edge = end - start
        side = np.einsum("ij,ij->i", np.cross(edge, projected - start), normal)
        on_face &= side >= 0
        fraction = np.clip(
            np.einsum("ij,ij->i", point - start, edge)
            / np.einsum("ij,ij->i", edge, edge),
            0,
            1,
        )
        delta = point - (start + fraction[:, None] * edge)
        edge_distance2 = np.minimum(edge_distance2, np.einsum("ij,ij->i", delta, delta))
    distance2 = np.where(on_face, height * height / norm2, edge_distance2)
    distance = float(np.sqrt(distance2.min()))
    a, b, c = (triangles[:, i] - point for i in range(3))
    na, nb, nc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        na * nb * nc
        + np.einsum("ij,ij->i", a, b) * nc
        + np.einsum("ij,ij->i", b, c) * na
        + np.einsum("ij,ij->i", c, a) * nb
    )
    inside = abs(float((2 * np.arctan2(numerator, denominator)).sum())) > 2 * np.pi
    return -distance if inside else distance
