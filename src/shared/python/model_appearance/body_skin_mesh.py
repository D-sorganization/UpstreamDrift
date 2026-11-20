"""Triangle-mesh steps shared by the CC0 body skin layers (#15299).

Normals, open edges, one Laplacian pass and an exact cut on a level set, on
plain ``(N, 3)`` and ``(F, 3)`` arrays. Appearance only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Faces = NDArray[np.int64]


def vertex_normals(vertices: Array, faces: Faces) -> Array:
    """Area-weighted unit vertex normals."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], np.cross(b - a, c - a))
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-12)


def boundary_edges(faces: Faces) -> Faces:
    """Directed edges ``(E, 2)`` that belong to one face only."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, count = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    return edges[count[inverse.ravel()] == 1]


def relax(points: Array, faces: Faces, fixed: NDArray[np.bool_], step: float) -> Array:
    """One Laplacian pass: free vertices move ``step`` of the way to their ring mean."""
    edges = np.unique(
        np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1),
        axis=0,
    )
    total, count = np.zeros_like(points), np.zeros(len(points))
    for near, far in ((0, 1), (1, 0)):
        np.add.at(total, edges[:, near], points[edges[:, far]])
        np.add.at(count, edges[:, near], 1.0)
    free = (count > 0) & ~fixed
    out = points.copy()
    out[free] += step * (total[free] / count[free, None] - points[free])
    return out


def clip_mesh(columns: Array, faces: Faces, level: Array) -> tuple[Array, Faces]:
    """The part of a mesh where ``level > 0``, cut exactly on ``level == 0``.

    ``columns`` holds per-vertex rows (positions, weights, fields); rows of the
    new vertices on the cut are interpolated. Raises ``ValueError`` when the
    sizes disagree or nothing is kept.
    """
    if len(columns) != len(level) or columns.ndim != 2:
        raise ValueError("columns must be (vertices, k) and match level")
    inside = level > 0.0
    count = inside[faces].sum(axis=1)
    made: dict[tuple[int, int], int] = {}
    rows: list[Array] = []

    def cut(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key not in made:
            share = level[i] / (level[i] - level[j])
            made[key] = len(columns) + len(made)
            rows.append((1.0 - share) * columns[i] + share * columns[j])
        return made[key]

    extra: list[list[int]] = []
    for face in faces[(count == 1) | (count == 2)]:
        lone = inside[face].sum() == 1
        turn = int(np.argmax(inside[face] if lone else ~inside[face]))
        a, b, c = (int(v) for v in np.roll(face, -turn))
        if lone:
            extra.append([a, cut(a, b), cut(a, c)])
        else:
            extra += [[cut(b, a), b, c], [cut(b, a), c, cut(c, a)]]
    kept = np.vstack([faces[count == 3], np.asarray(extra, np.int64).reshape(-1, 3)])
    if not len(kept):
        raise ValueError("the level set keeps no face")
    ids, inverse = np.unique(kept, return_inverse=True)
    return np.vstack([columns, *rows])[ids], inverse.reshape(-1, 3)
