"""Triangle-mesh steps shared by the CC0 body skin layers (#15299).

Normals, open edges, depth under the surface, height over a surface, one
Laplacian pass, an exact cut on a level set and a split along one, on
plain ``(N, 3)`` and ``(F, 3)`` arrays. Appearance only.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

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


def vertex_areas(vertices: Array, faces: Faces) -> Array:
    """Surface area per vertex: a third of each face it belongs to."""
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    share = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 6.0
    areas = np.zeros(len(vertices))
    for k in range(3):
        np.add.at(areas, faces[:, k], share)
    return areas


def _face_cross(vertices: Array, faces: Faces) -> Array:
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return np.asarray(np.cross(b - a, c - a), dtype=float)


def folded_faces(vertices: Array, faces: Faces) -> NDArray[np.bool_]:
    """Faces turned against the surface around them: inverted by a pose.

    A face is folded when its normal points against the sum of the
    area-weighted normals of its three vertices.
    """
    around = vertex_normals(vertices, faces)[faces].sum(axis=1)
    facing = np.einsum("ij,ij->i", _face_cross(vertices, faces), around)
    return np.asarray(facing <= 0.0)


def area_ratio(rest: Array, posed: Array, faces: Faces) -> Array:
    """Area of each face of ``posed`` over its area in ``rest``.

    Precondition: both surfaces hold the same vertices in the same order.
    """
    if rest.shape != posed.shape:
        raise ValueError("rest and posed must hold the same vertices")
    before = np.linalg.norm(_face_cross(rest, faces), axis=1)
    after = np.linalg.norm(_face_cross(posed, faces), axis=1)
    return np.asarray(after / np.maximum(before, 1e-18))


def inward_thickness(vertices: Array, faces: Faces, ids: NDArray[np.int64]) -> Array:
    """Depth of the body under each vertex of ``ids``: how far it is along the
    inward vertex normal to the surface on the other side.

    A limb wrung or folded by a pose is thinner there than at rest. Infinite
    where the normal meets no face. Raises ``ValueError`` for an unknown vertex.
    """
    if len(ids) and not 0 <= int(np.min(ids)) <= int(np.max(ids)) < len(vertices):
        raise ValueError("ids must index vertices")
    inward = -vertex_normals(vertices, faces)[ids]
    corner = vertices[faces[:, 0]]
    first, second = (vertices[faces[:, k]] - corner for k in (1, 2))
    depth = np.full(len(ids), np.inf)
    for row, (start, ray) in enumerate(zip(vertices[ids], inward, strict=True)):
        across = np.cross(ray, second)
        scale = np.einsum("ij,ij->i", first, across)
        scale = np.where(np.abs(scale) > 1e-18, scale, np.inf)
        offset = start - corner
        u = np.einsum("ij,ij->i", offset, across) / scale
        around = np.cross(offset, first)
        v = around @ ray / scale
        reach = np.einsum("ij,ij->i", second, around) / scale
        hit = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (reach > 1e-6)
        if hit.any():
            depth[row] = reach[hit].min()
    return depth


def surface_height(points: Array, vertices: Array, faces: Faces) -> Array:
    """Signed height of ``points`` over a surface, at its nearest sample.

    The surface is sampled at its vertices, edge midpoints and face centres;
    the height is measured along the sample's normal, positive outside. Raises
    ``ValueError`` for points that are not ``(N, 3)``.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    normals = vertex_normals(vertices, faces)
    picks = [faces[:, [k]] for k in range(3)] + [faces[:, [0, 1]], faces[:, [1, 2]]]
    picks += [faces[:, [2, 0]], faces]
    where = np.vstack([vertices[pick].mean(axis=1) for pick in picks])
    facing = np.vstack([normals[pick].mean(axis=1) for pick in picks])
    facing /= np.maximum(np.linalg.norm(facing, axis=1, keepdims=True), 1e-12)
    nearest = cKDTree(where).query(points)[1]
    return np.asarray(np.einsum("ij,ij->i", points - where[nearest], facing[nearest]))


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
        key = i, j = (min(i, j), max(i, j))  # one order: the same row from either side
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


def split_mesh(columns: Array, faces: Faces, level: Array) -> tuple[Array, Faces]:
    """The whole mesh with the line ``level == 0`` made edges of its own.

    Both sides of :func:`clip_mesh` on that line, joined in the vertices they
    share on it, so the surface is unchanged and stays one sheet. The rows on
    the line are those ``clip_mesh`` makes for the same ``level``: a layer cut
    there meets this mesh in common vertices and edges, at any pose. Raises
    ``ValueError`` when the sizes disagree or the line leaves one side empty.
    """
    above, upper = clip_mesh(columns, faces, level)
    below, lower = clip_mesh(columns, faces, -np.asarray(level, dtype=float))
    apart, twin = cKDTree(above).query(below)
    ids = np.where(np.asarray(apart) == 0.0, twin, len(above) + np.arange(len(below)))
    joined = np.vstack([upper, ids[lower]])
    kept, inverse = np.unique(joined, return_inverse=True)
    return np.vstack([above, below])[kept], inverse.reshape(-1, 3)
