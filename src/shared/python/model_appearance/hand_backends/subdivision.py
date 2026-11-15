"""Loop subdivision of the skinned hand mesh (epic #15150).

The MakeHuman hand is coarse (about 1 800 vertices), so smooth shading shows
its triangles as facets and a tight crease is a single edge. One level of
Loop subdivision splits every triangle in four and moves the vertices toward
the smooth limit surface. The skin weights go through the same linear
operator, so the finer mesh is skinned exactly like the coarse one. Visual
only and switchable: zero levels returns the asset untouched.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from src.shared.python.model_appearance.hand_backends.skinning import SkinnedHandAsset

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]
_ODD_NEAR, _ODD_FAR = 3.0 / 8.0, 1.0 / 8.0


def _edges(faces: IntArray) -> tuple[IntArray, IntArray, IntArray]:
    """Unique edges, the edge id of every face side, and each edge's two
    opposite vertices. Raises ``ValueError`` unless the mesh is closed."""
    sides = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    opposite = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]])
    edges, inverse, counts = np.unique(
        np.sort(sides, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    if not (counts == 2).all():
        raise ValueError("Loop subdivision needs a closed manifold mesh")
    order = np.argsort(inverse, kind="stable")
    wings = opposite[order].reshape(-1, 2)
    return edges.astype(np.int64), inverse.astype(np.int64), wings.astype(np.int64)


def _even_rows(edges: IntArray, vertex_count: int) -> sparse.csr_matrix:
    """Each original vertex moved toward the mean of its neighbours (Warren)."""
    valence = np.bincount(edges.ravel(), minlength=vertex_count).astype(np.float64)
    if (valence < 3).any():
        raise ValueError("every vertex needs at least three neighbours")
    beta = np.where(valence == 3, 3.0 / 16.0, 3.0 / (8.0 * valence))
    rows = np.concatenate([edges[:, 0], edges[:, 1], np.arange(vertex_count)])
    cols = np.concatenate([edges[:, 1], edges[:, 0], np.arange(vertex_count)])
    data = np.concatenate([beta[edges[:, 0]], beta[edges[:, 1]], 1.0 - valence * beta])
    return sparse.csr_matrix((data, (rows, cols)), shape=(vertex_count, vertex_count))


def _odd_rows(edges: IntArray, wings: IntArray, vertex_count: int) -> sparse.csr_matrix:
    """One new vertex per edge: 3/8 of each end, 1/8 of each opposite vertex."""
    count = len(edges)
    rows = np.tile(np.arange(count), 4)
    cols = np.concatenate([edges[:, 0], edges[:, 1], wings[:, 0], wings[:, 1]])
    data = np.repeat([_ODD_NEAR, _ODD_NEAR, _ODD_FAR, _ODD_FAR], count)
    return sparse.csr_matrix((data, (rows, cols)), shape=(count, vertex_count))


def loop_subdivision(
    faces: IntArray, vertex_count: int
) -> tuple[sparse.csr_matrix, IntArray]:
    """One level of Loop subdivision as a linear operator.

    Returns ``(operator, fine_faces)``: ``operator @ values`` gives the values
    on the fine mesh for any per-vertex quantity. Postconditions: the first
    ``vertex_count`` fine vertices are the original ones in order; every row
    of ``operator`` sums to one; ``fine_faces`` keeps the winding. Raises
    ``ValueError`` unless the mesh is a closed manifold.
    """
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3 or f.min() < 0 or f.max() >= vertex_count:
        raise ValueError("faces must be (F, 3) indices into the vertices")
    edges, side_edge, wings = _edges(f)
    operator = sparse.vstack(
        [_even_rows(edges, vertex_count), _odd_rows(edges, wings, vertex_count)]
    ).tocsr()
    mid = vertex_count + side_edge.reshape(3, -1)  # sides ab, bc, ca
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    ab, bc, ca = mid[0], mid[1], mid[2]
    fine = np.concatenate(
        [
            np.column_stack([a, ab, ca]),
            np.column_stack([b, bc, ab]),
            np.column_stack([c, ca, bc]),
            np.column_stack([ab, bc, ca]),
        ]
    )
    return operator, fine.astype(np.int64)


def _strongest(dense: Array, count: int) -> tuple[IntArray, Array]:
    """The ``count`` largest weights of every row, renormalised to sum to one."""
    order = np.argsort(-dense, axis=1, kind="stable")[:, :count]
    weights = np.take_along_axis(dense, order, axis=1)
    return order.astype(np.int64), weights / weights.sum(axis=1, keepdims=True)


def subdivide_asset(asset: SkinnedHandAsset, levels: int) -> SkinnedHandAsset:
    """``asset`` with its mesh Loop-subdivided ``levels`` times.

    The skeleton is unchanged and each vertex keeps as many bone influences
    as before. Original vertex ids stay valid. ``levels == 0`` returns
    ``asset`` itself; a negative or non-integer ``levels`` raises.
    """
    if isinstance(levels, bool) or not isinstance(levels, int):
        raise TypeError("subdivision levels must be an integer")
    if levels < 0:
        raise ValueError("subdivision levels must not be negative")
    if levels == 0:
        return asset
    vertices, faces, dense = asset.rest_vertices, asset.faces, asset.dense_weights()
    for _ in range(levels):
        operator, faces = loop_subdivision(faces, len(vertices))
        vertices, dense = operator @ vertices, operator @ dense
    indices, weights = _strongest(dense, asset.joint_indices.shape[1])
    return replace(
        asset,
        rest_vertices=np.asarray(vertices, dtype=np.float64),
        faces=faces,
        joint_indices=indices,
        joint_weights=weights,
    )


def original_resolution(
    fine: SkinnedHandAsset, source: SkinnedHandAsset
) -> SkinnedHandAsset:
    """``fine`` cut back to the mesh of ``source``, which it was subdivided from.

    The result has the faces of ``source`` and, at each of its vertices, the
    position and weights that vertex has in ``fine``: a coarse stand-in whose
    every vertex is a vertex of the fine skin. Returns ``fine`` itself when
    it was not subdivided.
    """
    count = len(source.rest_vertices)
    if len(fine.rest_vertices) == count:
        return fine
    if len(fine.rest_vertices) < count:
        raise ValueError("fine asset has fewer vertices than its source")
    return replace(
        fine,
        rest_vertices=fine.rest_vertices[:count].copy(),
        faces=source.faces,
        joint_indices=fine.joint_indices[:count].copy(),
        joint_weights=fine.joint_weights[:count].copy(),
    )
