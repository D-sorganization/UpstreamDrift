"""Closed grip-cylinder mesh for elastic-foundation contact (issue #11739, OSV-7).

OpenSim's ``ElasticFoundationForce`` needs a *closed*, water-tight triangle
mesh on the deformable side; an open (uncapped) tube throws in the
``ContactMesh`` constructor with an opaque Simbody error.  This module builds
the capped grip cylinder from the shared grip geometry and validates any mesh
up front, so a bad mesh fails here with a clear ``ValueError``.

Pure NumPy; engine-agnostic.  Faces are counter-clockwise seen from outside
(outward normals), as OpenSim expects.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

__all__ = ["capped_cylinder_mesh", "validate_closed_mesh", "write_obj"]


def validate_closed_mesh(vertices: np.ndarray, faces: np.ndarray) -> None:
    """Require a closed, consistently outward-oriented triangle mesh.

    Checks, in order: array shapes and finite vertices, face indices in range,
    no repeated vertex in a face, every directed edge used once and its
    reverse used once (closed, orientable 2-manifold), and positive enclosed
    volume (outward normals).

    Raises:
        ValueError: with the first violated condition named.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 4:
        raise ValueError("vertices must have shape (n >= 4, 3)")
    if not np.isfinite(v).all():
        raise ValueError("vertices must be finite")
    if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] < 4:
        raise ValueError("faces must have shape (m >= 4, 3)")
    if not np.issubdtype(f.dtype, np.integer):
        raise ValueError("faces must be integer vertex indices")
    if f.min() < 0 or f.max() >= v.shape[0]:
        raise ValueError("face index out of range")
    if (
        (f[:, 0] == f[:, 1]).any()
        or (f[:, 1] == f[:, 2]).any()
        or (f[:, 0] == f[:, 2]).any()
    ):
        raise ValueError("degenerate face (repeated vertex)")
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    keys, counts = np.unique(directed, axis=0, return_counts=True)
    if (counts > 1).any():
        raise ValueError("mesh is not a manifold: a directed edge is used twice")
    forward = {(int(a), int(b)) for a, b in keys}
    open_edges = [e for e in forward if (e[1], e[0]) not in forward]
    if open_edges:
        raise ValueError(
            f"mesh is not closed (water-tight): {len(open_edges)} boundary "
            "edge(s), e.g. an uncapped cylinder; ElasticFoundationForce needs "
            "a closed mesh"
        )
    a, b, c = (v[f[:, k]] for k in range(3))
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0
    if volume <= 0.0:
        raise ValueError("mesh normals point inward (non-positive enclosed volume)")


def capped_cylinder_mesh(
    radius_m: float,
    axis_point_m: np.ndarray,
    axis: np.ndarray,
    radial: np.ndarray,
    axial_range_m: tuple[float, float],
    segments: int = 96,
    ring_pitch_m: float = 1.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed cylinder of ``radius_m`` about the line ``axis_point + s * axis``.

    ``axial_range_m`` is the ``(lo, hi)`` range of ``s``; ``radial`` is a unit
    vector perpendicular to ``axis`` that fixes where ring vertex 0 sits.  The
    side is split into rings no farther apart than ``ring_pitch_m`` (the
    elastic foundation integrates over faces, so they must be small against
    the contact patch).  Postcondition: the mesh passes
    :func:`validate_closed_mesh`.

    Raises:
        ValueError: on a non-positive radius, ``hi <= lo``, fewer than 8
            segments, a non-positive pitch, or a non-orthonormal axis/radial.
    """
    lo, hi = axial_range_m
    if not (math.isfinite(radius_m) and radius_m > 0.0):
        raise ValueError("radius_m must be positive and finite")
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError("axial_range_m must satisfy hi > lo")
    if segments < 8 or not (math.isfinite(ring_pitch_m) and ring_pitch_m > 0.0):
        raise ValueError("segments >= 8 and ring_pitch_m > 0 required")
    ax = np.asarray(axis, dtype=float)
    e1 = np.asarray(radial, dtype=float)
    if (
        abs(np.linalg.norm(ax) - 1.0) > 1e-9
        or abs(np.linalg.norm(e1) - 1.0) > 1e-9
        or abs(float(ax @ e1)) > 1e-9
    ):
        raise ValueError("axis and radial must be orthonormal")
    e2 = np.cross(ax, e1)
    origin = np.asarray(axis_point_m, dtype=float)
    rings = int(math.ceil((hi - lo) / ring_pitch_m))
    s = np.linspace(lo, hi, rings + 1)
    ang = 2.0 * math.pi * np.arange(segments) / segments
    ring_xy = radius_m * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
    side = [origin + sv * ax + ring_xy for sv in s]
    verts = np.vstack([*side, origin + lo * ax, origin + hi * ax])
    c0, c1 = verts.shape[0] - 2, verts.shape[0] - 1
    k = np.arange(segments)
    nxt = (k + 1) % segments
    faces = []
    for j in range(rings):
        a, b = j * segments + k, j * segments + nxt
        c, d = a + segments, b + segments
        faces.append(np.column_stack([a, b, d]))
        faces.append(np.column_stack([a, d, c]))
    faces.append(np.column_stack([np.full(segments, c0), nxt, k]))
    top = rings * segments
    faces.append(np.column_stack([np.full(segments, c1), top + k, top + nxt]))
    out = np.vstack(faces).astype(np.int64)
    validate_closed_mesh(verts, out)
    return verts, out


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a validated mesh as a Wavefront OBJ (1-based indices).

    Raises:
        ValueError: if the mesh is not closed (see :func:`validate_closed_mesh`).
    """
    validate_closed_mesh(vertices, faces)
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in np.asarray(vertices, float)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in np.asarray(faces)]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
