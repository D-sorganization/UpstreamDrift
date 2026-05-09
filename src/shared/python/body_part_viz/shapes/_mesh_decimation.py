"""Mesh decimation utilities for :mod:`body_part_viz.shapes`.

Why a separate impl from
``humanoid_character_builder/mesh/_cg_decimation.py``:

The character-builder decimator targets *triangle counts* and returns a
``CollisionGeometryResult`` wrapper coupled to the collision-geometry
pipeline (volume preservation, primitive-fit ratios, hybrid fallback to
voxel marching-cubes). The body-part-viz contract here targets a
*vertex* budget, must return canonical NumPy arrays, and prefers a fast
uniform fallback over voxel-remeshing on failure. The contracts are
genuinely incompatible, so we ship a focused helper rather than wedging
the two pipelines together.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import trimesh

__all__ = ["DecimationStrategy", "decimate"]

DecimationStrategy = Literal["quadric", "uniform"]

_logger = logging.getLogger(__name__)


def _to_trimesh(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _target_face_count(
    n_vertices: int,
    n_faces: int,
    max_vertices: int,
) -> int:
    """Estimate face budget proportional to vertex budget, min 4.

    Precondition: ``n_vertices > 0`` (callers guarantee this via shape
    validation in :func:`decimate`).
    """
    ratio = max_vertices / n_vertices
    return max(4, int(round(n_faces * ratio)))


def _quadric_decimate(
    mesh: trimesh.Trimesh,
    max_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_faces = _target_face_count(len(mesh.vertices), len(mesh.faces), max_vertices)
    simplified = mesh.simplify_quadric_decimation(target_faces)
    return (
        np.asarray(simplified.vertices, dtype=np.float64),
        np.asarray(simplified.faces, dtype=np.int64),
    )


def _uniform_decimate(
    mesh: trimesh.Trimesh,
    max_vertices: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform fallback: keep every k-th face, then drop unused vertices.

    Iteratively shrinks the face budget until the surviving unique-vertex
    count is within ``max_vertices`` (closed manifolds typically have
    ~2x the vertices of an isolated face strip, so a single estimate
    can overshoot).
    """
    n_faces = len(mesh.faces)
    all_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    all_faces = np.asarray(mesh.faces, dtype=np.int64)

    target_faces = _target_face_count(len(mesh.vertices), n_faces, max_vertices)
    if target_faces >= n_faces:
        return all_vertices, all_faces

    sub_faces = all_faces
    used = np.arange(len(all_vertices))
    for _ in range(10):
        stride = max(1, n_faces // target_faces)
        keep = np.arange(0, n_faces, stride)[:target_faces]
        sub_faces = all_faces[keep]
        used = np.unique(sub_faces)
        if len(used) <= max_vertices:
            break
        # Overshot: tighten budget proportionally and retry.
        target_faces = max(4, (target_faces * max_vertices) // (len(used) + 1))

    remap = -np.ones(int(used.max()) + 1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    return all_vertices[used], remap[sub_faces]


def decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_vertices: int,
    strategy: DecimationStrategy = "quadric",
) -> tuple[np.ndarray, np.ndarray]:
    """Decimate a mesh to ``<= max_vertices`` vertices.

    Parameters
    ----------
    vertices:
        ``(V, 3)`` float array.
    faces:
        ``(F, 3)`` int array.
    max_vertices:
        Strict upper bound on the returned vertex count. Must be ``>= 4``.
    strategy:
        ``"quadric"`` (default) attempts edge-collapse decimation and
        falls back to uniform if it raises or overshoots. ``"uniform"``
        skips quadric entirely.

    Returns
    -------
    (vertices, faces):
        New canonical NumPy arrays. If the input already fits the budget
        the inputs are returned unchanged.
    """
    if max_vertices < 4:
        raise ValueError(f"max_vertices must be >= 4; got {max_vertices}")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3); got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3); got {faces.shape}")

    if len(vertices) <= max_vertices:
        return vertices, faces

    mesh = _to_trimesh(vertices, faces)

    if strategy == "quadric":
        try:
            new_v, new_f = _quadric_decimate(mesh, max_vertices)
            if len(new_v) <= max_vertices and len(new_v) > 0:
                return new_v, new_f
            _logger.warning(
                "quadric decimation overshot budget (%d > %d); falling back to uniform",
                len(new_v),
                max_vertices,
            )
        except (
            ValueError,
            RuntimeError,
            IndexError,
            AttributeError,
            ImportError,
            ModuleNotFoundError,
        ) as exc:
            _logger.warning(
                "quadric decimation failed (%s); falling back to uniform",
                exc.__class__.__name__,
            )

    return _uniform_decimate(mesh, max_vertices)
