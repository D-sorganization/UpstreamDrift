"""Mesh steps of the CC0 body skin layers (#15299)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.model_appearance import body_skin_mesh as mesh

pytestmark = pytest.mark.unit


def test_clip_mesh_cuts_on_the_level_set() -> None:
    points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    kept, cut = mesh.clip_mesh(points, faces, 0.25 - points[:, 0])
    assert np.isclose(kept[:, 0].max(), 0.25) and len(cut) == 3
    area = np.cross(
        kept[cut[:, 1]] - kept[cut[:, 0]], kept[cut[:, 2]] - kept[cut[:, 0]]
    )
    assert np.isclose(0.5 * area[:, 2].sum(), 0.25) and (area[:, 2] > 0).all()
    with pytest.raises(ValueError):
        mesh.clip_mesh(points, faces, -np.ones(4))
    with pytest.raises(ValueError):
        mesh.clip_mesh(points, faces, np.ones(3))


def test_open_edges_normals_and_relaxation_of_a_fan() -> None:
    rim = np.array([[1.0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    points = np.vstack([[[0.3, 0.2, 0.0]], rim])
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]])
    edges = mesh.boundary_edges(faces)
    assert sorted(map(tuple, edges.tolist())) == [(1, 2), (2, 3), (3, 4), (4, 1)]
    assert np.allclose(mesh.vertex_normals(points, faces), [0.0, 0.0, 1.0])
    fixed = np.zeros(5, dtype=bool)
    fixed[edges] = True
    moved = mesh.relax(points, faces, fixed, 1.0)
    assert np.allclose(moved[0], 0.0) and np.array_equal(moved[1:], rim)
    assert np.array_equal(mesh.relax(points, faces, ~fixed | fixed, 1.0), points)
