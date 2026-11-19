"""Signed triangle distance must detect deep interiors, not vertex proximity."""

import numpy as np
import pytest

from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.mesh_penetration import signed_mesh_distances

pytestmark = pytest.mark.unit


def test_cube_distinguishes_deep_inside_outside_and_face_contact():
    vertices = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    mesh = Mesh(vertices, faces)
    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 0.0]]
    )
    expected = [-1.0, 1.0, 0.0, np.sqrt(2.0)]
    np.testing.assert_allclose(
        signed_mesh_distances(points, mesh), expected, atol=1e-12
    )
    # A global winding reversal changes no containment or distance result.
    np.testing.assert_allclose(
        signed_mesh_distances(points, Mesh(vertices, faces[:, ::-1])),
        expected,
        atol=1e-12,
    )


def test_open_triangle_surface_is_refused():
    mesh = Mesh(np.eye(3), np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="closed"):
        signed_mesh_distances(np.zeros((1, 3)), mesh)


def test_concave_notch_is_outside_even_inside_the_convex_hull():
    outline = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
    vertices = np.vstack(
        [np.column_stack([outline, np.full(6, z)]) for z in (0.0, 1.0)]
    )
    cap = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]])
    sides = []
    for i in range(6):
        j = (i + 1) % 6
        sides.extend([[i, j, j + 6], [i, j + 6, i + 6]])
    mesh = Mesh(vertices, np.vstack([cap[:, ::-1], cap + 6, sides]))
    points = np.array([[0.5, 0.5, 0.5], [1.4, 1.4, 0.5]])
    np.testing.assert_allclose(signed_mesh_distances(points, mesh), [-0.5, 0.4])
