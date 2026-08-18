"""Analytic verification solids and their closed-form inertia (#8609).

These are *code-verification* primitives: solids whose volume, centroid
and inertia tensor are known in closed form, used to show that the
divergence-theorem integrator in :mod:`.mass_properties` is exact on
polyhedra and convergent on tessellated curved surfaces.  They are part
of the library rather than the test suite because a design tool has to
be able to re-verify itself in the field.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mesh import TriangleMesh, signed_volume_m3

__all__ = [
    "box_inertia_about_centroid",
    "box_mesh",
    "cylinder_inertia_about_centroid",
    "cylinder_mesh",
    "icosphere_mesh",
    "sphere_inertia_about_centroid",
    "tetrahedron_mesh",
]


def _oriented(vertices: NDArray[np.float64], faces: NDArray[np.int64]) -> TriangleMesh:
    """Return a mesh whose winding encloses a positive volume."""
    mesh = TriangleMesh(vertices, faces)
    if signed_volume_m3(mesh) < 0.0:
        return TriangleMesh(vertices, faces[:, ::-1])
    return mesh


def _require_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")


def box_mesh(
    length_x_m: float,
    length_y_m: float,
    length_z_m: float,
    centre: ArrayLike = (0.0, 0.0, 0.0),
) -> TriangleMesh:
    """An axis-aligned rectangular box as 8 vertices and 12 triangles."""
    _require_positive("length_x_m", length_x_m)
    _require_positive("length_y_m", length_y_m)
    _require_positive("length_z_m", length_z_m)
    half = np.array([length_x_m, length_y_m, length_z_m], dtype=np.float64) / 2.0
    offset = np.asarray(centre, dtype=np.float64).reshape(3)
    signs = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    vertices = signs * half + offset
    faces = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ],
        dtype=np.int64,
    )
    return _oriented(vertices, faces)


def tetrahedron_mesh(vertices: ArrayLike) -> TriangleMesh:
    """A tetrahedron from four corners, wound outward."""
    corners = np.asarray(vertices, dtype=np.float64)
    if corners.shape != (4, 3):
        raise ValueError(f"a tetrahedron needs (4, 3) vertices, got {corners.shape}")
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int64,
    )
    mesh = _oriented(corners, faces)
    if abs(signed_volume_m3(mesh)) <= 0.0:
        raise ValueError("tetrahedron vertices are coplanar (zero volume)")
    return mesh


def _unit_midpoint(
    points: list[NDArray[np.float64]],
    cache: dict[tuple[int, int], int],
    first: int,
    second: int,
) -> int:
    """Index of the unit-normalised midpoint of two sphere vertices."""
    key = (min(first, second), max(first, second))
    if key not in cache:
        middle = points[first] + points[second]
        points.append(middle / np.linalg.norm(middle))
        cache[key] = len(points) - 1
    return cache[key]


def icosphere_mesh(radius_m: float, subdivisions: int = 2) -> TriangleMesh:
    """A geodesic sphere: an icosahedron with midpoint subdivision.

    Args:
        radius_m: Sphere radius.
        subdivisions: Number of midpoint refinements (0 = icosahedron).
    """
    _require_positive("radius_m", radius_m)
    if subdivisions < 0 or subdivisions > 6:
        raise ValueError(f"subdivisions must lie in [0, 6], got {subdivisions}")

    golden = (1.0 + math.sqrt(5.0)) / 2.0
    base = np.array(
        [
            [-1, golden, 0],
            [1, golden, 0],
            [-1, -golden, 0],
            [1, -golden, 0],
            [0, -1, golden],
            [0, 1, golden],
            [0, -1, -golden],
            [0, 1, -golden],
            [golden, 0, -1],
            [golden, 0, 1],
            [-golden, 0, -1],
            [-golden, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip

    points = [row / np.linalg.norm(row) for row in base]
    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}
        refined: list[tuple[int, int, int]] = []
        for one, two, three in faces:
            a = _unit_midpoint(points, midpoints, one, two)
            b = _unit_midpoint(points, midpoints, two, three)
            c = _unit_midpoint(points, midpoints, three, one)
            refined.extend([(one, a, c), (two, b, a), (three, c, b), (a, b, c)])
        faces = refined

    vertices = np.asarray(points, dtype=np.float64) * radius_m
    return _oriented(vertices, np.asarray(faces, dtype=np.int64))


def cylinder_mesh(
    radius_m: float, height_m: float, n_segments: int = 32
) -> TriangleMesh:
    """A closed right circular cylinder about the z axis, centred at 0."""
    _require_positive("radius_m", radius_m)
    _require_positive("height_m", height_m)
    if n_segments < 3:
        raise ValueError(f"n_segments must be at least 3, got {n_segments}")

    angles = np.linspace(0.0, 2.0 * math.pi, n_segments, endpoint=False)
    ring = np.column_stack(
        [radius_m * np.cos(angles), radius_m * np.sin(angles), np.zeros(n_segments)]
    )
    bottom = ring - np.array([0.0, 0.0, height_m / 2.0])
    top = ring + np.array([0.0, 0.0, height_m / 2.0])
    vertices = np.vstack(
        [
            bottom,
            top,
            [[0.0, 0.0, -height_m / 2.0]],
            [[0.0, 0.0, height_m / 2.0]],
        ]
    )
    bottom_centre = 2 * n_segments
    top_centre = bottom_centre + 1

    faces: list[tuple[int, int, int]] = []
    for index in range(n_segments):
        nxt = (index + 1) % n_segments
        faces.append((index, nxt, n_segments + nxt))
        faces.append((index, n_segments + nxt, n_segments + index))
        faces.append((bottom_centre, nxt, index))
        faces.append((top_centre, n_segments + index, n_segments + nxt))
    return _oriented(vertices, np.asarray(faces, dtype=np.int64))


def box_inertia_about_centroid(
    mass_kg: float, length_x_m: float, length_y_m: float, length_z_m: float
) -> NDArray[np.float64]:
    """Closed-form inertia tensor of a uniform rectangular box."""
    x2, y2, z2 = length_x_m**2, length_y_m**2, length_z_m**2
    return (mass_kg / 12.0) * np.diag([y2 + z2, x2 + z2, x2 + y2])


def sphere_inertia_about_centroid(
    mass_kg: float, radius_m: float
) -> NDArray[np.float64]:
    """Closed-form inertia tensor of a uniform solid sphere."""
    return (2.0 / 5.0) * mass_kg * radius_m**2 * np.eye(3)


def cylinder_inertia_about_centroid(
    mass_kg: float, radius_m: float, height_m: float
) -> NDArray[np.float64]:
    """Closed-form inertia tensor of a uniform solid cylinder (z axis)."""
    lateral = mass_kg * (3.0 * radius_m**2 + height_m**2) / 12.0
    axial = mass_kg * radius_m**2 / 2.0
    return np.diag([lateral, lateral, axial])
