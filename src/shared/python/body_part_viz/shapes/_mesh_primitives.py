"""Pure-NumPy mesh-primitive helpers for the body_part_viz shapes.

Each helper returns a ``(vertices, faces)`` pair where ``vertices`` has
shape ``(V, 3)`` (float64) and ``faces`` has shape ``(F, 3)`` (int64).

These helpers are deliberately backend-agnostic: nothing here imports
matplotlib, trimesh, or any rendering library. They are reused by the
concrete :class:`BodyPartShape` implementations.
"""

from __future__ import annotations

import numpy as np

__all__ = ["make_uv_sphere", "make_cylinder", "make_ellipsoid"]


def _validate_facets(n_facets: int, n_lat: int | None = None) -> None:
    if not isinstance(n_facets, int) or isinstance(n_facets, bool):
        raise TypeError(f"n_facets must be int; got {type(n_facets).__name__}")
    if n_facets < 3:
        raise ValueError(f"n_facets must be >= 3; got {n_facets}")
    if n_lat is not None:
        if not isinstance(n_lat, int) or isinstance(n_lat, bool):
            raise TypeError(f"n_lat must be int; got {type(n_lat).__name__}")
        if n_lat < 2:
            raise ValueError(f"n_lat must be >= 2; got {n_lat}")


def make_uv_sphere(n_lon: int, n_lat: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, faces)`` for a unit UV sphere.

    The sphere has ``n_lon`` longitudinal segments and ``n_lat`` latitudinal
    segments, yielding ``n_lon * (n_lat + 1)`` vertices.
    """
    _validate_facets(n_lon, n_lat)

    lats = np.linspace(0.0, np.pi, n_lat + 1)
    lons = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    sin_lat = np.sin(lat_grid)
    x = sin_lat * np.cos(lon_grid)
    y = sin_lat * np.sin(lon_grid)
    z = np.cos(lat_grid)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)

    faces: list[tuple[int, int, int]] = []
    for i in range(n_lat):
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            v00 = i * n_lon + j
            v01 = i * n_lon + j_next
            v10 = (i + 1) * n_lon + j
            v11 = (i + 1) * n_lon + j_next
            faces.append((v00, v10, v11))
            faces.append((v00, v11, v01))
    faces_arr = np.asarray(faces, dtype=np.int64)
    return vertices, faces_arr


def make_cylinder(
    length: float, radius: float, n_facets: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, faces)`` for a closed cylinder along the x-axis.

    The cylinder spans ``x in [0, length]``, with the side ring of radius
    ``radius`` in the ``yz``-plane. Each cap is a triangle fan around its
    centre. Total vertex count is ``2 * (n_facets + 1)``; total triangle
    count is ``4 * n_facets`` (``2 * n_facets`` for the side, ``n_facets``
    for each cap).
    """
    if length <= 0.0:
        raise ValueError(f"length must be > 0; got {length}")
    if radius <= 0.0:
        raise ValueError(f"radius must be > 0; got {radius}")
    _validate_facets(n_facets)

    angles = np.linspace(0.0, 2.0 * np.pi, n_facets, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    base_ring = np.stack([np.zeros(n_facets), radius * cos_a, radius * sin_a], axis=-1)
    top_ring = np.stack(
        [np.full(n_facets, length), radius * cos_a, radius * sin_a], axis=-1
    )
    base_centre = np.array([[0.0, 0.0, 0.0]])
    top_centre = np.array([[length, 0.0, 0.0]])

    vertices = np.concatenate(
        [base_ring, top_ring, base_centre, top_centre], axis=0
    ).astype(np.float64)
    base_centre_idx = 2 * n_facets
    top_centre_idx = 2 * n_facets + 1

    faces: list[tuple[int, int, int]] = []
    for j in range(n_facets):
        j_next = (j + 1) % n_facets
        v_b0 = j
        v_b1 = j_next
        v_t0 = n_facets + j
        v_t1 = n_facets + j_next
        faces.append((v_b0, v_t0, v_t1))
        faces.append((v_b0, v_t1, v_b1))
    for j in range(n_facets):
        j_next = (j + 1) % n_facets
        faces.append((base_centre_idx, j_next, j))
    for j in range(n_facets):
        j_next = (j + 1) % n_facets
        faces.append((top_centre_idx, n_facets + j, n_facets + j_next))

    return vertices, np.asarray(faces, dtype=np.int64)


def make_ellipsoid(
    a: float, b: float, c: float, n_lon: int, n_lat: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, faces)`` for an ellipsoid with semi-axes ``(a, b, c)``.

    Built from a unit UV sphere scaled by ``(a, b, c)``.
    """
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        raise ValueError(f"a, b, c must be > 0; got ({a}, {b}, {c})")
    vertices, faces = make_uv_sphere(n_lon, n_lat)
    vertices = vertices * np.array([a, b, c], dtype=np.float64)
    return vertices, faces
