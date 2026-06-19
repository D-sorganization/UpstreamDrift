"""Geometry builders for golf simulation rendering.

Pure-numpy constructors for the meshes and polylines used by the putting
green, driving range, and ball-flight viewers.  They return plain numpy
arrays (vertices / faces / colours) so they can be unit-tested headlessly and
fed to any backend (``pyqtgraph.opengl``, matplotlib, PyVista, ...).

Design by Contract:
    * Public functions validate argument shapes and raise ``ValueError`` with
      descriptive messages.
    * Vertex arrays are ``(N, 3)`` float; face arrays are ``(M, 3)`` int with
      indices in ``[0, N)``.
"""

from __future__ import annotations

import numpy as np

from src.shared.python.golf_viz._colors import terrain_colors

__all__ = [
    "circle_fan_vertices",
    "disc_mesh",
    "flagstick_lines",
    "grid_surface_mesh",
    "rect_vertices",
]


def rect_vertices(
    x: float, y: float, width: float, height: float, *, z: float = 0.0
) -> np.ndarray:
    """Return the 6 vertices (two triangles) of an axis-aligned rectangle."""
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"rectangle size must be positive; got {width}x{height}")
    x2, y2 = x + width, y + height
    return np.array(
        [
            [x, y, z],
            [x2, y, z],
            [x, y2, z],
            [x, y2, z],
            [x2, y, z],
            [x2, y2, z],
        ],
        dtype=float,
    )


def circle_fan_vertices(
    cx: float, cy: float, radius: float, *, z: float = 0.0, segments: int = 32
) -> np.ndarray:
    """Return a flat triangle-soup disc as ``(3 * (segments - 1), 3)`` vertices.

    Matches the legacy renderer contract (one triangle per arc segment, fanned
    from the centre) so it can drop in for hand-rolled circle builders.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive; got {radius}")
    if segments < 3:
        raise ValueError(f"need at least 3 segments; got {segments}")
    angles = np.linspace(0.0, 2.0 * np.pi, segments)
    ring_x = cx + radius * np.cos(angles)
    ring_y = cy + radius * np.sin(angles)
    verts: list[list[float]] = []
    for i in range(segments - 1):
        verts.append([cx, cy, z])
        verts.append([ring_x[i], ring_y[i], z])
        verts.append([ring_x[i + 1], ring_y[i + 1], z])
    return np.array(verts, dtype=float)


def disc_mesh(
    center: tuple[float, float],
    radius: float,
    *,
    z: float = 0.0,
    segments: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an indexed triangle-fan disc as ``(vertices, faces)``.

    ``vertices`` is ``(segments + 1, 3)`` (centre followed by the ring);
    ``faces`` is ``(segments, 3)`` int indices.
    """
    if radius <= 0.0:
        raise ValueError(f"radius must be positive; got {radius}")
    if segments < 3:
        raise ValueError(f"need at least 3 segments; got {segments}")
    cx, cy = center
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.column_stack(
        [
            cx + radius * np.cos(angles),
            cy + radius * np.sin(angles),
            np.full(segments, z),
        ]
    )
    vertices = np.vstack([[cx, cy, z], ring])
    faces = np.array(
        [[0, 1 + i, 1 + (i + 1) % segments] for i in range(segments)], dtype=int
    )
    return vertices, faces


def grid_surface_mesh(
    xs: np.ndarray,
    ys: np.ndarray,
    zz: np.ndarray,
    *,
    colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate an elevation grid into ``(vertices, faces, colors)``.

    Args:
        xs: ``(nx,)`` x sample coordinates (monotonic).
        ys: ``(ny,)`` y sample coordinates (monotonic).
        zz: ``(ny, nx)`` elevation grid (row-major over ``(y, x)``).
        colors: Optional ``(nx * ny, 4)`` per-vertex RGBA. When omitted, a
            golf turf gradient keyed on elevation is generated.

    Returns:
        ``vertices`` ``(nx * ny, 3)``, ``faces`` ``(2 * (nx - 1) * (ny - 1), 3)``
        int, and per-vertex ``colors`` ``(nx * ny, 4)``.
    """
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    zz = np.asarray(zz, dtype=float)
    nx, ny = xs.size, ys.size
    if nx < 2 or ny < 2:
        raise ValueError(f"grid must be at least 2x2; got {ny}x{nx}")
    if zz.shape != (ny, nx):
        raise ValueError(
            f"elevation grid shape {zz.shape} must equal (len(ys), len(xs)) = ({ny}, {nx})"
        )

    grid_x, grid_y = np.meshgrid(xs, ys)
    vertices = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1), zz.reshape(-1)])

    # Two triangles per grid cell, indices row-major over (y, x).
    i = np.arange(nx - 1)
    j = np.arange(ny - 1)
    jj, ii = np.meshgrid(j, i, indexing="ij")
    top_left = (jj * nx + ii).reshape(-1)
    top_right = top_left + 1
    bot_left = top_left + nx
    bot_right = bot_left + 1
    tri1 = np.column_stack([top_left, top_right, bot_left])
    tri2 = np.column_stack([top_right, bot_right, bot_left])
    faces = np.empty((tri1.shape[0] + tri2.shape[0], 3), dtype=int)
    faces[0::2] = tri1
    faces[1::2] = tri2

    if colors is None:
        colors = terrain_colors(zz.reshape(-1))
    else:
        colors = np.asarray(colors, dtype=float)
        if colors.shape != (nx * ny, 4):
            raise ValueError(f"colors shape {colors.shape} must equal ({nx * ny}, 4)")
    return vertices, faces, colors


def flagstick_lines(
    base: tuple[float, float], *, z: float = 0.0, height: float = 1.5
) -> np.ndarray:
    """Return the two endpoints ``(2, 3)`` of a vertical flagstick."""
    if height <= 0.0:
        raise ValueError(f"flagstick height must be positive; got {height}")
    bx, by = base
    return np.array([[bx, by, z], [bx, by, z + height]], dtype=float)
