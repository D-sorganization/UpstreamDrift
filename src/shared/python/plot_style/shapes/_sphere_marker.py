"""Sphere marker primitive.

A UV sphere whose bounding sphere has radius ``size_px / 2``. The
underlying mesh generation is delegated to
:func:`body_part_viz.shapes._mesh_primitives.make_uv_sphere` (DRY).
"""

from __future__ import annotations

import numpy as np

from ...body_part_viz.shapes._mesh_primitives import make_uv_sphere
from ..markers import MarkerShape, MarkerStyle

__all__ = ["SphereMarker"]


class SphereMarker:
    """Sphere marker shape renderer.

    Default tessellation is ``n_lon=16``, ``n_lat=8`` which yields
    ``144`` vertices and ``256`` triangles.
    """

    shape_id: str = MarkerShape.SPHERE.value

    def __init__(self, n_lon: int = 16, n_lat: int = 8) -> None:
        if not isinstance(n_lon, int) or isinstance(n_lon, bool):
            raise TypeError(f"n_lon must be int; got {type(n_lon).__name__}")
        if not isinstance(n_lat, int) or isinstance(n_lat, bool):
            raise TypeError(f"n_lat must be int; got {type(n_lat).__name__}")
        if n_lon < 3:
            raise ValueError(f"n_lon must be >= 3; got {n_lon}")
        if n_lat < 2:
            raise ValueError(f"n_lat must be >= 2; got {n_lat}")
        self._n_lon = n_lon
        self._n_lat = n_lat
        self._unit_vertices, self._faces = make_uv_sphere(n_lon, n_lat)

    def mesh(self, style: MarkerStyle) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        radius = float(style.size_px) / 2.0
        return self._unit_vertices * radius, self._faces.copy()
