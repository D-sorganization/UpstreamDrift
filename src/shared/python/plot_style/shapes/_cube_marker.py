"""Cube marker primitive.

An axis-aligned cube whose bounding sphere has radius ``size_px / 2``,
i.e. its half-edge equals ``size_px / (2 * sqrt(3))``. The cube has
8 vertices and 12 triangles.
"""

from __future__ import annotations

import math

import numpy as np

from ..markers import MarkerShape, MarkerStyle

__all__ = ["CubeMarker"]


# Unit cube: vertices at +/- 1 on each axis (bounding sphere radius sqrt(3)).
_UNIT_VERTICES: np.ndarray = np.array(
    [
        [-1.0, -1.0, -1.0],
        [+1.0, -1.0, -1.0],
        [+1.0, +1.0, -1.0],
        [-1.0, +1.0, -1.0],
        [-1.0, -1.0, +1.0],
        [+1.0, -1.0, +1.0],
        [+1.0, +1.0, +1.0],
        [-1.0, +1.0, +1.0],
    ],
    dtype=np.float64,
)

# 6 faces × 2 triangles each, outward-facing winding.
_UNIT_FACES: np.ndarray = np.array(
    [
        [0, 3, 2],
        [0, 2, 1],  # -z
        [4, 5, 6],
        [4, 6, 7],  # +z
        [0, 1, 5],
        [0, 5, 4],  # -y
        [2, 3, 7],
        [2, 7, 6],  # +y
        [1, 2, 6],
        [1, 6, 5],  # +x
        [0, 4, 7],
        [0, 7, 3],  # -x
    ],
    dtype=np.int64,
)

_INV_SQRT3 = 1.0 / math.sqrt(3.0)


class CubeMarker:
    """Cube marker shape renderer (8 vertices, 12 triangles)."""

    shape_id: str = MarkerShape.CUBE.value

    def __init__(self) -> None:
        self._unit_vertices = _UNIT_VERTICES * _INV_SQRT3
        self._faces = _UNIT_FACES

    def mesh(self, style: MarkerStyle) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        radius = float(style.size_px) / 2.0
        return self._unit_vertices * radius, self._faces.copy()
