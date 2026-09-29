"""Diamond (octahedron) marker primitive.

A regular octahedron with vertices on the unit axes. Bounding sphere
radius equals ``size_px / 2``. 6 vertices, 8 triangles.
"""

from __future__ import annotations

import numpy as np

from ..markers import MarkerShape, MarkerStyle

__all__ = ["DiamondMarker"]


_UNIT_VERTICES: np.ndarray = np.array(
    [
        [+1.0, 0.0, 0.0],  # +x  (idx 0)
        [-1.0, 0.0, 0.0],  # -x  (idx 1)
        [0.0, +1.0, 0.0],  # +y  (idx 2)
        [0.0, -1.0, 0.0],  # -y  (idx 3)
        [0.0, 0.0, +1.0],  # +z  (idx 4)
        [0.0, 0.0, -1.0],  # -z  (idx 5)
    ],
    dtype=np.float64,
)

# 8 outward-facing triangles, one per octant.
_UNIT_FACES: np.ndarray = np.array(
    [
        [0, 2, 4],  # +x +y +z
        [2, 1, 4],  # -x +y +z
        [1, 3, 4],  # -x -y +z
        [3, 0, 4],  # +x -y +z
        [2, 0, 5],  # +x +y -z
        [1, 2, 5],  # -x +y -z
        [3, 1, 5],  # -x -y -z
        [0, 3, 5],  # +x -y -z
    ],
    dtype=np.int64,
)


class DiamondMarker:
    """Octahedral diamond marker (6 vertices, 8 triangles)."""

    shape_id: str = MarkerShape.DIAMOND.value

    def __init__(self) -> None:
        self._unit_vertices = _UNIT_VERTICES
        self._faces = _UNIT_FACES

    def mesh(self, style: MarkerStyle) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        radius = float(style.size_px) / 2.0
        return self._unit_vertices * radius, self._faces.copy()
