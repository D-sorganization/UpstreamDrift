"""Cross (3D plus-sign) marker primitive.

Three orthogonal rectangular bars meeting at the origin, forming a 3D
``+`` shape. The long axis of each bar reaches ``+/-1`` so the
bounding sphere has radius 1 in unit form, scaled to ``size_px / 2``.

Each bar contributes 8 vertices and 12 triangles → 24 vertices and
36 triangles total.
"""

from __future__ import annotations

import numpy as np

from ..markers import MarkerShape, MarkerStyle

__all__ = ["CrossMarker"]


_BAR_FACES_TEMPLATE: np.ndarray = np.array(
    [
        [0, 3, 2], [0, 2, 1],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [0, 4, 7], [0, 7, 3],
    ],
    dtype=np.int64,
)


def _bar_vertices(long_axis: int, half_thickness: float) -> np.ndarray:
    """Build 8 vertices for an axis-aligned bar.

    The bar spans ``[-1, 1]`` along ``long_axis`` and
    ``[-half_thickness, half_thickness]`` along the two short axes.
    """
    h = half_thickness
    # ranges per axis
    spans = [(-h, h)] * 3
    spans[long_axis] = (-1.0, 1.0)
    (x0, x1), (y0, y1), (z0, z1) = spans
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )


class CrossMarker:
    """Cross / 3D plus-sign marker (24 vertices, 36 triangles).

    ``thickness`` is the full thickness of each bar in unit-radius
    coordinates (i.e. relative to a bar half-length of 1). Defaults to
    ``0.25``.
    """

    shape_id: str = MarkerShape.CROSS.value

    def __init__(self, thickness: float = 0.25) -> None:
        if not isinstance(thickness, (int, float)) or isinstance(thickness, bool):
            raise TypeError(
                f"thickness must be numeric; got {type(thickness).__name__}"
            )
        t = float(thickness)
        if not np.isfinite(t) or not 0.0 < t < 2.0:
            raise ValueError(
                f"thickness must be finite and in (0, 2); got {thickness!r}"
            )
        half = t / 2.0
        bars = [_bar_vertices(axis, half) for axis in range(3)]
        verts = np.concatenate(bars, axis=0)
        faces_chunks = [_BAR_FACES_TEMPLATE + (8 * i) for i in range(3)]
        self._unit_vertices = verts
        self._faces = np.concatenate(faces_chunks, axis=0)

    def mesh(self, style: MarkerStyle) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(style, MarkerStyle):
            raise TypeError(
                f"style must be MarkerStyle; got {type(style).__name__}"
            )
        radius = float(style.size_px) / 2.0
        return self._unit_vertices * radius, self._faces.copy()
