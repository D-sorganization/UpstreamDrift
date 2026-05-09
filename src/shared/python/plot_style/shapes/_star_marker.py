"""Star marker primitive.

A bipyramidal 3D star with ``n_points`` outer spikes around the
equator. The equator alternates between ``n_points`` outer vertices on
the unit circle and ``n_points`` inner vertices at radius
``inner_ratio``. Two apex vertices sit at ``+z`` and ``-z``, both on
the unit sphere — so the bounding sphere has radius 1.

Vertex layout (default ``n_points=5``):

* indices ``[0, 2, 4, 6, 8]`` — outer equator points (on unit circle)
* indices ``[1, 3, 5, 7, 9]`` — inner equator points
* index ``10`` — top apex (``+z``)
* index ``11`` — bottom apex (``-z``)

Total: ``2 * n_points + 2`` vertices, ``4 * n_points`` triangles.
"""

from __future__ import annotations

import numpy as np

from ..markers import MarkerShape, MarkerStyle

__all__ = ["StarMarker"]


class StarMarker:
    """3D star bipyramid marker."""

    shape_id: str = MarkerShape.STAR.value

    def __init__(self, n_points: int = 5, inner_ratio: float = 0.4) -> None:
        if not isinstance(n_points, int) or isinstance(n_points, bool):
            raise TypeError(f"n_points must be int; got {type(n_points).__name__}")
        if n_points < 3:
            raise ValueError(f"n_points must be >= 3; got {n_points}")
        if not isinstance(inner_ratio, (int, float)) or isinstance(inner_ratio, bool):
            raise TypeError(
                f"inner_ratio must be numeric; got {type(inner_ratio).__name__}"
            )
        ir = float(inner_ratio)
        if not np.isfinite(ir) or not 0.0 < ir < 1.0:
            raise ValueError(
                f"inner_ratio must be finite and in (0, 1); got {inner_ratio!r}"
            )

        n_eq = 2 * n_points
        # Equator: alternate outer (radius 1) / inner (radius ir).
        thetas = np.arange(n_eq, dtype=np.float64) * (np.pi / n_points)
        radii = np.where(np.arange(n_eq) % 2 == 0, 1.0, ir)
        eq_xy = np.stack([radii * np.cos(thetas), radii * np.sin(thetas)], axis=-1)
        equator = np.concatenate([eq_xy, np.zeros((n_eq, 1))], axis=1)
        top_apex = np.array([[0.0, 0.0, 1.0]])
        bot_apex = np.array([[0.0, 0.0, -1.0]])
        vertices = np.concatenate([equator, top_apex, bot_apex], axis=0).astype(
            np.float64
        )

        top_idx = n_eq
        bot_idx = n_eq + 1

        faces: list[tuple[int, int, int]] = []
        # Triangles connecting each equator edge to the two apexes.
        for j in range(n_eq):
            j_next = (j + 1) % n_eq
            faces.append((top_idx, j, j_next))
            faces.append((bot_idx, j_next, j))

        self._n_points = n_points
        self._inner_ratio = ir
        self._unit_vertices = vertices
        self._faces = np.asarray(faces, dtype=np.int64)

    def mesh(self, style: MarkerStyle) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(style, MarkerStyle):
            raise TypeError(f"style must be MarkerStyle; got {type(style).__name__}")
        radius = float(style.size_px) / 2.0
        return self._unit_vertices * radius, self._faces.copy()
