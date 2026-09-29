"""Cylinder shape: closed cylinder along the local x-axis."""

from __future__ import annotations

import numpy as np

from .._types import FittedShape
from ._mesh_primitives import make_cylinder
from ._transform import apply_fitted_to_rest_vertices

__all__ = ["CylinderShape"]


class CylinderShape:
    """A closed cylinder of length ``length`` and radius ``radius``.

    The local frame has the cylinder axis along x; the cap rings lie in
    the ``yz``-plane. Vertex count is ``2 * (n_facets + 1)``; triangle
    count is ``4 * n_facets``.
    """

    shape_id: str
    rest_dimensions: tuple[float, ...]

    def __init__(
        self,
        length: float = 1.0,
        radius: float = 0.05,
        n_facets: int = 16,
        *,
        shape_id: str = "cylinder",
    ) -> None:
        if not np.isfinite(float(length)) or float(length) <= 0.0:
            raise ValueError(f"length must be finite and > 0; got {length}")
        if not np.isfinite(float(radius)) or float(radius) <= 0.0:
            raise ValueError(f"radius must be finite and > 0; got {radius}")
        if not isinstance(n_facets, int) or isinstance(n_facets, bool):
            raise TypeError(f"n_facets must be int; got {type(n_facets).__name__}")
        if n_facets < 3:
            raise ValueError(f"n_facets must be >= 3; got {n_facets}")
        if not isinstance(shape_id, str) or not shape_id:
            raise ValueError(f"shape_id must be non-empty str; got {shape_id!r}")

        self.shape_id = shape_id
        self.rest_dimensions = (float(length), float(radius))
        self._n_facets = n_facets
        self._vertices, self._faces = make_cylinder(
            float(length), float(radius), n_facets
        )

    @property
    def n_facets(self) -> int:
        return self._n_facets

    def vertices_at_rest(self) -> np.ndarray:
        return self._vertices.copy()

    def faces(self) -> np.ndarray:
        return self._faces.copy()

    def transform(self, fitted: FittedShape) -> np.ndarray:
        return apply_fitted_to_rest_vertices(self._vertices, fitted)
