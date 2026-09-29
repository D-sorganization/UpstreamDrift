"""Ellipsoid shape: UV-sphere with three independent semi-axes."""

from __future__ import annotations

import numpy as np

from .._types import FittedShape
from ._mesh_primitives import make_ellipsoid
from ._transform import apply_fitted_to_rest_vertices

__all__ = ["EllipsoidShape"]


class EllipsoidShape:
    """An ellipsoid with semi-axes ``(a, b, c)`` along the local x/y/z."""

    shape_id: str
    rest_dimensions: tuple[float, ...]

    def __init__(
        self,
        a: float,
        b: float,
        c: float,
        n_lon: int = 16,
        n_lat: int = 8,
        *,
        shape_id: str = "ellipsoid",
    ) -> None:
        for name, value in (("a", a), ("b", b), ("c", c)):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0; got {value}")
        for name, value in (("n_lon", n_lon), ("n_lat", n_lat)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be int; got {type(value).__name__}")
        if n_lon < 3:
            raise ValueError(f"n_lon must be >= 3; got {n_lon}")
        if n_lat < 2:
            raise ValueError(f"n_lat must be >= 2; got {n_lat}")
        if not isinstance(shape_id, str) or not shape_id:
            raise ValueError(f"shape_id must be non-empty str; got {shape_id!r}")

        self.shape_id = shape_id
        self.rest_dimensions = (float(a), float(b), float(c))
        self._n_lon = n_lon
        self._n_lat = n_lat
        self._vertices, self._faces = make_ellipsoid(
            float(a), float(b), float(c), n_lon, n_lat
        )

    @property
    def n_lon(self) -> int:
        return self._n_lon

    @property
    def n_lat(self) -> int:
        return self._n_lat

    def vertices_at_rest(self) -> np.ndarray:
        return self._vertices.copy()

    def faces(self) -> np.ndarray:
        return self._faces.copy()

    def transform(self, fitted: FittedShape) -> np.ndarray:
        return apply_fitted_to_rest_vertices(self._vertices, fitted)
