"""Line shape: two endpoints along the local x-axis."""

from __future__ import annotations

import numpy as np

from .._types import FittedShape
from ._transform import apply_fitted_to_rest_vertices

__all__ = ["LineShape"]


class LineShape:
    """A 1D line segment of length ``length`` along the local x-axis.

    ``vertices_at_rest()`` returns the two endpoints; ``faces()`` returns
    an empty ``(0, 3)`` array — line shapes are rendered as edges, not
    triangles.
    """

    shape_id: str
    rest_dimensions: tuple[float, ...]

    def __init__(self, length: float, *, shape_id: str = "line") -> None:
        if not isinstance(length, (int, float)) or isinstance(length, bool):
            raise TypeError(f"length must be numeric; got {type(length).__name__}")
        if not np.isfinite(float(length)) or float(length) <= 0.0:
            raise ValueError(f"length must be finite and > 0; got {length}")
        if not isinstance(shape_id, str) or not shape_id:
            raise ValueError(f"shape_id must be non-empty str; got {shape_id!r}")
        self.shape_id = shape_id
        self.rest_dimensions = (float(length),)

    def vertices_at_rest(self) -> np.ndarray:
        length = self.rest_dimensions[0]
        return np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=np.float64)

    def faces(self) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.int64)

    def transform(self, fitted: FittedShape) -> np.ndarray:
        return apply_fitted_to_rest_vertices(self.vertices_at_rest(), fitted)
