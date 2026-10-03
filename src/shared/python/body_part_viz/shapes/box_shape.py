"""Axis-aligned reusable box visual geometry in its local frame."""

from itertools import product
from numbers import Real
import numpy as np
from .._types import FittedShape
from ._transform import apply_fitted_to_rest_vertices


class BoxShape:
    """An authored box with positive finite half sizes, never inferred dimensions."""

    def __init__(
        self, half_sizes: tuple[float, float, float], *, shape_id: str = "box"
    ) -> None:
        if (
            not isinstance(half_sizes, tuple)
            or len(half_sizes) != 3
            or any(
                isinstance(v, bool)
                or not isinstance(v, Real)
                or not np.isfinite(v)
                or v <= 0
                for v in half_sizes
            )
        ):
            raise ValueError("Box requires three positive real finite half sizes")
        if not isinstance(shape_id, str) or not shape_id:
            raise ValueError("Box identity must be nonempty")
        self.shape_id = shape_id
        self.rest_dimensions = tuple(float(v) * 2 for v in half_sizes)
        self._vertices = np.array(list(product((-1, 1), repeat=3)), float) * half_sizes
        self._faces = np.array(
            [
                (0, 1, 3),
                (0, 3, 2),
                (4, 6, 7),
                (4, 7, 5),
                (0, 4, 5),
                (0, 5, 1),
                (2, 3, 7),
                (2, 7, 6),
                (0, 2, 6),
                (0, 6, 4),
                (1, 5, 7),
                (1, 7, 3),
            ],
            int,
        )

    def vertices_at_rest(self) -> np.ndarray:
        """Return detached local vertices."""
        return self._vertices.copy()

    def faces(self) -> np.ndarray:
        """Return detached triangle indices."""
        return self._faces.copy()

    def transform(self, fitted: FittedShape) -> np.ndarray:
        """Reuse the canonical shape transform."""
        return apply_fitted_to_rest_vertices(self._vertices, fitted)
