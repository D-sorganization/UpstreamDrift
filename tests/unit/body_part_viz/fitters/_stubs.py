"""Lightweight stub shapes for fitter unit tests."""

from __future__ import annotations

import numpy as np

from src.shared.python.body_part_viz import FittedShape


class StubShape:
    """Minimal :class:`BodyPartShape` implementation for tests."""

    def __init__(
        self,
        shape_id: str = "stub",
        rest_dimensions: tuple[float, ...] = (1.0,),
    ) -> None:
        self.shape_id = shape_id
        self.rest_dimensions = rest_dimensions

    def vertices_at_rest(self) -> np.ndarray:
        return np.zeros((0, 3))

    def faces(self) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.int64)

    def transform(self, fitted: FittedShape) -> np.ndarray:
        return np.zeros((0, 3))
