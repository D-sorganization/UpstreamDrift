"""Shared marker-coordinate helper tests."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.sources._marker_coordinates import (
    has_nan_coordinate,
)


@pytest.mark.parametrize(
    ("x", "y", "z"),
    [
        (float("nan"), 1.0, 2.0),
        (1.0, float("nan"), 2.0),
        (1.0, 2.0, float("nan")),
    ],
)
def test_has_nan_coordinate_detects_occluded_axis(x: float, y: float, z: float) -> None:
    assert has_nan_coordinate(x, y, z) is True


def test_has_nan_coordinate_allows_finite_triplet() -> None:
    assert has_nan_coordinate(1.0, 2.0, 3.0) is False
