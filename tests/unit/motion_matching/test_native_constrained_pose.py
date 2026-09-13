"""Native adapter contracts for shared constrained marker-pose fitting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import pytest

from src.engines.physics_engines.pinocchio.python.native_constrained_pose import (
    NativeConstrainedPoseOracle,
)


class _Markers(NamedTuple):
    positions_m: NDArray[np.float64]


class _Model:
    def __init__(self) -> None:
        self.coordinates: dict[str, float] | None = None

    def marker_derivatives(
        self,
        coordinates: dict[str, float],
        bodies: Sequence[str],
        offsets: NDArray[np.float64],
    ) -> _Markers:
        self.coordinates = coordinates
        assert bodies == ("pelvis", "club")
        np.testing.assert_array_equal(offsets, [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
        return _Markers(
            np.array([[coordinates["x"], 0.0, 1.0], [0.0, coordinates["y"], 2.0]])
        )

    def closure_residuals(
        self, coordinates: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.array([coordinates["x"] + coordinates["y"]]), np.zeros(1)


@pytest.mark.unit
def test_native_oracle_maps_ordered_coordinates_to_shared_solver_contract() -> None:
    """Marker and closure calls share one explicit native coordinate mapping."""
    model = _Model()
    oracle = NativeConstrainedPoseOracle(
        model,
        ("x", "y"),
        ("pelvis", "club"),
        np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]]),
    )

    predicted = oracle.forward(np.array([2.0, -3.0]))
    closure = oracle.closure(np.array([2.0, -3.0]))

    np.testing.assert_array_equal(predicted, [[2.0, 0.0, 1.0], [0.0, -3.0, 2.0]])
    np.testing.assert_array_equal(closure, [-1.0])
    assert model.coordinates == {"x": 2.0, "y": -3.0}


@pytest.mark.unit
@pytest.mark.parametrize(
    "coordinates",
    [np.array([1.0]), np.array([1.0, np.nan]), np.ones((2, 1))],
)
def test_native_oracle_rejects_invalid_coordinate_vectors(
    coordinates: np.ndarray,
) -> None:
    """A pose solver cannot silently alter native state dimension or values."""
    oracle = NativeConstrainedPoseOracle(
        _Model(),
        ("x", "y"),
        ("pelvis", "club"),
        np.zeros((2, 3)),
    )

    with pytest.raises(ValueError, match="native coordinate vector"):
        oracle.forward(coordinates)


@pytest.mark.unit
def test_native_oracle_rejects_invalid_native_marker_output() -> None:
    """A malformed engine response fails before it reaches the solver."""

    class _BadModel(_Model):
        def marker_derivatives(
            self,
            coordinates: dict[str, float],
            bodies: Sequence[str],
            offsets: NDArray[np.float64],
        ) -> _Markers:
            return _Markers(np.ones((1, 3)))

    oracle = NativeConstrainedPoseOracle(
        _BadModel(),
        ("x", "y"),
        ("pelvis", "club"),
        np.zeros((2, 3)),
    )

    with pytest.raises(ValueError, match="marker positions"):
        oracle.forward(np.zeros(2))
