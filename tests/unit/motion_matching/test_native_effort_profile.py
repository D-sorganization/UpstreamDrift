"""Native effort conversion preserves polynomial order and virtual-work frame."""

import numpy as np
import pytest

from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile

pytestmark = pytest.mark.unit


def test_native_order_absolute_seconds_and_force_rotation() -> None:
    coefficients = np.arange(28, dtype=float).reshape(4, 7)
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    profile = NativeEffortProfile(("x", "y", "z", "angle"), coefficients, rotation)
    expected = np.array([np.polyval(row, 0.7) for row in coefficients])
    expected[:3] = rotation @ expected[:3]
    np.testing.assert_allclose(list(profile.evaluate(0.7).values()), expected)
    coefficients[:] = 0
    rotation[:] = 0
    np.testing.assert_allclose(list(profile.evaluate(0.7).values()), expected)


@pytest.mark.parametrize("coefficients", [np.zeros((4, 6)), np.full((4, 7), np.nan)])
def test_invalid_coefficients_rejected(coefficients: np.ndarray) -> None:
    with pytest.raises(ValueError, match="coefficients"):
        NativeEffortProfile(("x", "y", "z", "a"), coefficients, np.eye(3))


def test_duplicate_coordinates_rejected() -> None:
    with pytest.raises(ValueError, match="coordinate"):
        NativeEffortProfile(("x", "x", "z"), np.zeros((3, 7)), np.eye(3))


def test_reflection_is_not_a_force_frame_rotation() -> None:
    with pytest.raises(ValueError, match="rotation"):
        NativeEffortProfile(("x", "y", "z"), np.zeros((3, 7)), -np.eye(3))
