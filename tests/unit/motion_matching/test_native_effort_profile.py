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


def test_bernstein_derivatives_preserve_force_rotation_and_column_order() -> None:
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    profile = NativeEffortProfile(("x", "y", "z", "angle"), np.zeros((4, 7)), rotation)
    actual = profile.bernstein_control_jacobian(
        0.4, basis_duration_s=0.8, first_control=4
    )
    expected = np.zeros((4, 12))
    basis = np.array([15.0, 6.0, 1.0]) / 64
    expected[0, 3:6] = -basis
    expected[1, :3] = basis
    expected[2, 6:9] = basis
    expected[3, 9:] = basis
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert not actual.flags.writeable


@pytest.mark.parametrize(
    "duration,first", [(0, 4), (np.nan, 4), (1, -1), (1, 7), (1, True)]
)
def test_invalid_bernstein_derivative_basis_rejected(
    duration: float, first: int
) -> None:
    profile = NativeEffortProfile(("x", "y", "z"), np.zeros((3, 7)), np.eye(3))
    with pytest.raises(ValueError):
        profile.bernstein_control_jacobian(
            0.5, basis_duration_s=duration, first_control=first
        )
