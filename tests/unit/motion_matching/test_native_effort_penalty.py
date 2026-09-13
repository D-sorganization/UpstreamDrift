"""Independent integral and derivative checks for native sextic effort cost."""

import numpy as np
import pytest

from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile
from src.shared.python.motion_matching.native_effort_penalty import (
    native_effort_penalty,
)

pytestmark = pytest.mark.unit


def test_penalty_integrates_squared_sextic_exactly():
    coefficients = np.zeros((4, 7))
    coefficients[3, 0] = 1  # tau(t)=t**6 over [0,2]
    profile = NativeEffortProfile(("x", "y", "z", "joint"), coefficients, np.eye(3))
    penalty = native_effort_penalty(profile, duration_s=2, effort_scales=np.ones(4))
    residual = penalty.residual(np.zeros(4))
    assert residual @ residual == pytest.approx(2**12 / 13, rel=1e-12)


def test_scaled_native_penalty_derivative_and_ownership():
    coefficients = np.zeros((4, 7))
    coefficients[:, -1] = [1, 2, 3, 4]
    profile = NativeEffortProfile(
        ("x", "y", "z", "joint"),
        coefficients,
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    )
    scales = np.array([2.0, 1.0, 3.0, 4.0])
    penalty = native_effort_penalty(
        profile, duration_s=0.85, effort_scales=scales, weight=2
    )
    scales[:] = 1000
    x = np.array([0.1, 0.2, -0.3, 0.4])
    direction = np.array([1.0, -0.2, 0.3, 2.0])
    eps = 1e-5
    numerical = (
        penalty.residual(x + eps * direction) - penalty.residual(x - eps * direction)
    ) / (2 * eps)
    np.testing.assert_allclose(numerical, penalty.jacobian(x) @ direction, atol=1e-10)
    assert not penalty.jacobian(x).flags.writeable
    base = penalty.residual(np.zeros(4))
    assert base @ base == pytest.approx(2 * (1 + 1 + 1 + 1))


@pytest.mark.parametrize(
    "scales", [np.ones(3), np.array([1, 1, 0, 1]), np.array([1, 1, np.nan, 1])]
)
def test_penalty_rejects_invalid_scales(scales):
    profile = NativeEffortProfile(("x", "y", "z", "joint"), np.zeros((4, 7)), np.eye(3))
    with pytest.raises(ValueError):
        native_effort_penalty(profile, duration_s=1, effort_scales=scales)


def test_penalty_rejects_bad_parameter_shape():
    profile = NativeEffortProfile(("x", "y", "z", "joint"), np.zeros((4, 7)), np.eye(3))
    penalty = native_effort_penalty(profile, duration_s=1, effort_scales=np.ones(4))
    with pytest.raises(ValueError):
        penalty.residual(np.ones(5))
