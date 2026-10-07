"""Engine-free tests of the convention-free closure projection (#11606)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import project_to_closure

pytestmark = pytest.mark.unit


class _CirclePlant:
    """Point on the unit circle in the plane: one pose and one rate residual."""

    def closure_pose_residual(self, q: np.ndarray) -> np.ndarray:
        return np.array([q @ q - 1.0])

    def closure_rate_matrix(self, q: np.ndarray) -> np.ndarray:
        return 2.0 * q[None, :]


def test_projection_is_the_closest_point_and_velocity_is_tangent() -> None:
    q = np.array([1.2, 0.9])
    v = np.array([0.3, -0.7])
    result = project_to_closure(_CirclePlant(), q, v)
    np.testing.assert_allclose(result.q, q / np.linalg.norm(q), atol=1e-14)
    assert abs(result.q @ result.v) < 1e-14
    assert result.pose_residual_after <= 1e-13
    assert result.rate_residual_after <= 1e-13
    assert result.pose_residual_before == pytest.approx(q @ q - 1.0)


def test_projection_leaves_a_consistent_state_unchanged() -> None:
    q = np.array([0.6, 0.8])
    v = np.array([-0.8, 0.6])
    result = project_to_closure(_CirclePlant(), q, v)
    np.testing.assert_allclose(result.q, q, atol=1e-15)
    np.testing.assert_allclose(result.v, v, atol=1e-15)
    assert result.iterations == 0


@pytest.mark.parametrize(
    ("q", "v", "kwargs"),
    [
        (np.ones(2), np.ones(3), {}),
        (np.array([np.nan, 1.0]), np.ones(2), {}),
        (np.ones(2), np.ones(2), {"tolerance": 0.0}),
        (np.ones(2), np.ones(2), {"max_iterations": 0}),
    ],
)
def test_projection_rejects_invalid_inputs(q, v, kwargs) -> None:
    with pytest.raises(ValueError):
        project_to_closure(_CirclePlant(), q, v, **kwargs)


def test_projection_reports_non_convergence() -> None:
    class _Unreachable(_CirclePlant):
        def closure_pose_residual(self, q: np.ndarray) -> np.ndarray:
            return np.array([q @ q + 1.0])  # no real solution

    with pytest.raises(ArithmeticError):
        project_to_closure(_Unreachable(), np.ones(2), np.ones(2), max_iterations=5)
