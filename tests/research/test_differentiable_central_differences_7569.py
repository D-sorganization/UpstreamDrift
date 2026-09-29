"""Precision and contract tests for issue #7569."""

from __future__ import annotations

import numpy as np
import pytest

from src.research.differentiable.engine import DifferentiableEngine


class _QuadraticControlEngine:
    """One-DOF deterministic engine with analytic finite-difference oracles."""

    n_q = 1
    n_v = 1

    def __init__(self) -> None:
        self._q = np.zeros(1)
        self._v = np.zeros(1)
        self._tau = np.zeros(1)

    def set_joint_positions(self, q: np.ndarray) -> None:
        self._q = np.array(q, dtype=float)

    def set_joint_velocities(self, v: np.ndarray) -> None:
        self._v = np.array(v, dtype=float)

    def set_joint_torques(self, tau: np.ndarray) -> None:
        self._tau = np.array(tau, dtype=float)

    def step(self, dt: float) -> None:
        self._v = self._v + self._tau**2 * dt
        self._q = self._q + self._v * dt

    def get_joint_positions(self) -> np.ndarray:
        return self._q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self._v.copy()


def _final_position_loss(trajectory: np.ndarray) -> float:
    return float(trajectory[-1, 0])


@pytest.mark.unit
def test_control_gradient_uses_scaled_central_differences() -> None:
    engine = DifferentiableEngine(_QuadraticControlEngine(), backend="numpy")
    controls = np.array([[2.0]])
    dt = 0.75

    gradient = engine.compute_gradient(
        np.array([0.0, 0.0]),
        controls,
        _final_position_loss,
        dt=dt,
    )

    analytic = np.array([[2.0 * controls[0, 0] * dt**2]])
    np.testing.assert_allclose(gradient, analytic, rtol=0.0, atol=1e-10)


@pytest.mark.unit
def test_jacobian_uses_scaled_central_differences() -> None:
    engine = DifferentiableEngine(_QuadraticControlEngine(), backend="numpy")
    state = np.array([1.25, -0.5])
    control = np.array([2.0])
    dt = 0.75

    A, B = engine.compute_jacobian(state, control, dt=dt)

    expected_A = np.array([[1.0, dt], [0.0, 1.0]])
    expected_B = np.array([[2.0 * control[0] * dt**2], [2.0 * control[0] * dt]])
    np.testing.assert_allclose(A, expected_A, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(B, expected_B, rtol=0.0, atol=1e-9)


@pytest.mark.unit
def test_gradient_rejects_non_finite_controls() -> None:
    engine = DifferentiableEngine(_QuadraticControlEngine(), backend="numpy")

    with pytest.raises(ValueError, match="controls.*finite"):
        engine.compute_gradient(
            np.zeros(2),
            np.array([[np.nan]]),
            _final_position_loss,
        )


@pytest.mark.unit
def test_jacobian_rejects_bad_control_shape() -> None:
    engine = DifferentiableEngine(_QuadraticControlEngine(), backend="numpy")

    with pytest.raises(ValueError, match="control.*shape"):
        engine.compute_jacobian(np.zeros(2), np.zeros(2))
