"""Smoke tests for research MPC controller module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import linalg

from src.research.mpc.controller import (
    Constraint,
    CostFunction,
    ModelPredictiveController,
    MPCResult,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def mock_engine() -> MagicMock:
    """Create a minimal mock physics engine."""
    engine = MagicMock()
    engine.n_q = 3
    engine.n_v = 3
    engine.get_joint_positions.return_value = np.zeros(3)
    engine.get_joint_velocities.return_value = np.zeros(3)
    return engine


class TestCostFunction:
    """Smoke tests for CostFunction dataclass."""

    def test_running_cost_basic(self) -> None:
        cost = CostFunction(Q=np.eye(2), R=np.eye(2))
        x = np.array([1.0, 0.0])
        u = np.array([0.0, 1.0])
        result = cost.evaluate_running_cost(x, u)
        assert result == pytest.approx(2.0)

    def test_terminal_cost_with_no_P(self) -> None:
        cost = CostFunction(Q=np.eye(2), R=np.eye(2), P=None)
        assert cost.evaluate_terminal_cost(np.array([1.0, 2.0])) == 0.0

    def test_terminal_cost_with_P(self) -> None:
        cost = CostFunction(Q=np.eye(2), R=np.eye(2), P=np.eye(2))
        val = cost.evaluate_terminal_cost(np.array([1.0, 2.0]))
        assert val == pytest.approx(5.0)


class TestConstraint:
    """Smoke tests for Constraint dataclass."""

    def test_default_constraint(self) -> None:
        c = Constraint()
        assert c.constraint_type == "mixed"
        assert c.A is None

    def test_constraint_with_bounds(self) -> None:
        c = Constraint(
            lb=np.array([-1.0]),
            ub=np.array([1.0]),
        )
        assert c.lb is not None
        assert c.ub is not None


class TestModelPredictiveController:
    """Smoke tests for MPC controller."""

    def test_mpc_controller_construction(self, mock_engine: MagicMock) -> None:
        mpc = ModelPredictiveController(mock_engine, horizon=5, dt=0.01)
        assert mpc.n_states == 6
        assert mpc.n_controls == 3
        assert mpc.horizon == 5

    def test_set_cost_function(self, mock_engine: MagicMock) -> None:
        mpc = ModelPredictiveController(mock_engine, horizon=5)
        cost = CostFunction(Q=np.eye(6), R=np.eye(3))
        mpc.set_cost_function(cost)

    def test_add_and_clear_constraints(self, mock_engine: MagicMock) -> None:
        mpc = ModelPredictiveController(mock_engine, horizon=5)
        mpc.add_constraint(Constraint())
        mpc.add_constraint(Constraint())
        mpc.clear_constraints()

    def test_get_first_control_from_empty_result(self, mock_engine: MagicMock) -> None:
        mpc = ModelPredictiveController(mock_engine, horizon=5)
        result = MPCResult(
            success=False,
            optimal_states=None,
            optimal_controls=None,
            cost=0.0,
        )
        u0 = mpc.get_first_control(result)
        assert u0.shape == (3,)
        np.testing.assert_array_equal(u0, np.zeros(3))


class DeterministicLinearizedMPC(ModelPredictiveController):
    """MPC with fixed dynamics linearization for backward-pass tests."""

    def __init__(self) -> None:
        engine = MagicMock()
        engine.n_q = 1
        engine.n_v = 1
        super().__init__(engine, horizon=1, dt=0.01)

    def _dynamics_linearize(
        self,
        x: np.ndarray,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.array([[1.0, 0.1], [0.0, 1.0]]), np.array([[0.0], [1.0]])


class TestILQRBackwardPassSolve:
    """Focused tests for iLQR backward-pass gain solves."""

    @staticmethod
    def _sample_inputs() -> tuple[np.ndarray, np.ndarray, CostFunction]:
        x = np.array([[0.4, -0.2], [0.1, 0.3]])
        u = np.array([[0.25]])
        cost = CostFunction(
            Q=np.diag([2.0, 3.0]),
            R=np.array([[0.7]]),
            P=np.diag([1.5, 2.5]),
        )
        return x, u, cost

    def test_backward_pass_matches_previous_inverse_math(self) -> None:
        mpc = DeterministicLinearizedMPC()
        x, u, cost = self._sample_inputs()

        K, d = mpc._backward_pass(x, u, cost)

        A, B = mpc._dynamics_linearize(x[0], u[0])
        Vx = 2 * cost.P @ x[-1]
        Vxx = 2 * cost.P
        lu = 2 * cost.R @ u[0]
        Qu = lu + B.T @ Vx
        Quu = 2 * cost.R + B.T @ Vxx @ B
        Qux = B.T @ Vxx @ A
        Quu_reg = Quu + np.eye(mpc.n_controls) * 1e-6
        expected_inverse = np.linalg.inv(Quu_reg)

        np.testing.assert_allclose(K[0], -expected_inverse @ Qux, rtol=1e-12)
        np.testing.assert_allclose(d[0], -expected_inverse @ Qu, rtol=1e-12)

    def test_backward_pass_uses_cholesky_solve_without_explicit_inverse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.research.mpc import controller

        mpc = DeterministicLinearizedMPC()
        x, u, cost = self._sample_inputs()
        calls = {"cho_factor": 0, "cho_solve": 0}
        original_cho_factor = linalg.cho_factor
        original_cho_solve = linalg.cho_solve

        def tracking_cho_factor(
            a: np.ndarray, **kwargs: object
        ) -> tuple[np.ndarray, bool]:
            calls["cho_factor"] += 1
            return original_cho_factor(a, **kwargs)

        def tracking_cho_solve(
            c_and_lower: tuple[np.ndarray, bool],
            b: np.ndarray,
            **kwargs: object,
        ) -> np.ndarray:
            calls["cho_solve"] += 1
            return original_cho_solve(c_and_lower, b, **kwargs)

        def fail_inverse(_a: np.ndarray) -> np.ndarray:
            raise AssertionError("iLQR backward pass must not call np.linalg.inv")

        monkeypatch.setattr(controller.linalg, "cho_factor", tracking_cho_factor)
        monkeypatch.setattr(controller.linalg, "cho_solve", tracking_cho_solve)
        monkeypatch.setattr(controller.np.linalg, "inv", fail_inverse)

        K, d = mpc._backward_pass(x, u, cost)

        assert calls == {"cho_factor": 1, "cho_solve": 1}
        assert K[0].shape == (1, 2)
        assert d[0].shape == (1,)
        assert np.all(np.isfinite(K[0]))
        assert np.all(np.isfinite(d[0]))

    def test_backward_pass_falls_back_to_general_solve_when_cholesky_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.research.mpc import controller

        mpc = DeterministicLinearizedMPC()
        x, u, cost = self._sample_inputs()
        calls = {"solve": 0}
        original_solve = np.linalg.solve

        def fail_cho_factor(
            _a: np.ndarray, **_kwargs: object
        ) -> tuple[np.ndarray, bool]:
            raise np.linalg.LinAlgError("not positive definite")

        def tracking_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            calls["solve"] += 1
            return original_solve(a, b)

        monkeypatch.setattr(controller.linalg, "cho_factor", fail_cho_factor)
        monkeypatch.setattr(controller.np.linalg, "solve", tracking_solve)

        K, d = mpc._backward_pass(x, u, cost)

        assert calls == {"solve": 1}
        assert K[0].shape == (1, 2)
        assert d[0].shape == (1,)
        assert np.all(np.isfinite(K[0]))
        assert np.all(np.isfinite(d[0]))

    def test_backward_pass_rejects_nonfinite_gain_system(self) -> None:
        mpc = DeterministicLinearizedMPC()
        x, u, cost = self._sample_inputs()
        cost.R = np.array([[np.nan]])

        with pytest.raises(ValueError, match="Quu_reg must contain only finite values"):
            mpc._backward_pass(x, u, cost)
