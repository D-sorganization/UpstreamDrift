"""Smoke tests for research MPC controller module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from src.research.mpc.controller import (
    Constraint,
    CostFunction,
    ModelPredictiveController,
    MPCResult,
)


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
