"""Tests for pendulum_simulator.constraint_solver (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.pendulum_simulator.constraint_solver import (
    constrained_accelerations,
    constraint_violation,
    equations_of_motion,
)
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams


def _make_params() -> GolferParams:
    return GolferParams(
        m_hub=0.01,
        m_r_upper=2.0,
        m_r_fore=1.5,
        m_l_upper=2.0,
        m_l_fore=1.5,
        m_club=0.3,
        L_hub=0.05,
        L_r_upper=0.35,
        L_r_fore=0.30,
        L_l_upper=0.35,
        L_l_fore=0.30,
        L_club=1.0,
        d_rs=0.2,
        d_ls=0.2,
        grip_right=0.2,
        grip_left=0.25,
    )


def _zero_torque(t) -> tuple[float, ...]:  # noqa: ARG001
    return (0.0,) * 7


class TestConstrainedAccelerations:
    def test_constraint_solver_returns_ndarray(self) -> None:
        params = _make_params()
        state = np.zeros(2 * N_DOF)
        acc = constrained_accelerations(state, 0.0, params, _zero_torque)
        assert isinstance(acc, np.ndarray)

    def test_constraint_solver_output_shape(self) -> None:
        params = _make_params()
        state = np.zeros(2 * N_DOF)
        acc = constrained_accelerations(state, 0.0, params, _zero_torque)
        assert acc.shape == (N_DOF,)


class TestConstraintViolation:
    def test_zero_state_returns_float(self) -> None:
        params = _make_params()
        state = np.zeros(2 * N_DOF)
        cv = constraint_violation(state, params)
        assert isinstance(cv, float)

    def test_zero_state_small_violation(self) -> None:
        params = _make_params()
        state = np.zeros(2 * N_DOF)
        cv = constraint_violation(state, params)
        assert cv >= 0.0


class TestEquationsOfMotion:
    def test_returns_state_derivative(self) -> None:
        params = _make_params()
        state = np.zeros(2 * N_DOF)
        dstate = equations_of_motion(state, 0.0, params, _zero_torque)
        assert isinstance(dstate, np.ndarray)
        assert dstate.shape == (2 * N_DOF,)
