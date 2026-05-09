"""Tests for pendulum_simulator.simulation_golfer (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.physics_golfer import N_DOF, GolferParams
from src.shared.python.pendulum_simulator.simulation_golfer import (
    GolferSimulationResult,
    run_simulation,
)


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
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class TestGolferParams:
    def test_simulation_golfer_construction(self) -> None:
        p = _make_params()
        assert p.m_r_upper == pytest.approx(2.0)

    def test_n_dof(self) -> None:
        assert N_DOF == 8

    def test_g_positive(self) -> None:
        p = _make_params()
        assert p.g > 0.0


class TestRunSimulationGolfer:
    def test_returns_golfer_simulation_result(self) -> None:
        params = _make_params()
        y0 = np.zeros(2 * N_DOF)
        result = run_simulation(params, y0, 0.05, _zero_torque, dt=0.02)
        assert isinstance(result, GolferSimulationResult)

    def test_trajectory_has_multiple_points(self) -> None:
        params = _make_params()
        y0 = np.zeros(2 * N_DOF)
        result = run_simulation(params, y0, 0.05, _zero_torque, dt=0.02)
        assert len(result.t) >= 2

    def test_states_correct_shape(self) -> None:
        params = _make_params()
        y0 = np.zeros(2 * N_DOF)
        result = run_simulation(params, y0, 0.05, _zero_torque, dt=0.02)
        assert result.states.shape[1] == 2 * N_DOF

    def test_simulation_golfer_invalid_t_end_raises(self) -> None:
        params = _make_params()
        y0 = np.zeros(2 * N_DOF)
        with pytest.raises((AssertionError, ValueError)):
            run_simulation(params, y0, -1.0, _zero_torque, dt=0.02)
