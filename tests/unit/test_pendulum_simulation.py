"""Tests for pendulum_simulator.simulation (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.physics import PendulumParams
from src.shared.python.pendulum_simulator.simulation import (
    SimulationResult,
    run_simulation,
)


def _make_params(**kwargs) -> PendulumParams:
    defaults = {"m1": 5.0, "m2": 0.3, "L1": 0.65, "L2": 1.1, "mClub": 0.2}
    defaults.update(kwargs)
    return PendulumParams(**defaults)


def _zero_torque(t) -> tuple[float, float]:
    return 0.0, 0.0


class TestPendulumParams:
    def test_pendulum_simulation_construction(self) -> None:
        p = _make_params()
        assert p.m1 == pytest.approx(5.0)

    def test_pendulum_simulation_negative_mass_raises(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            _make_params(m1=-1.0)

    def test_zero_length_raises(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            _make_params(L1=0.0)

    def test_gravity_default(self) -> None:
        p = _make_params()
        assert p.g > 0.0


class TestRunSimulation:
    def test_returns_simulation_result(self) -> None:
        params = _make_params()
        y0 = np.array([0.1, 0.0, 0.0, 0.0])
        result = run_simulation(params, y0, 0.5, _zero_torque, dt=0.01)
        assert isinstance(result, SimulationResult)

    def test_trajectory_has_two_or_more_points(self) -> None:
        params = _make_params()
        y0 = np.array([0.0, 0.0, 0.0, 0.0])
        result = run_simulation(params, y0, 0.2, _zero_torque, dt=0.01)
        assert len(result.t) >= 2

    def test_states_finite(self) -> None:
        params = _make_params()
        y0 = np.array([0.1, 0.0, 0.0, 0.0])
        result = run_simulation(params, y0, 0.5, _zero_torque, dt=0.01)
        assert np.all(np.isfinite(result.states))

    def test_initial_state_near_zero_angle_stays_bounded(self) -> None:
        params = _make_params()
        y0 = np.array([0.01, 0.0, 0.0, 0.0])
        result = run_simulation(params, y0, 0.5, _zero_torque, dt=0.01)
        # Small perturbation, gravity-driven → stays reasonably small
        assert np.all(np.abs(result.states[:, :2]) < np.pi * 2)

    def test_pendulum_simulation_invalid_t_end_raises(self) -> None:
        params = _make_params()
        y0 = np.array([0.0, 0.0, 0.0, 0.0])
        with pytest.raises((AssertionError, ValueError)):
            run_simulation(params, y0, -1.0, _zero_torque, dt=0.01)
