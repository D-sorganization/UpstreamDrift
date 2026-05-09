"""Tests for pendulum_simulator.simulation_triple (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.physics_triple import TriplePendulumParams
from src.shared.python.pendulum_simulator.simulation_triple import (
    TripleSimulationResult,
    run_simulation,
)


def _make_params(**kwargs) -> TriplePendulumParams:
    defaults = {"m1": 5.0, "m2": 0.3, "m3": 0.05, "L1": 0.65, "L2": 1.1, "L3": 0.1}
    defaults.update(kwargs)
    return TriplePendulumParams(**defaults)


def _zero_torque(t) -> tuple[float, float, float]:
    return 0.0, 0.0, 0.0


class TestTriplePendulumParams:
    def test_simulation_triple_construction(self) -> None:
        p = _make_params()
        assert p.m1 == pytest.approx(5.0)

    def test_simulation_triple_negative_mass_raises(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            _make_params(m1=-1.0)


class TestRunSimulationTriple:
    def test_returns_simulation_result(self) -> None:
        params = _make_params()
        y0 = np.zeros(6)
        result = run_simulation(params, y0, 0.2, _zero_torque, dt=0.01)
        assert isinstance(result, TripleSimulationResult)

    def test_trajectory_length(self) -> None:
        params = _make_params()
        y0 = np.zeros(6)
        result = run_simulation(params, y0, 0.2, _zero_torque, dt=0.01)
        assert len(result.t) >= 2

    def test_states_finite(self) -> None:
        params = _make_params()
        y0 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = run_simulation(params, y0, 0.2, _zero_torque, dt=0.01)
        assert np.all(np.isfinite(result.states))

    def test_simulation_triple_invalid_t_end_raises(self) -> None:
        params = _make_params()
        y0 = np.zeros(6)
        with pytest.raises((AssertionError, ValueError)):
            run_simulation(params, y0, -1.0, _zero_torque, dt=0.01)

    def test_wrong_initial_state_raises(self) -> None:
        params = _make_params()
        y0 = np.zeros(4)  # Wrong shape — should be (6,)
        with pytest.raises((AssertionError, ValueError)):
            run_simulation(params, y0, 0.2, _zero_torque, dt=0.01)
