"""Tests for pendulum_simulator.simulation_core (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.simulation_core import integrate_ode


def _linear_decay(t, y) -> np.ndarray:
    """dy/dt = -y → y = y0 * exp(-t)."""
    return -y


def _harmonic(t, y) -> np.ndarray:
    """Simple harmonic oscillator: [x, v] → [v, -x]."""
    return np.array([y[1], -y[0]])


class TestIntegrateOde:
    def test_simulation_core_returns_tuple(self) -> None:
        t, states = integrate_ode(_linear_decay, np.array([1.0]), 1.0, dt=0.01)
        assert isinstance(t, np.ndarray)
        assert isinstance(states, np.ndarray)

    def test_shape_consistent(self) -> None:
        t, states = integrate_ode(_linear_decay, np.array([1.0]), 1.0, dt=0.01)
        assert states.shape[0] == len(t)
        assert states.shape[1] == 1

    def test_at_least_two_time_points(self) -> None:
        t, states = integrate_ode(_linear_decay, np.array([1.0]), 1.0, dt=0.1)
        assert len(t) >= 2

    def test_initial_condition_respected(self) -> None:
        y0 = np.array([5.0])
        t, states = integrate_ode(_linear_decay, y0, 1.0, dt=0.01)
        assert states[0, 0] == pytest.approx(5.0, rel=1e-3)

    def test_exponential_decay(self) -> None:
        y0 = np.array([1.0])
        t, states = integrate_ode(_linear_decay, y0, 2.0, dt=0.01)
        expected = np.exp(-t)
        np.testing.assert_allclose(states[:, 0], expected, rtol=1e-3)

    def test_two_variable_harmonic(self) -> None:
        y0 = np.array([1.0, 0.0])
        t, states = integrate_ode(_harmonic, y0, 2 * np.pi, dt=0.01)
        assert states.shape[1] == 2
        # x(t) ≈ cos(t)
        np.testing.assert_allclose(states[:, 0], np.cos(t), atol=0.02)

    def test_simulation_core_invalid_t_end_raises(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            integrate_ode(_linear_decay, np.array([1.0]), -1.0, dt=0.01)

    def test_non_finite_initial_state_raises(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            integrate_ode(_linear_decay, np.array([np.inf]), 1.0, dt=0.01)

    def test_dt_larger_than_t_end_raises(self) -> None:
        with pytest.raises((AssertionError, ValueError)):
            integrate_ode(_linear_decay, np.array([1.0]), 0.5, dt=1.0)

    def test_all_states_finite(self) -> None:
        y0 = np.array([1.0])
        t, states = integrate_ode(_linear_decay, y0, 1.0, dt=0.01)
        assert np.all(np.isfinite(states))
