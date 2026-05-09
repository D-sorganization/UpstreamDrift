"""Tests for src.shared.python.pendulum_simulator.simulation_result_base (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.simulation_result_base import (
    TrajectoryResultMixin,
)


class _ConcreteResult(TrajectoryResultMixin):
    """Minimal concrete implementation of TrajectoryResultMixin for testing."""

    def __init__(self, n: int = 10, state_width: int = 4) -> None:
        self.t = np.linspace(0.0, 1.0, n)
        self.states = np.zeros((n, state_width))
        self._state_width = state_width

    def positions_at(self, idx: int) -> dict[str, np.ndarray]:
        return {"joint": np.zeros(2)}

    def energy_at(self, idx: int) -> dict[str, float]:
        return {"kinetic": 1.0, "potential": 2.0, "total": 3.0}

    def accelerations_at(self, idx: int) -> np.ndarray:
        return np.zeros(self._state_width // 2)

    def torques_at(self, idx: int) -> np.ndarray:
        return np.ones(self._state_width // 2)

    def friction_torques_at(self, idx: int) -> np.ndarray:
        return np.zeros(self._state_width // 2)

    def mass_matrix_at(self, idx: int) -> np.ndarray:
        n = self._state_width // 2
        return np.eye(n)


class TestTrajectoryResultMixin:
    def test_simulation_result_base_n_steps(self) -> None:
        result = _ConcreteResult(n=10)
        assert result.n_steps == 10

    def test_n_steps_single(self) -> None:
        result = _ConcreteResult(n=1)
        assert result.n_steps == 1

    def test_validate_trajectory_valid(self) -> None:
        result = _ConcreteResult(n=10, state_width=4)
        result._validate_trajectory(4)  # should not raise

    def test_validate_wrong_state_width_raises(self) -> None:
        result = _ConcreteResult(n=10, state_width=4)
        with pytest.raises(AssertionError):
            result._validate_trajectory(6)

    def test_validate_non_finite_states_raises(self) -> None:
        result = _ConcreteResult(n=5, state_width=4)
        result.states[2, 1] = np.nan
        with pytest.raises(AssertionError):
            result._validate_trajectory(4)

    def test_validate_strictly_increasing_time(self) -> None:
        result = _ConcreteResult(n=5, state_width=4)
        # t should be strictly increasing by default
        result._validate_trajectory(4)  # should not raise

    def test_validate_non_monotone_time_raises(self) -> None:
        result = _ConcreteResult(n=5, state_width=4)
        result.t = np.array([0.0, 0.5, 0.3, 0.8, 1.0])  # not increasing
        with pytest.raises(AssertionError):
            result._validate_trajectory(4)

    def test_check_idx_valid(self) -> None:
        result = _ConcreteResult(n=5)
        result._check_idx(0)  # should not raise
        result._check_idx(4)  # last valid index

    def test_check_idx_negative_raises(self) -> None:
        result = _ConcreteResult(n=5)
        with pytest.raises(AssertionError):
            result._check_idx(-1)

    def test_check_idx_out_of_range_raises(self) -> None:
        result = _ConcreteResult(n=5)
        with pytest.raises(AssertionError):
            result._check_idx(5)

    def test_all_positions_length(self) -> None:
        result = _ConcreteResult(n=10)
        positions = result.all_positions()
        assert len(positions) == 10

    def test_all_energies_keys(self) -> None:
        result = _ConcreteResult(n=5)
        energies = result.all_energies()
        assert "kinetic" in energies
        assert "potential" in energies
        assert "total" in energies

    def test_all_energies_shape(self) -> None:
        result = _ConcreteResult(n=5)
        energies = result.all_energies()
        for v in energies.values():
            assert v.shape == (5,)

    def test_all_accelerations_shape(self) -> None:
        result = _ConcreteResult(n=5, state_width=4)
        acc = result.all_accelerations()
        assert acc.shape == (5, 2)

    def test_all_torques_shape(self) -> None:
        result = _ConcreteResult(n=5, state_width=4)
        torques = result.all_torques()
        assert torques.shape == (5, 2)

    def test_all_friction_torques_shape(self) -> None:
        result = _ConcreteResult(n=5, state_width=4)
        fric = result.all_friction_torques()
        assert fric.shape == (5, 2)

    def test_assert_energy_finite_valid(self) -> None:
        TrajectoryResultMixin._assert_energy_finite(
            {"kinetic": 1.0, "potential": 2.0}, idx=0
        )  # should not raise

    def test_assert_energy_finite_nan_raises(self) -> None:
        with pytest.raises(AssertionError):
            TrajectoryResultMixin._assert_energy_finite(
                {"kinetic": float("nan"), "potential": 2.0}, idx=0
            )
