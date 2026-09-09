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
        with pytest.raises(ValueError):
            result._validate_trajectory(6)

    def test_validate_non_finite_states_raises(self) -> None:
        result = _ConcreteResult(n=5, state_width=4)
        result.states[2, 1] = np.nan
        with pytest.raises(ValueError):
            result._validate_trajectory(4)

    def test_validate_strictly_increasing_time(self) -> None:
        result = _ConcreteResult(n=5, state_width=4)
        # t should be strictly increasing by default
        result._validate_trajectory(4)  # should not raise

    def test_validate_non_monotone_time_raises(self) -> None:
        result = _ConcreteResult(n=5, state_width=4)
        result.t = np.array([0.0, 0.5, 0.3, 0.8, 1.0])  # not increasing
        with pytest.raises(ValueError):
            result._validate_trajectory(4)

    def test_check_idx_valid(self) -> None:
        result = _ConcreteResult(n=5)
        result._check_idx(0)  # should not raise
        result._check_idx(4)  # last valid index

    def test_check_idx_negative_raises(self) -> None:
        result = _ConcreteResult(n=5)
        with pytest.raises(ValueError):
            result._check_idx(-1)

    def test_check_idx_out_of_range_raises(self) -> None:
        result = _ConcreteResult(n=5)
        with pytest.raises(ValueError):
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
        with pytest.raises(ValueError):
            TrajectoryResultMixin._assert_energy_finite(
                {"kinetic": float("nan"), "potential": 2.0}, idx=0
            )


@pytest.mark.unit
class TestBatchAccessorCaching:
    """Batch accessors must be single-pass and cached (#8928).

    ``all_energies`` used to iterate the trajectory once per energy key,
    and every ``all_*`` accessor recomputed its per-index series from
    scratch on each call, so plotting two joints re-integrated the
    trajectory twice.
    """

    def test_all_energies_is_single_pass(self) -> None:
        result = _ConcreteResult(n=5)
        calls = {"count": 0}
        real_energy_at = result.energy_at

        def counting_energy_at(idx: int) -> dict[str, float]:
            calls["count"] += 1
            return real_energy_at(idx)

        result.energy_at = counting_energy_at  # type: ignore[method-assign]

        result.all_energies()

        assert calls["count"] == 5, (
            "all_energies must evaluate energy_at once per index "
            f"(single pass), got {calls['count']} calls for 5 steps"
        )

    def test_all_energies_is_cached(self) -> None:
        result = _ConcreteResult(n=5)
        calls = {"count": 0}
        real_energy_at = result.energy_at

        def counting_energy_at(idx: int) -> dict[str, float]:
            calls["count"] += 1
            return real_energy_at(idx)

        result.energy_at = counting_energy_at  # type: ignore[method-assign]

        first = result.all_energies()
        second = result.all_energies()

        assert calls["count"] == 5
        for key, values in second.items():
            np.testing.assert_array_equal(first[key], values)

    def test_all_accelerations_is_cached(self) -> None:
        result = _ConcreteResult(n=5)
        calls = {"count": 0}
        real_acc_at = result.accelerations_at

        def counting_acc_at(idx: int) -> np.ndarray:
            calls["count"] += 1
            return real_acc_at(idx)

        result.accelerations_at = counting_acc_at  # type: ignore[method-assign]

        result.all_accelerations()
        result.all_accelerations()

        assert calls["count"] == 5, (
            "a second all_accelerations() call must reuse the cached "
            f"series, got {calls['count']} accelerations_at calls"
        )

    def test_all_torques_is_cached(self) -> None:
        result = _ConcreteResult(n=5)
        calls = {"count": 0}
        real_torques_at = result.torques_at

        def counting_torques_at(idx: int) -> np.ndarray:
            calls["count"] += 1
            return real_torques_at(idx)

        result.torques_at = counting_torques_at  # type: ignore[method-assign]

        result.all_torques()
        result.all_torques()

        assert calls["count"] == 5

    def test_cached_series_matches_uncached_values(self) -> None:
        result = _ConcreteResult(n=5)
        energies = result.all_energies()
        cached = result.all_energies()
        for key in energies:
            np.testing.assert_array_equal(energies[key], cached[key])


@pytest.mark.unit
class TestExtractSeriesMemoization:
    """``extract_series`` must memoize on the result object (#8928)."""

    def test_repeated_extract_does_not_recompute(self) -> None:
        from src.shared.python.pendulum_simulator.data_extractor import (
            extract_series,
        )

        result = _ConcreteResult(n=5)
        calls = {"count": 0}
        real_acc_at = result.accelerations_at

        def counting_acc_at(idx: int) -> np.ndarray:
            calls["count"] += 1
            return real_acc_at(idx)

        result.accelerations_at = counting_acc_at  # type: ignore[method-assign]

        first, _, _ = extract_series(result, "accel_shoulder")
        second, _, _ = extract_series(result, "accel_shoulder")

        assert calls["count"] == 5, (
            "a repeated extract_series() for the same series must reuse "
            f"the memoized array, got {calls['count']} calls"
        )
        np.testing.assert_array_equal(first, second)
