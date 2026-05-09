"""Tests for src.shared.python.pendulum_simulator.data_extractor (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.pendulum_simulator.data_extractor import (
    extract_series,
    list_available_series,
)


class _MockDoubleResult:
    """Minimal mock of a double-pendulum simulation result."""

    def __init__(self, n: int = 20) -> None:
        self.t = np.linspace(0.0, 1.0, n)
        self.states = np.random.default_rng(42).normal(size=(n, 4))
        self._torques = np.ones((n, 2)) * 5.0
        self.n_steps = n

    def torques_at(self, i: int) -> list[float]:
        return list(self._torques[i])

    def all_total_torques(self) -> np.ndarray:
        return self._torques

    def all_energies(self) -> dict[str, np.ndarray]:
        return {
            "kinetic": np.ones(self.n_steps),
            "potential": np.ones(self.n_steps) * 2.0,
            "total": np.ones(self.n_steps) * 3.0,
        }

    def joint_velocities_at(self, i: int) -> dict[str, float]:
        return {"tip_speed": float(self.states[i, 2])}

    def all_accelerations(self) -> np.ndarray:
        return np.zeros((self.n_steps, 2))

    def coriolis_at(self, i: int) -> list[float]:
        return [0.0, 0.0]

    def gravity_at(self, i: int) -> list[float]:
        return [1.0, 0.5]

    def all_friction_torques(self) -> np.ndarray:
        return np.zeros((self.n_steps, 2))

    def base_force_at(self, i: int) -> dict[str, float]:
        return {"fx": 1.0, "fy": 0.5, "magnitude": 1.118}


class TestListAvailableSeries:
    def test_data_extractor_returns_list(self) -> None:
        result = list_available_series()
        assert isinstance(result, list)

    def test_data_extractor_non_empty(self) -> None:
        result = list_available_series()
        assert len(result) > 0

    def test_each_entry_is_3_tuple(self) -> None:
        result = list_available_series()
        for entry in result:
            assert len(entry) == 3

    def test_time_key_present(self) -> None:
        keys = [entry[0] for entry in list_available_series()]
        assert "time" in keys

    def test_theta1_key_present(self) -> None:
        keys = [entry[0] for entry in list_available_series()]
        assert "theta1" in keys

    def test_torque_shoulder_key_present(self) -> None:
        keys = [entry[0] for entry in list_available_series()]
        assert "torque_shoulder" in keys

    def test_descriptions_are_strings(self) -> None:
        for _, desc, unit in list_available_series():
            assert isinstance(desc, str)
            assert isinstance(unit, str)


class TestExtractSeries:
    def setup_method(self) -> None:
        self._result = _MockDoubleResult(n=20)

    def test_extract_time(self) -> None:
        values, desc, unit = extract_series(self._result, "time")
        assert isinstance(values, np.ndarray)
        assert values.ndim == 1
        assert len(values) == 20

    def test_extract_theta1(self) -> None:
        values, desc, unit = extract_series(self._result, "theta1")
        assert values.shape == (20,)

    def test_extract_torque_shoulder(self) -> None:
        values, desc, unit = extract_series(self._result, "torque_shoulder")
        assert values.shape == (20,)
        # All torques should be 5.0 from mock
        np.testing.assert_allclose(values, 5.0)

    def test_returns_description_string(self) -> None:
        _, desc, _ = extract_series(self._result, "time")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_returns_unit_string(self) -> None:
        _, _, unit = extract_series(self._result, "time")
        assert isinstance(unit, str)

    def test_unknown_key_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            extract_series(self._result, "nonexistent_series")

    def test_returned_values_are_1d(self) -> None:
        for key, _, _ in list_available_series():
            try:
                values, _, _ = extract_series(self._result, key)
                assert values.ndim == 1, f"Series {key!r} returned non-1D array"
            except (AttributeError, KeyError):
                # Some series may not be present in the minimal mock
                pass
