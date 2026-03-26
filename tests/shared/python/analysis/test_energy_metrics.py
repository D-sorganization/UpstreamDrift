"""Tests for analysis.energy_metrics module.

Validates EnergyMetricsMixin.compute_energy_metrics() using a minimal
stub object. Tests cover precondition enforcement, metric correctness,
optional club_head_speed path, and postcondition guarantees.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.energy_metrics import EnergyMetricsMixin

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stub helper
# ---------------------------------------------------------------------------


class _Stub(EnergyMetricsMixin):
    """Minimal host for the mixin — optionally provides club_head_speed."""

    def __init__(self, club_head_speed: np.ndarray | None = None) -> None:
        self.club_head_speed = club_head_speed


# ---------------------------------------------------------------------------
# Tests — precondition enforcement
# ---------------------------------------------------------------------------


class TestEnergyMetricsPreconditions:
    def test_raises_on_empty_kinetic_energy(self) -> None:
        stub = _Stub()
        with pytest.raises((AssertionError, ValueError)):
            stub.compute_energy_metrics(
                kinetic_energy=np.array([]),
                potential_energy=np.array([1.0]),
            )

    def test_raises_on_empty_potential_energy(self) -> None:
        stub = _Stub()
        with pytest.raises((AssertionError, ValueError)):
            stub.compute_energy_metrics(
                kinetic_energy=np.array([1.0]),
                potential_energy=np.array([]),
            )

    def test_raises_on_length_mismatch(self) -> None:
        stub = _Stub()
        with pytest.raises((AssertionError, ValueError)):
            stub.compute_energy_metrics(
                kinetic_energy=np.array([1.0, 2.0]),
                potential_energy=np.array([1.0]),
            )

    def test_raises_on_negative_kinetic_energy(self) -> None:
        stub = _Stub()
        with pytest.raises((AssertionError, ValueError)):
            stub.compute_energy_metrics(
                kinetic_energy=np.array([-1.0, 2.0]),
                potential_energy=np.array([1.0, 1.0]),
            )

    def test_raises_on_non_finite_kinetic_energy(self) -> None:
        stub = _Stub()
        with pytest.raises((AssertionError, ValueError)):
            stub.compute_energy_metrics(
                kinetic_energy=np.array([float("nan")]),
                potential_energy=np.array([1.0]),
            )

    def test_raises_on_non_finite_potential_energy(self) -> None:
        stub = _Stub()
        with pytest.raises((AssertionError, ValueError)):
            stub.compute_energy_metrics(
                kinetic_energy=np.array([1.0]),
                potential_energy=np.array([float("inf")]),
            )


# ---------------------------------------------------------------------------
# Tests — metric correctness
# ---------------------------------------------------------------------------


class TestEnergyMetricsCorrectness:
    def _basic_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        ke = np.array([1.0, 3.0, 2.0])
        pe = np.array([5.0, 2.0, 3.0])
        return ke, pe

    def test_max_kinetic_energy(self) -> None:
        ke, pe = self._basic_arrays()
        result = _Stub().compute_energy_metrics(ke, pe)
        assert result["max_kinetic_energy"] == pytest.approx(3.0)

    def test_max_potential_energy(self) -> None:
        ke, pe = self._basic_arrays()
        result = _Stub().compute_energy_metrics(ke, pe)
        assert result["max_potential_energy"] == pytest.approx(5.0)

    def test_max_total_energy(self) -> None:
        ke, pe = self._basic_arrays()
        # total = [6, 5, 5]; max = 6
        result = _Stub().compute_energy_metrics(ke, pe)
        assert result["max_total_energy"] == pytest.approx(6.0)

    def test_energy_drift(self) -> None:
        ke = np.array([1.0, 2.0, 3.0])
        pe = np.array([5.0, 5.0, 6.0])
        result = _Stub().compute_energy_metrics(ke, pe)
        # drift = (3+6) - (1+5) = 9 - 6 = 3
        assert result["energy_drift"] == pytest.approx(3.0)

    def test_energy_drift_zero_for_constant_total(self) -> None:
        ke = np.array([1.0, 2.0, 3.0])
        pe = np.array([4.0, 3.0, 2.0])
        result = _Stub().compute_energy_metrics(ke, pe)
        # total always 5; drift = 5 - 5 = 0
        assert result["energy_drift"] == pytest.approx(0.0)

    def test_energy_efficiency_zero_when_no_club_head_speed(self) -> None:
        ke = np.array([2.0, 4.0])
        pe = np.array([1.0, 1.0])
        result = _Stub(club_head_speed=None).compute_energy_metrics(ke, pe)
        assert result["energy_efficiency"] == pytest.approx(0.0)

    def test_energy_efficiency_with_club_head_speed(self) -> None:
        ke = np.array([1.0, 4.0, 2.0])
        pe = np.array([2.0, 1.0, 3.0])
        # total = [3, 5, 5]; max_total = 5
        # club_head_speed peaks at index 1 → ke_at_impact = 4
        # efficiency = (4 / 5) * 100 = 80%
        chs = np.array([1.0, 3.0, 2.0])
        stub = _Stub(club_head_speed=chs)
        result = stub.compute_energy_metrics(ke, pe)
        assert result["energy_efficiency"] == pytest.approx(80.0)

    def test_energy_efficiency_non_negative(self) -> None:
        ke = np.array([0.5, 1.5])
        pe = np.array([2.0, 1.0])
        result = _Stub().compute_energy_metrics(ke, pe)
        assert result["energy_efficiency"] >= 0.0


# ---------------------------------------------------------------------------
# Tests — postconditions
# ---------------------------------------------------------------------------


class TestEnergyMetricsPostconditions:
    def test_all_returned_values_finite(self) -> None:
        ke = np.array([1.0, 2.0, 3.0])
        pe = np.array([3.0, 2.0, 1.0])
        result = _Stub().compute_energy_metrics(ke, pe)
        for key, val in result.items():
            assert np.isfinite(val), f"metric '{key}' is not finite: {val}"

    def test_energy_efficiency_non_negative_postcondition(self) -> None:
        ke = np.array([0.0, 0.0])
        pe = np.array([1.0, 1.0])
        # total energy = 1 everywhere; ke_at_impact = 0 → efficiency = 0
        chs = np.array([0.5, 0.5])
        stub = _Stub(club_head_speed=chs)
        result = stub.compute_energy_metrics(ke, pe)
        assert result["energy_efficiency"] >= 0.0

    def test_single_element_arrays(self) -> None:
        ke = np.array([5.0])
        pe = np.array([3.0])
        result = _Stub().compute_energy_metrics(ke, pe)
        assert result["max_kinetic_energy"] == pytest.approx(5.0)
        assert result["max_total_energy"] == pytest.approx(8.0)
        assert result["energy_drift"] == pytest.approx(0.0)
