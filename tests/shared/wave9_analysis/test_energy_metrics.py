"""Tests for src/shared/python/analysis/energy_metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.analysis.energy_metrics import EnergyMetricsMixin


class _Holder(EnergyMetricsMixin):
    def __init__(self, club_head_speed: np.ndarray | None = None) -> None:
        self.club_head_speed = club_head_speed


class TestComputeEnergyMetrics:
    def test_basic(self) -> None:
        h = _Holder(club_head_speed=np.array([1.0, 2.0, 5.0, 3.0, 1.0]))
        ke = np.array([1.0, 2.0, 4.0, 3.0, 1.0])
        pe = np.array([5.0, 4.0, 3.0, 4.0, 5.0])
        result = h.compute_energy_metrics(ke, pe)
        assert result["max_kinetic_energy"] == 4.0
        assert result["max_potential_energy"] == 5.0
        # total = [6,6,7,7,6], max=7
        assert result["max_total_energy"] == 7.0
        # KE at impact (idx=2 where chs is max) = 4, max_total=7 -> ~57.14
        assert result["energy_efficiency"] == pytest.approx(4.0 / 7.0 * 100.0)
        assert result["energy_variation"] >= 0
        assert result["energy_drift"] == 0.0  # ends same as starts

    def test_no_club_head_speed(self) -> None:
        h = _Holder(club_head_speed=None)
        ke = np.array([1.0, 2.0, 3.0])
        pe = np.array([3.0, 2.0, 1.0])
        result = h.compute_energy_metrics(ke, pe)
        assert result["energy_efficiency"] == 0.0

    def test_zero_max_total(self) -> None:
        h = _Holder(club_head_speed=np.array([1.0, 2.0, 3.0]))
        ke = np.array([0.0, 0.0, 0.0])
        pe = np.array([0.0, 0.0, 0.0])
        result = h.compute_energy_metrics(ke, pe)
        assert result["energy_efficiency"] == 0.0
        assert result["energy_drift"] == 0.0

    def test_mismatched_lengths(self) -> None:
        h = _Holder()
        with pytest.raises(ValueError):
            h.compute_energy_metrics(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self) -> None:
        h = _Holder()
        with pytest.raises(ValueError):
            h.compute_energy_metrics(np.array([]), np.array([]))

    def test_negative_ke_raises(self) -> None:
        h = _Holder()
        with pytest.raises(ValueError):
            h.compute_energy_metrics(np.array([-1.0, 1.0]), np.array([1.0, 1.0]))

    def test_nonfinite_raises(self) -> None:
        h = _Holder()
        with pytest.raises(ValueError):
            h.compute_energy_metrics(np.array([1.0, np.nan]), np.array([1.0, 1.0]))

    def test_none_ke_raises(self) -> None:
        h = _Holder()
        with pytest.raises(ValueError, match="must be provided"):
            h.compute_energy_metrics(None, np.array([1.0]))  # type: ignore[arg-type]

    def test_drift_detection(self) -> None:
        h = _Holder(club_head_speed=None)
        ke = np.array([1.0, 1.0, 1.0])
        pe = np.array([1.0, 2.0, 3.0])
        result = h.compute_energy_metrics(ke, pe)
        # total = [2,3,4]; drift = 4-2 = 2
        assert result["energy_drift"] == 2.0
