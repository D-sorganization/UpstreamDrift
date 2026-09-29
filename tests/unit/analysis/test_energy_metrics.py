"""Tests for src.shared.python.analysis.energy_metrics (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.analysis.energy_metrics import EnergyMetricsMixin


class _Concrete(EnergyMetricsMixin):
    """Minimal concrete subclass for testing EnergyMetricsMixin."""

    def __init__(self, n: int = 50) -> None:
        self.club_head_speed = np.abs(np.sin(np.linspace(0, np.pi, n))) * 40.0


class TestComputeEnergyMetrics:
    def setup_method(self) -> None:
        n = 50
        self.obj = _Concrete(n=n)
        self.ke = 0.5 * np.sin(np.linspace(0, np.pi, n)) ** 2 + 0.1
        self.pe = np.cos(np.linspace(0, np.pi / 2, n)) ** 2 + 0.1

    def test_energy_metrics_returns_dict(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert isinstance(result, dict)

    def test_has_max_kinetic_energy(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert "max_kinetic_energy" in result

    def test_has_max_potential_energy(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert "max_potential_energy" in result

    def test_has_max_total_energy(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert "max_total_energy" in result

    def test_has_energy_efficiency(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert "energy_efficiency" in result

    def test_has_energy_variation(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert "energy_variation" in result

    def test_has_energy_drift(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert "energy_drift" in result

    def test_max_kinetic_energy_positive(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert result["max_kinetic_energy"] > 0.0

    def test_max_total_energy_gte_ke(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert result["max_total_energy"] >= result["max_kinetic_energy"]

    def test_energy_efficiency_non_negative(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        assert result["energy_efficiency"] >= 0.0

    def test_energy_metrics_all_values_finite(self) -> None:
        result = self.obj.compute_energy_metrics(self.ke, self.pe)
        for key, val in result.items():
            assert np.isfinite(val), f"'{key}' is not finite"

    def test_constant_energy_zero_variation(self) -> None:
        ke = np.ones(50) * 2.0
        pe = np.ones(50) * 1.0
        result = self.obj.compute_energy_metrics(ke, pe)
        assert result["energy_variation"] == pytest.approx(0.0, abs=1e-12)

    def test_constant_energy_zero_drift(self) -> None:
        ke = np.ones(50) * 2.0
        pe = np.ones(50) * 1.0
        result = self.obj.compute_energy_metrics(ke, pe)
        assert result["energy_drift"] == pytest.approx(0.0, abs=1e-12)

    def test_negative_ke_raises(self) -> None:
        from src.shared.python.core.contracts.exceptions import PreconditionError

        ke_neg = np.ones(50) * -1.0
        with pytest.raises(PreconditionError):
            self.obj.compute_energy_metrics(ke_neg, self.pe)

    def test_mismatched_lengths_raises(self) -> None:
        from src.shared.python.core.contracts.exceptions import PreconditionError

        ke_short = np.ones(30)
        with pytest.raises(PreconditionError):
            self.obj.compute_energy_metrics(ke_short, self.pe)

    def test_empty_ke_raises(self) -> None:
        from src.shared.python.core.contracts.exceptions import PreconditionError

        with pytest.raises(PreconditionError):
            self.obj.compute_energy_metrics(np.array([]), np.array([]))

    def test_no_club_head_speed_efficiency_zero(self) -> None:
        obj = _Concrete(n=50)
        obj.club_head_speed = None
        result = obj.compute_energy_metrics(self.ke, self.pe)
        assert result["energy_efficiency"] == pytest.approx(0.0)
