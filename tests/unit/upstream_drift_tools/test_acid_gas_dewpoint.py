"""Tests for src.shared.python.upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
    AcidGasComposition,
    AcidGasDewpointCalculator,
    quick_dewpoint_calculation,
)

# ---------------------------------------------------------------------------
# AcidGasComposition
# ---------------------------------------------------------------------------


class TestAcidGasComposition:
    def test_defaults_zero(self) -> None:
        c = AcidGasComposition()
        assert c.h2o == 0.0
        assert c.total == 0.0

    def test_total(self) -> None:
        c = AcidGasComposition(h2o=0.3, hcl=0.1, other=0.6)
        assert abs(c.total - 1.0) < 1e-10

    def test_normalize(self) -> None:
        c = AcidGasComposition(h2o=30.0, hcl=10.0, other=60.0)
        n = c.normalize()
        assert abs(n.total - 1.0) < 1e-6
        assert abs(n.h2o - 0.3) < 1e-6

    def test_normalize_zeros_returns_same(self) -> None:
        c = AcidGasComposition()
        n = c.normalize()
        assert n.total == 0.0

    def test_acid_gas_dewpoint_to_dict_keys(self) -> None:
        c = AcidGasComposition(h2o=0.5)
        d = c.to_dict()
        assert "H2O" in d
        assert "HCl" in d


# ---------------------------------------------------------------------------
# AcidGasDewpointCalculator.calculate_vapor_pressure
# ---------------------------------------------------------------------------


class TestCalculateVaporPressure:
    _CALC = AcidGasDewpointCalculator()

    def test_h2o_at_100c_approx_atm(self) -> None:
        # At 100°C, water vapor pressure ≈ 101325 Pa
        vp = self._CALC.calculate_vapor_pressure(100.0, "H2O")
        assert abs(vp - 101325.0) / 101325.0 < 0.05  # within 5%

    def test_acid_gas_dewpoint_vapor_pressure_positive(self) -> None:
        vp = self._CALC.calculate_vapor_pressure(50.0, "H2O")
        assert vp > 0.0

    def test_higher_temp_higher_vp(self) -> None:
        vp_low = self._CALC.calculate_vapor_pressure(50.0, "H2O")
        vp_high = self._CALC.calculate_vapor_pressure(80.0, "H2O")
        assert vp_high > vp_low

    def test_hcl_vapor_pressure_positive(self) -> None:
        vp = self._CALC.calculate_vapor_pressure(25.0, "HCl")
        assert vp > 0.0

    def test_unknown_component_raises(self) -> None:
        with pytest.raises(ValueError):
            self._CALC.calculate_vapor_pressure(50.0, "UNKNOWN_GAS")

    def test_extended_antoine_method(self) -> None:
        vp = self._CALC.calculate_vapor_pressure(80.0, "H2O", method="extended_antoine")
        assert vp > 0.0

    def test_acid_gas_dewpoint_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError):
            self._CALC.calculate_vapor_pressure(50.0, "H2O", method="magic")


# ---------------------------------------------------------------------------
# quick_dewpoint_calculation
# ---------------------------------------------------------------------------


class TestQuickDewpointCalculation:
    def test_acid_gas_dewpoint_returns_dict(self) -> None:
        result = quick_dewpoint_calculation(200.0, 1.0, h2o_fraction=0.1)
        assert isinstance(result, dict)

    def test_required_keys(self) -> None:
        result = quick_dewpoint_calculation(200.0, 1.0, h2o_fraction=0.1)
        assert "overall_dewpoint_c" in result
        assert "limiting_component" in result
        assert "condensation_risk" in result
        assert "dewpoint_margin_c" in result

    def test_temp_above_dewpoint_no_condensation(self) -> None:
        result = quick_dewpoint_calculation(300.0, 1.0, h2o_fraction=0.01)
        # At high T and low fraction, we expect no condensation
        assert (
            "low" in str(result["condensation_risk"]).lower()
            or result["dewpoint_margin_c"] > 0
        )

    def test_dewpoint_is_numeric(self) -> None:
        result = quick_dewpoint_calculation(150.0, 2.0, h2o_fraction=0.2)
        assert isinstance(result["overall_dewpoint_c"], float)
