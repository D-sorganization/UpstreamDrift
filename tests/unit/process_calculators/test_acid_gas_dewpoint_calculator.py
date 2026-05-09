"""Tests for upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.acid_gas_dewpoint_calculator import (
    AcidGasComposition,
    AcidGasDewpointCalculator,
    DewpointResult,
    estimate_condensation_risk,
    quick_dewpoint_calculation,
)


class TestAcidGasComposition:
    def test_acid_gas_dewpoint_calculator_default_construction(self) -> None:
        comp = AcidGasComposition()
        assert comp.h2o == 0.0
        assert comp.hcl == 0.0

    def test_acid_gas_dewpoint_calculator_custom_values(self) -> None:
        comp = AcidGasComposition(h2o=0.1, hcl=0.05)
        assert comp.h2o == pytest.approx(0.1)
        assert comp.hcl == pytest.approx(0.05)

    def test_acid_gas_dewpoint_calculator_normalize_sums_to_one(self) -> None:
        comp = AcidGasComposition(h2o=2.0, hcl=2.0, h2s=4.0, other=2.0)
        norm = comp.normalize()
        assert norm.total == pytest.approx(1.0)

    def test_normalize_preserves_ratios(self) -> None:
        comp = AcidGasComposition(h2o=1.0, hcl=1.0)
        norm = comp.normalize()
        assert norm.h2o == pytest.approx(norm.hcl)

    def test_normalize_empty_unchanged(self) -> None:
        comp = AcidGasComposition()
        norm = comp.normalize()
        assert norm.total == pytest.approx(0.0)

    def test_total_property(self) -> None:
        comp = AcidGasComposition(h2o=0.3, hcl=0.2, h2s=0.1, hf=0.1, other=0.3)
        assert comp.total == pytest.approx(1.0)

    def test_acid_gas_dewpoint_calculator_to_dict_keys(self) -> None:
        comp = AcidGasComposition(h2o=0.5)
        d = comp.to_dict()
        assert "H2O" in d
        assert "HCl" in d
        assert "H2S" in d
        assert "HF" in d

    def test_to_dict_values(self) -> None:
        comp = AcidGasComposition(h2o=0.5, hcl=0.1)
        d = comp.to_dict()
        assert d["H2O"] == pytest.approx(0.5)
        assert d["HCl"] == pytest.approx(0.1)


class TestAcidGasDewpointCalculator:
    def _make_comp(self) -> AcidGasComposition:
        return AcidGasComposition(h2o=0.1, hcl=0.02, other=0.88)

    def test_acid_gas_dewpoint_calculator_construction(self) -> None:
        calc = AcidGasDewpointCalculator()
        assert calc is not None

    def test_antoine_constants_present(self) -> None:
        calc = AcidGasDewpointCalculator()
        assert "H2O" in calc.antoine_constants
        assert "HCl" in calc.antoine_constants

    def test_calculate_dewpoint_returns_result(self) -> None:
        calc = AcidGasDewpointCalculator()
        comp = self._make_comp()
        result = calc.calculate_dewpoint_mixture(200.0, 1.5, comp)
        assert isinstance(result, DewpointResult)

    def test_dewpoint_result_has_overall_dewpoint(self) -> None:
        calc = AcidGasDewpointCalculator()
        result = calc.calculate_dewpoint_mixture(200.0, 1.5, self._make_comp())
        assert isinstance(result.overall_dewpoint_c, float)

    def test_dewpoint_result_has_limiting_component(self) -> None:
        calc = AcidGasDewpointCalculator()
        result = calc.calculate_dewpoint_mixture(200.0, 1.5, self._make_comp())
        assert isinstance(result.limiting_component, str)

    def test_dewpoint_result_has_condensation_risk(self) -> None:
        calc = AcidGasDewpointCalculator()
        result = calc.calculate_dewpoint_mixture(200.0, 1.5, self._make_comp())
        assert isinstance(result.condensation_risk, str)

    def test_high_temp_no_condensation(self) -> None:
        calc = AcidGasDewpointCalculator()
        result = calc.calculate_dewpoint_mixture(400.0, 1.0, self._make_comp())
        # At high temperature, margin should be positive (no condensation)
        assert result.dewpoint_margin_c > 0 or result.condensation_risk in (
            "Low",
            "Medium",
            "None",
        )

    def test_warnings_is_list(self) -> None:
        calc = AcidGasDewpointCalculator()
        result = calc.calculate_dewpoint_mixture(200.0, 1.5, self._make_comp())
        assert isinstance(result.warnings, list)


class TestQuickDewpointCalculation:
    def test_acid_gas_dewpoint_calculator_returns_dict(self) -> None:
        result = quick_dewpoint_calculation(200.0, 1.5, h2o_fraction=0.1)
        assert isinstance(result, dict)

    def test_has_overall_dewpoint_key(self) -> None:
        result = quick_dewpoint_calculation(200.0, 1.5, h2o_fraction=0.1)
        assert "overall_dewpoint_c" in result

    def test_has_condensation_risk_key(self) -> None:
        result = quick_dewpoint_calculation(200.0, 1.5, h2o_fraction=0.1)
        assert "condensation_risk" in result

    def test_has_dewpoint_margin_key(self) -> None:
        result = quick_dewpoint_calculation(200.0, 1.5, h2o_fraction=0.1)
        assert "dewpoint_margin_c" in result

    def test_limiting_component_present(self) -> None:
        result = quick_dewpoint_calculation(
            200.0, 1.5, h2o_fraction=0.1, hcl_fraction=0.05
        )
        assert "limiting_component" in result


class TestEstimateCondensationRisk:
    def _comp(self) -> AcidGasComposition:
        return AcidGasComposition(h2o=0.1, hcl=0.02, other=0.88)

    def test_acid_gas_dewpoint_calculator_returns_dict(self) -> None:
        result = estimate_condensation_risk(200.0, 1.5, self._comp())
        assert isinstance(result, dict)

    def test_has_risk_level_key(self) -> None:
        result = estimate_condensation_risk(200.0, 1.5, self._comp())
        assert "risk_level" in result

    def test_has_recommendation_key(self) -> None:
        result = estimate_condensation_risk(200.0, 1.5, self._comp())
        assert "recommendation" in result
