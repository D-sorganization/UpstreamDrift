"""Tests for upstream_drift_tools.process_calculators.wgs_reactor_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import math

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.wgs_reactor_calculator import (
    WGSReactorEngine,
)

_SYNGAS = {"CO": 30.0, "H2": 25.0, "CO2": 10.0, "H2O": 5.0}


class TestWGSReactorEngineConstruction:
    def test_wgs_reactor_calculator_construction(self) -> None:
        engine = WGSReactorEngine()
        assert engine is not None

    def test_has_gas_constant(self) -> None:
        engine = WGSReactorEngine()
        assert pytest.approx(8.314, rel=1e-3) == engine.R


class TestEquilibriumConstant:
    def test_returns_positive_float(self) -> None:
        engine = WGSReactorEngine()
        K = engine.calculate_equilibrium_constant(temperature=500.0)
        assert K > 0.0

    def test_low_temp_high_K(self) -> None:
        engine = WGSReactorEngine()
        K_low = engine.calculate_equilibrium_constant(temperature=300.0)
        K_high = engine.calculate_equilibrium_constant(temperature=1000.0)
        # WGS is exothermic, K decreases with increasing temperature
        assert K_low > K_high

    def test_value_at_500k(self) -> None:
        engine = WGSReactorEngine()
        K = engine.calculate_equilibrium_constant(temperature=500.0)
        # Should be finite and not NaN
        assert math.isfinite(K)


class TestEquilibriumComposition:
    def test_wgs_reactor_calculator_returns_dict(self) -> None:
        engine = WGSReactorEngine()
        result = engine.calculate_equilibrium_composition(
            _SYNGAS, temperature=500.0, pressure=1.0
        )
        assert isinstance(result, dict)

    def test_has_conversion_key(self) -> None:
        engine = WGSReactorEngine()
        result = engine.calculate_equilibrium_composition(_SYNGAS, 500.0, 1.0)
        assert "conversion" in result

    def test_has_composition_key(self) -> None:
        engine = WGSReactorEngine()
        result = engine.calculate_equilibrium_composition(_SYNGAS, 500.0, 1.0)
        assert "composition" in result

    def test_conversion_in_range(self) -> None:
        engine = WGSReactorEngine()
        result = engine.calculate_equilibrium_composition(_SYNGAS, 500.0, 1.0)
        assert 0.0 <= result["conversion"] <= 100.0

    def test_h2_co_ratio_present(self) -> None:
        engine = WGSReactorEngine()
        result = engine.calculate_equilibrium_composition(_SYNGAS, 500.0, 1.0)
        assert "h2_co_ratio" in result

    def test_equilibrium_constant_in_result(self) -> None:
        engine = WGSReactorEngine()
        result = engine.calculate_equilibrium_composition(_SYNGAS, 500.0, 1.0)
        assert "equilibrium_constant" in result
        assert result["equilibrium_constant"] > 0.0

    def test_empty_composition_returns_zeros(self) -> None:
        engine = WGSReactorEngine()
        result = engine.calculate_equilibrium_composition({}, 500.0, 1.0)
        assert result["conversion"] == pytest.approx(0.0)

    def test_low_temp_higher_conversion(self) -> None:
        engine = WGSReactorEngine()
        result_low = engine.calculate_equilibrium_composition(_SYNGAS, 300.0, 1.0)
        result_high = engine.calculate_equilibrium_composition(_SYNGAS, 800.0, 1.0)
        # At lower temperature, exothermic reaction favors products → higher conversion
        assert result_low["conversion"] >= result_high["conversion"]

    def test_heat_released_present(self) -> None:
        engine = WGSReactorEngine()
        result = engine.calculate_equilibrium_composition(_SYNGAS, 500.0, 1.0)
        assert "heat_released" in result
