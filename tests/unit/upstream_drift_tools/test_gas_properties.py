"""Tests for pressure_drop_calculator gas_properties utilities (Issues #1949, #1744)."""

from __future__ import annotations

from sidekick.process_calculators.pressure_drop_calculator.utils.gas_properties import (
    calculate_ideal_gas_density,
    calculate_mixture_molecular_weight,
)

_SYNGAS = {"H2": 0.4, "CO": 0.3, "CO2": 0.2, "N2": 0.1}


class TestCalculateMixtureMolecularWeight:
    def test_returns_positive_value(self) -> None:
        mw = calculate_mixture_molecular_weight(_SYNGAS)
        assert mw > 0.0

    def test_pure_n2_approx_28(self) -> None:
        mw = calculate_mixture_molecular_weight({"N2": 1.0})
        assert abs(mw - 28.0) < 1.0

    def test_pure_h2_approx_2(self) -> None:
        mw = calculate_mixture_molecular_weight({"H2": 1.0})
        assert abs(mw - 2.0) < 0.5

    def test_h2_rich_lighter_than_n2_rich(self) -> None:
        h2_rich = calculate_mixture_molecular_weight({"H2": 0.9, "N2": 0.1})
        n2_rich = calculate_mixture_molecular_weight({"H2": 0.1, "N2": 0.9})
        assert h2_rich < n2_rich

    def test_unknown_component_skipped(self) -> None:
        # Should not raise, just skip unknown
        mw = calculate_mixture_molecular_weight({"N2": 0.5, "UNKNOWN_GAS": 0.5})
        assert mw > 0.0


class TestCalculateIdealGasDensity:
    def test_returns_positive(self) -> None:
        density = calculate_ideal_gas_density(28.0, 300.0, 101325.0)
        assert density > 0.0

    def test_n2_at_stp_approx(self) -> None:
        # N2 at 273.15 K, 101325 Pa: ~1.25 kg/m3
        density = calculate_ideal_gas_density(28.0, 273.15, 101325.0)
        assert abs(density - 1.25) < 0.05

    def test_gas_properties_higher_pressure_higher_density(self) -> None:
        low = calculate_ideal_gas_density(28.0, 300.0, 101325.0)
        high = calculate_ideal_gas_density(28.0, 300.0, 500000.0)
        assert high > low

    def test_higher_temperature_lower_density(self) -> None:
        cold = calculate_ideal_gas_density(28.0, 300.0, 101325.0)
        hot = calculate_ideal_gas_density(28.0, 600.0, 101325.0)
        assert hot < cold
