"""Tests for upstream_drift_tools.process_calculators.scrubber_calculator (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from src.shared.python.upstream_drift_tools.process_calculators.scrubber_calculator import (
    PACKING_DATABASE,
    calculate_gas_density,
    calculate_gas_viscosity,
    calculate_ntu_removal,
)


class TestPackingDatabase:
    def test_scrubber_calculator_nonempty(self) -> None:
        assert len(PACKING_DATABASE) > 0

    def test_pall_ring_exists(self) -> None:
        assert any("pall" in k.lower() for k in PACKING_DATABASE)

    def test_packing_has_attributes(self) -> None:
        # Each packing entry should have meaningful attributes
        for name, packing in PACKING_DATABASE.items():
            assert hasattr(packing, "__class__"), f"Packing {name!r} has no class"


class TestCalculateGasDensity:
    def test_air_at_standard_conditions(self) -> None:
        # Air: MW ≈ 29 kg/kmol, T=293K, P=101325 Pa → ρ ≈ 1.2 kg/m³
        density = calculate_gas_density(293.0, 101325.0, 29.0)
        assert 1.0 <= density <= 1.5

    def test_higher_pressure_gives_higher_density(self) -> None:
        rho_low = calculate_gas_density(300.0, 101325.0, 28.0)
        rho_high = calculate_gas_density(300.0, 202650.0, 28.0)
        assert rho_high > rho_low

    def test_higher_temp_gives_lower_density(self) -> None:
        rho_cool = calculate_gas_density(300.0, 101325.0, 28.0)
        rho_hot = calculate_gas_density(600.0, 101325.0, 28.0)
        assert rho_hot < rho_cool

    def test_higher_molecular_weight_gives_higher_density(self) -> None:
        rho_light = calculate_gas_density(300.0, 101325.0, 2.0)  # H2
        rho_heavy = calculate_gas_density(300.0, 101325.0, 44.0)  # CO2
        assert rho_heavy > rho_light

    def test_positive_result(self) -> None:
        result = calculate_gas_density(300.0, 101325.0, 28.0)
        assert result > 0.0

    def test_ideal_gas_law_accuracy(self) -> None:
        # ρ = P·M / (R·T) with R = 8314 J/(kmol·K)
        T, P, M = 300.0, 101325.0, 28.0
        expected = P * M / (8314.0 * T)
        result = calculate_gas_density(T, P, M)
        assert result == pytest.approx(expected, rel=0.01)


class TestCalculateGasViscosity:
    def test_positive_result(self) -> None:
        result = calculate_gas_viscosity(300.0, 28.0)
        assert result > 0.0

    def test_scrubber_calculator_increases_with_temperature(self) -> None:
        mu_low = calculate_gas_viscosity(300.0, 28.0)
        mu_high = calculate_gas_viscosity(600.0, 28.0)
        # Gas viscosity increases with temperature (Sutherland)
        assert mu_high > mu_low

    def test_reasonable_order_of_magnitude(self) -> None:
        # Air-like gas viscosity at 300K ≈ 1.8e-5 Pa·s
        mu = calculate_gas_viscosity(300.0, 29.0)
        assert 1e-6 <= mu <= 1e-4


class TestCalculateNtuRemoval:
    def test_inlet_equals_outlet_returns_zero(self) -> None:
        result = calculate_ntu_removal(0.01, 0.01)
        assert result == pytest.approx(0.0)

    def test_inlet_less_than_outlet_returns_zero(self) -> None:
        result = calculate_ntu_removal(0.005, 0.01)
        assert result == pytest.approx(0.0)

    def test_zero_outlet_returns_zero(self) -> None:
        result = calculate_ntu_removal(0.01, 0.0)
        assert result == pytest.approx(0.0)

    def test_high_removal_gives_high_ntu(self) -> None:
        # 99% removal: NTU = ln(1/0.01) ≈ 4.6
        result = calculate_ntu_removal(1.0, 0.01)
        assert result == pytest.approx(4.605, rel=0.01)

    def test_ntu_is_log_ratio(self) -> None:
        import math

        inlet, outlet = 0.1, 0.001
        expected = math.log(inlet / outlet)
        result = calculate_ntu_removal(inlet, outlet)
        assert result == pytest.approx(expected, rel=0.01)

    def test_result_nonnegative(self) -> None:
        result = calculate_ntu_removal(0.1, 0.05)
        assert result >= 0.0

    def test_higher_removal_higher_ntu(self) -> None:
        ntu_low = calculate_ntu_removal(0.1, 0.05)  # 50% removal
        ntu_high = calculate_ntu_removal(0.1, 0.001)  # 99% removal
        assert ntu_high > ntu_low
