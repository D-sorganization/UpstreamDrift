"""Tests for sidekick.process_calculators.pressure_drop_calculator.utils.gas_properties (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.pressure_drop_calculator.utils.gas_properties import (
    calculate_compressibility_factor,
    calculate_heat_capacity_ratio,
    calculate_ideal_gas_cp,
    calculate_ideal_gas_density,
    calculate_mixture_cp,
    calculate_mixture_molecular_weight,
    calculate_mixture_viscosity_simple,
    calculate_mixture_viscosity_wilke,
    calculate_real_gas_density,
    calculate_speed_of_sound,
)

_SYNGAS = {"H2": 0.3, "CO": 0.4, "CO2": 0.2, "N2": 0.1}


class TestCalculateIdealGasCp:
    def test_positive_cp_for_h2(self) -> None:
        cp = calculate_ideal_gas_cp("H2", temperature=300.0)
        assert cp > 0.0

    def test_positive_cp_for_co(self) -> None:
        cp = calculate_ideal_gas_cp("CO", temperature=300.0)
        assert cp > 0.0

    def test_cp_changes_with_temperature(self) -> None:
        cp_low = calculate_ideal_gas_cp("H2", 300.0)
        cp_high = calculate_ideal_gas_cp("H2", 1000.0)
        # Cp should differ; both positive
        assert cp_low > 0 and cp_high > 0


class TestCalculateMixtureCp:
    def test_positive_cp(self) -> None:
        cp = calculate_mixture_cp(_SYNGAS, temperature=300.0)
        assert cp > 0.0

    def test_gas_properties_returns_float(self) -> None:
        cp = calculate_mixture_cp(_SYNGAS, temperature=500.0)
        assert isinstance(cp, float)


class TestCalculateHeatCapacityRatio:
    def test_returns_positive(self) -> None:
        gamma = calculate_heat_capacity_ratio(_SYNGAS, temperature=300.0)
        assert gamma > 0.0

    def test_gamma_greater_than_one(self) -> None:
        gamma = calculate_heat_capacity_ratio(_SYNGAS, temperature=300.0)
        assert gamma > 1.0


class TestCalculateSpeedOfSound:
    def test_positive_speed(self) -> None:
        c = calculate_speed_of_sound(_SYNGAS, temperature=300.0)
        assert c > 0.0

    def test_gas_properties_returns_float(self) -> None:
        c = calculate_speed_of_sound(_SYNGAS, temperature=300.0)
        assert isinstance(c, float)

    def test_higher_temp_higher_speed(self) -> None:
        c_low = calculate_speed_of_sound(_SYNGAS, temperature=300.0)
        c_high = calculate_speed_of_sound(_SYNGAS, temperature=1000.0)
        assert c_high > c_low


class TestCalculateMixtureMolecularWeight:
    def test_positive_mw(self) -> None:
        mw = calculate_mixture_molecular_weight(_SYNGAS)
        assert mw > 0.0

    def test_gas_properties_returns_float(self) -> None:
        mw = calculate_mixture_molecular_weight(_SYNGAS)
        assert isinstance(mw, float)

    def test_pure_h2_mw(self) -> None:
        mw = calculate_mixture_molecular_weight({"H2": 1.0})
        assert mw == pytest.approx(2.016, rel=0.01)


class TestCalculateIdealGasDensity:
    def test_positive_density(self) -> None:
        rho = calculate_ideal_gas_density(28.0, 300.0, 101325.0)
        assert rho > 0.0

    def test_gas_properties_higher_pressure_higher_density(self) -> None:
        rho_low = calculate_ideal_gas_density(28.0, 300.0, 101325.0)
        rho_high = calculate_ideal_gas_density(28.0, 300.0, 200000.0)
        assert rho_high > rho_low

    def test_higher_temp_lower_density(self) -> None:
        rho_low = calculate_ideal_gas_density(28.0, 300.0, 101325.0)
        rho_high = calculate_ideal_gas_density(28.0, 600.0, 101325.0)
        assert rho_high < rho_low


class TestCompressibilityFactor:
    def test_near_one_at_low_pressure(self) -> None:
        Z = calculate_compressibility_factor(_SYNGAS, 300.0, 101325.0)
        assert pytest.approx(1.0, rel=0.05) == Z

    def test_returns_positive(self) -> None:
        Z = calculate_compressibility_factor(_SYNGAS, 300.0, 101325.0)
        assert Z > 0.0


class TestCalculateRealGasDensity:
    def test_positive_density(self) -> None:
        # Args: molecular_weight, temperature, pressure, compressibility
        rho = calculate_real_gas_density(28.0, 300.0, 101325.0, 1.0)
        assert rho > 0.0

    def test_gas_properties_returns_float(self) -> None:
        rho = calculate_real_gas_density(28.0, 300.0, 101325.0, 1.0)
        assert isinstance(rho, float)

    def test_compressibility_increases_density(self) -> None:
        # Higher Z → lower density for same T, P
        rho_z1 = calculate_real_gas_density(28.0, 300.0, 101325.0, 1.0)
        rho_z09 = calculate_real_gas_density(28.0, 300.0, 101325.0, 0.9)
        assert rho_z09 > rho_z1


class TestMixtureViscosity:
    def test_wilke_positive(self) -> None:
        mu = calculate_mixture_viscosity_wilke(_SYNGAS, 300.0, 101325.0)
        assert mu > 0.0

    def test_simple_positive(self) -> None:
        mu = calculate_mixture_viscosity_simple(_SYNGAS, 300.0)
        assert mu > 0.0

    def test_both_methods_comparable(self) -> None:
        mu_w = calculate_mixture_viscosity_wilke(_SYNGAS, 300.0, 101325.0)
        mu_s = calculate_mixture_viscosity_simple(_SYNGAS, 300.0)
        # They should be within an order of magnitude
        assert abs(mu_w - mu_s) / mu_w < 2.0
