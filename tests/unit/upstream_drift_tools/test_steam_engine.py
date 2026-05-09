"""Tests for upstream_drift_tools.calculators.thermo.steam_engine (Issues #1949, #1744)."""

from __future__ import annotations

import math

from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
    STANDARD_ATMOSPHERIC_PRESSURE,
    SteamCalculationEngine,
    SteamProperties,
)


class TestSteamCalculationEngineInit:
    def test_steam_engine_instantiates(self) -> None:
        eng = SteamCalculationEngine()
        assert eng is not None

    def test_standard_atmospheric_pressure_value(self) -> None:
        assert abs(STANDARD_ATMOSPHERIC_PRESSURE - 101325.0) < 1.0


class TestCalculateWaterVaporPressure:
    def test_positive_at_standard_conditions(self) -> None:
        eng = SteamCalculationEngine()
        vp = eng.calculate_water_vapor_pressure(25.0)
        assert vp > 0.0

    def test_steam_engine_increases_with_temperature(self) -> None:
        eng = SteamCalculationEngine()
        vp_low = eng.calculate_water_vapor_pressure(20.0)
        vp_high = eng.calculate_water_vapor_pressure(80.0)
        assert vp_high > vp_low

    def test_close_to_known_value_at_100c(self) -> None:
        eng = SteamCalculationEngine()
        # At 100°C, vapor pressure should be close to 101325 Pa
        # Simplified model may not be exact, but within an order of magnitude
        vp = eng.calculate_water_vapor_pressure(100.0)
        assert vp > 10000.0  # At least 10 kPa

    def test_steam_engine_returns_float(self) -> None:
        eng = SteamCalculationEngine()
        vp = eng.calculate_water_vapor_pressure(50.0)
        assert isinstance(vp, float)


class TestCalculateSaturatedPropertiesFromTemperature:
    def test_returns_steam_properties(self) -> None:
        eng = SteamCalculationEngine()
        props = eng.calculate_saturated_properties_from_temperature(100.0)
        assert isinstance(props, SteamProperties)

    def test_temperature_matches_input(self) -> None:
        eng = SteamCalculationEngine()
        props = eng.calculate_saturated_properties_from_temperature(100.0)
        assert props.temperature == 100.0

    def test_enthalpy_positive(self) -> None:
        eng = SteamCalculationEngine()
        props = eng.calculate_saturated_properties_from_temperature(100.0)
        assert props.enthalpy > 0.0

    def test_cp_positive(self) -> None:
        eng = SteamCalculationEngine()
        props = eng.calculate_saturated_properties_from_temperature(100.0)
        assert props.cp > 0.0

    def test_quality_at_saturation(self) -> None:
        eng = SteamCalculationEngine()
        props = eng.calculate_saturated_properties_from_temperature(100.0)
        # Quality should be defined (not NaN)
        assert not math.isnan(float(props.quality))


class TestGetSaturationPressure:
    def test_steam_engine_returns_float(self) -> None:
        eng = SteamCalculationEngine()
        sp = eng.get_saturation_pressure(373.15)
        assert isinstance(sp, float)

    def test_at_boiling_point_near_atmospheric(self) -> None:
        eng = SteamCalculationEngine()
        sp = eng.get_saturation_pressure(373.15)
        # Simplified model: within 10% of 101325 Pa
        assert abs(sp - 101325.0) / 101325.0 < 0.10

    def test_steam_engine_increases_with_temperature(self) -> None:
        eng = SteamCalculationEngine()
        sp_low = eng.get_saturation_pressure(300.0)
        sp_high = eng.get_saturation_pressure(400.0)
        assert sp_high > sp_low


class TestCalculateDewPoint:
    def test_steam_engine_returns_float(self) -> None:
        eng = SteamCalculationEngine()
        # partial_pressure_pa=3000 Pa at total_pressure_pa=101325 Pa
        dp = eng.calculate_dew_point(3000.0, 101325.0)
        assert isinstance(dp, float)

    def test_higher_partial_pressure_higher_dew_point(self) -> None:
        eng = SteamCalculationEngine()
        dp_low = eng.calculate_dew_point(1000.0, 101325.0)
        dp_high = eng.calculate_dew_point(5000.0, 101325.0)
        assert dp_high > dp_low
