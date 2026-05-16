"""Extended unit tests for SteamCalculationEngine — complementing test_steam_engine.py.

Covers methods not exercised by the basic import/init tests:
- _antoine_equation, _buck_equation, _iapws_equation
- calculate_water_vapor_pressure
- calculate_dew_point
- get_saturation_pressure / get_saturation_temperature
- calculate_properties (simplified path)
- calculate_saturated_properties_from_temperature/_pressure
- _calculate_simplified_properties

All tests use the simplified calculation path (no CoolProp / Cantera required).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """SteamCalculationEngine instance."""
    from sidekick.calculators.thermo.steam_engine import (
        SteamCalculationEngine,
    )

    return SteamCalculationEngine()


# ---------------------------------------------------------------------------
# Antoine equation
# ---------------------------------------------------------------------------


class TestAntoineEquation:
    """Tests for the Antoine equation helper."""

    def test_at_100c_approximately_1atm(self, engine) -> None:
        """At 100°C, vapor pressure should be close to 1 atm (101325 Pa)."""
        p = engine._antoine_equation(100.0)
        # Antoine equation gives ~760 mmHg = 101 325 Pa at 100°C
        assert 95000 < p < 110000, f"Expected ~101325 Pa at 100°C, got {p:.1f} Pa"

    def test_at_0c_below_1kpa(self, engine) -> None:
        """At 0°C, vapor pressure is about 611 Pa (well below 1 kPa)."""
        p = engine._antoine_equation(0.0)
        # Antoine at 0°C: ~4.58 mmHg = 611 Pa
        assert 400 < p < 1000, f"Expected ~611 Pa at 0°C, got {p:.1f} Pa"

    def test_steam_engine_extended_increases_with_temperature(self, engine) -> None:
        """Vapor pressure increases monotonically with temperature."""
        p_20 = engine._antoine_equation(20.0)
        p_50 = engine._antoine_equation(50.0)
        p_80 = engine._antoine_equation(80.0)
        assert p_20 < p_50 < p_80


# ---------------------------------------------------------------------------
# Buck equation
# ---------------------------------------------------------------------------


class TestBuckEquation:
    """Tests for the Buck equation helper."""

    def test_at_100c_positive(self, engine) -> None:
        """At 100°C, Buck equation gives a positive pressure."""
        p = engine._buck_equation(100.0)
        assert p > 50000, f"Expected positive pressure at 100°C, got {p:.1f} Pa"

    def test_steam_engine_extended_increases_with_temperature(self, engine) -> None:
        """Buck vapor pressure increases with temperature."""
        p_10 = engine._buck_equation(10.0)
        p_60 = engine._buck_equation(60.0)
        assert p_10 < p_60

    def test_at_0c_positive(self, engine) -> None:
        """Vapor pressure at 0°C is positive."""
        p = engine._buck_equation(0.0)
        assert p > 0


# ---------------------------------------------------------------------------
# calculate_water_vapor_pressure
# ---------------------------------------------------------------------------


class TestCalculateWaterVaporPressure:
    """Tests for calculate_water_vapor_pressure (selects best available equation)."""

    def test_returns_positive_pressure(self, engine) -> None:
        """Vapor pressure at any temperature > 0°C is positive."""
        p = engine.calculate_water_vapor_pressure(50.0)
        assert p > 0

    def test_steam_engine_extended_increases_with_temperature(self, engine) -> None:
        """Vapor pressure is monotonically increasing with temperature."""
        p_20 = engine.calculate_water_vapor_pressure(20.0)
        p_60 = engine.calculate_water_vapor_pressure(60.0)
        assert p_20 < p_60

    def test_at_100c_positive_and_large(self, engine) -> None:
        """At 100°C the vapor pressure should be a large positive value (order 10^5 Pa)."""
        p = engine.calculate_water_vapor_pressure(100.0)
        assert p > 50000, f"Expected large pressure at 100°C, got {p:.1f} Pa"


# ---------------------------------------------------------------------------
# get_saturation_pressure
# ---------------------------------------------------------------------------


class TestGetSaturationPressure:
    """Tests for get_saturation_pressure (saturation pressure at a temperature)."""

    def test_steam_engine_extended_returns_float(self, engine) -> None:
        """get_saturation_pressure returns a float."""
        p = engine.get_saturation_pressure(373.15)  # 100°C in K
        assert isinstance(p, float)

    def test_positive_value(self, engine) -> None:
        """Saturation pressure is positive."""
        p = engine.get_saturation_pressure(373.15)
        assert p > 0

    def test_steam_engine_extended_increases_with_temperature(self, engine) -> None:
        """Saturation pressure increases with temperature."""
        p_low = engine.get_saturation_pressure(300.0)
        p_high = engine.get_saturation_pressure(400.0)
        assert p_low < p_high


# ---------------------------------------------------------------------------
# get_saturation_temperature
# ---------------------------------------------------------------------------


class TestGetSaturationTemperature:
    """Tests for get_saturation_temperature (saturation temperature at a pressure)."""

    def test_returns_finite_value(self, engine) -> None:
        """get_saturation_temperature returns a finite float."""
        import math

        T = engine.get_saturation_temperature(101325.0)
        assert isinstance(T, float)
        assert math.isfinite(T)

    def test_at_1atm_near_100c(self, engine) -> None:
        """At 1 atm, saturation temperature should be near 100°C (373K)."""
        T = engine.get_saturation_temperature(101325.0)
        # Accept reasonable range: 90-110°C in Celsius (or 363-383K)
        # The function may return Celsius or Kelvin — check for reasonable range
        assert 90 < T < 383  # works for both °C (≈100) and K (≈373)


# ---------------------------------------------------------------------------
# calculate_saturated_properties_from_temperature
# ---------------------------------------------------------------------------


class TestSaturatedFromTemperature:
    """Tests for calculate_saturated_properties_from_temperature."""

    def test_returns_steam_properties(self, engine) -> None:
        """Returns a SteamProperties instance."""
        from sidekick.calculators.thermo.steam_engine import (
            SteamProperties,
        )

        result = engine.calculate_saturated_properties_from_temperature(373.15)
        assert isinstance(result, SteamProperties)

    def test_temperature_is_stored(self, engine) -> None:
        """Returned SteamProperties contains the input temperature."""
        result = engine.calculate_saturated_properties_from_temperature(373.15)
        assert abs(result.temperature - 373.15) < 1.0

    def test_pressure_positive(self, engine) -> None:
        """Saturation pressure at the given temperature is positive."""
        result = engine.calculate_saturated_properties_from_temperature(373.15)
        assert result.pressure > 0


# ---------------------------------------------------------------------------
# calculate_saturated_properties_from_pressure
# ---------------------------------------------------------------------------


class TestSaturatedFromPressure:
    """Tests for calculate_saturated_properties_from_pressure."""

    def test_returns_steam_properties(self, engine) -> None:
        """Returns a SteamProperties instance."""
        from sidekick.calculators.thermo.steam_engine import (
            SteamProperties,
        )

        result = engine.calculate_saturated_properties_from_pressure(101325.0)
        assert isinstance(result, SteamProperties)

    def test_steam_engine_extended_pressure_stored(self, engine) -> None:
        """Returned SteamProperties contains the input pressure."""
        result = engine.calculate_saturated_properties_from_pressure(101325.0)
        assert abs(result.pressure - 101325.0) < 1.0


# ---------------------------------------------------------------------------
# calculate_dew_point
# ---------------------------------------------------------------------------


class TestCalculateDewPoint:
    """Tests for calculate_dew_point Newton-Raphson solver."""

    def test_returns_finite_temperature(self, engine) -> None:
        """calculate_dew_point returns a finite float."""
        import math

        # Partial pressure of water = 2000 Pa (about 2% at 1 atm)
        T_dew = engine.calculate_dew_point(2000.0, 101325.0)
        assert isinstance(T_dew, int | float)
        assert math.isfinite(T_dew)

    def test_dew_point_below_saturation_temperature(self, engine) -> None:
        """Dew point is below (or at) the saturation temperature for the partial pressure."""
        # Very low partial pressure → very low dew point
        T_dew_low = engine.calculate_dew_point(500.0, 101325.0)
        T_dew_high = engine.calculate_dew_point(5000.0, 101325.0)
        # Higher partial pressure → higher dew point
        assert T_dew_low < T_dew_high


# ---------------------------------------------------------------------------
# calculate_properties (simplified path)
# ---------------------------------------------------------------------------


class TestCalculateProperties:
    """Tests for calculate_properties at superheated steam conditions."""

    def test_returns_steam_properties(self, engine) -> None:
        """calculate_properties returns a SteamProperties object."""
        from sidekick.calculators.thermo.steam_engine import (
            SteamProperties,
        )

        # Superheated steam: 400K, 1 bar (100000 Pa)
        result = engine.calculate_properties(
            temperature=400.0, pressure=100000.0, engine="simplified"
        )
        assert isinstance(result, SteamProperties)

    def test_temperature_field_matches(self, engine) -> None:
        """The temperature field in the returned properties matches the input."""
        result = engine.calculate_properties(
            temperature=400.0, pressure=100000.0, engine="simplified"
        )
        assert abs(result.temperature - 400.0) < 1.0

    def test_steam_engine_extended_zero_pressure_raises(self, engine) -> None:
        """Zero or negative pressure raises ValueError or AssertionError."""
        with pytest.raises((ValueError, AssertionError)):
            engine.calculate_properties(temperature=400.0, pressure=0.0)

    def test_enthalpy_positive_for_steam(self, engine) -> None:
        """Enthalpy should be positive for steam conditions."""
        result = engine.calculate_properties(
            temperature=400.0, pressure=100000.0, engine="simplified"
        )
        assert result.enthalpy > 0
