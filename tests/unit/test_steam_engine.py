"""Tests for SteamCalculationEngine using the simplified (fallback) engine.

These tests exercise the pure-Python simplified calculation path which
does not require CoolProp or Cantera, so they run in all environments.
"""

from __future__ import annotations

import pytest


class TestSteamCalculationEngineImport:
    """Tests that the module and class are importable."""

    def test_steam_engine_importable(self) -> None:
        """SteamCalculationEngine should be importable."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamCalculationEngine,
        )

        assert SteamCalculationEngine is not None

    def test_steam_properties_importable(self) -> None:
        """SteamProperties dataclass should be importable."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamProperties,
        )

        assert SteamProperties is not None

    def test_constants_importable(self) -> None:
        """Module-level constants should be importable and sane."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            CRITICAL_PRESSURE_WATER,
            CRITICAL_TEMPERATURE_WATER,
            STANDARD_ATMOSPHERIC_PRESSURE,
        )

        assert pytest.approx(101325.0) == STANDARD_ATMOSPHERIC_PRESSURE
        assert pytest.approx(647.15) == CRITICAL_TEMPERATURE_WATER
        assert CRITICAL_PRESSURE_WATER > 22e6  # ~22 MPa


class TestSteamEngineInit:
    """Tests for SteamCalculationEngine initialisation."""

    def test_engine_creates_without_error(self) -> None:
        """Engine should instantiate regardless of optional dependencies."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamCalculationEngine,
        )

        engine = SteamCalculationEngine()
        assert engine is not None

    def test_engine_select_simplified_when_no_optional_deps(self) -> None:
        """With no optional libs, auto-selection should fall back to 'simplified'."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            CANTERA_AVAILABLE,
            COOLPROP_AVAILABLE,
            SteamCalculationEngine,
        )

        engine = SteamCalculationEngine()
        selected = engine._select_best_engine("auto")
        if not COOLPROP_AVAILABLE and not CANTERA_AVAILABLE:
            assert selected == "simplified"
        else:
            # If optional libs present, should not be simplified
            assert selected in ("coolprop", "cantera", "simplified")

    def test_engine_accepts_simplified_engine_request(self) -> None:
        """Explicitly requesting 'simplified' should always return 'simplified'."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamCalculationEngine,
        )

        engine = SteamCalculationEngine()
        assert engine._select_best_engine("simplified") == "simplified"


class TestSimplifiedSteamProperties:
    """Tests for the simplified steam property calculations."""

    @pytest.fixture
    def engine(self):
        """Create a SteamCalculationEngine."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamCalculationEngine,
        )

        return SteamCalculationEngine()

    def test_calculate_properties_returns_steam_properties(self, engine) -> None:
        """calculate_properties should return a SteamProperties instance."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamProperties,
        )

        result = engine.calculate_properties(
            temperature=400.0,  # 127°C, superheated steam
            pressure=101325.0,  # 1 atm
            engine="simplified",
        )
        assert isinstance(result, SteamProperties)

    def test_temperature_preserved(self, engine) -> None:
        """The returned temperature should match the input."""
        result = engine.calculate_properties(
            temperature=400.0, pressure=101325.0, engine="simplified"
        )
        assert result.temperature == pytest.approx(400.0)

    def test_pressure_preserved(self, engine) -> None:
        """The returned pressure should match the input."""
        result = engine.calculate_properties(
            temperature=400.0, pressure=200000.0, engine="simplified"
        )
        assert result.pressure == pytest.approx(200000.0)

    def test_enthalpy_is_finite(self, engine) -> None:
        """Enthalpy should always be a finite number."""
        import math

        result = engine.calculate_properties(
            temperature=373.15, pressure=101325.0, engine="simplified"
        )
        assert math.isfinite(result.enthalpy)

    def test_density_positive(self, engine) -> None:
        """Density should be positive."""
        result = engine.calculate_properties(
            temperature=400.0, pressure=101325.0, engine="simplified"
        )
        assert result.density > 0.0

    def test_specific_volume_positive(self, engine) -> None:
        """Specific volume should be positive."""
        result = engine.calculate_properties(
            temperature=400.0, pressure=101325.0, engine="simplified"
        )
        assert result.specific_volume > 0.0

    def test_precondition_negative_temperature_raises(self, engine) -> None:
        """Negative temperature (K) should raise AssertionError."""
        with pytest.raises(AssertionError):
            engine.calculate_properties(temperature=-10.0, pressure=101325.0)

    def test_precondition_zero_pressure_raises(self, engine) -> None:
        """Zero pressure should raise AssertionError."""
        with pytest.raises(AssertionError):
            engine.calculate_properties(temperature=400.0, pressure=0.0)

    def test_to_dict_returns_dict(self, engine) -> None:
        """to_dict() should return a non-empty dict."""
        result = engine.calculate_properties(
            temperature=400.0, pressure=101325.0, engine="simplified"
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert len(d) > 0
        assert "Temperature (K)" in d


class TestWaterVaporPressure:
    """Tests for water vapor pressure calculation methods."""

    @pytest.fixture
    def engine(self):
        """Create engine."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamCalculationEngine,
        )

        return SteamCalculationEngine()

    def test_buck_equation_at_0c(self, engine) -> None:
        """At 0°C, Buck equation vapor pressure should be ~611 Pa (triple point)."""
        p = engine._buck_equation(0.0)
        # At 0°C, saturation pressure is ~611 Pa
        assert p == pytest.approx(611.21, rel=0.01)

    def test_buck_equation_increases_with_temperature(self, engine) -> None:
        """Buck equation vapor pressure should increase monotonically with temperature."""
        p_0c = engine._buck_equation(0.0)
        p_50c = engine._buck_equation(50.0)
        p_100c = engine._buck_equation(100.0)
        assert p_0c < p_50c < p_100c

    def test_antoine_equation_at_100c(self, engine) -> None:
        """Antoine equation at 100°C should give ~101325 Pa (1 atm)."""
        p = engine._antoine_equation(100.0)
        # Antoine equation is calibrated to give ~1 atm at 100°C
        assert p == pytest.approx(101325.0, rel=0.01)

    def test_vapor_pressure_increases_with_temperature(self, engine) -> None:
        """Vapor pressure should increase with temperature."""
        p_cold = engine.calculate_water_vapor_pressure(20.0, method="buck")
        p_hot = engine.calculate_water_vapor_pressure(80.0, method="buck")
        assert p_hot > p_cold

    def test_vapor_pressure_positive(self, engine) -> None:
        """Vapor pressure should always be positive for positive temperatures."""
        for T in [10.0, 25.0, 50.0, 80.0]:
            p = engine.calculate_water_vapor_pressure(T, method="buck")
            assert p > 0.0, f"Vapor pressure should be positive at {T}°C"


class TestSteamPropertiesDataclass:
    """Tests for SteamProperties dataclass."""

    def test_create_steam_properties(self) -> None:
        """Should create a SteamProperties instance with all fields."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamProperties,
        )

        props = SteamProperties(
            temperature=400.0,
            pressure=101325.0,
            density=0.6,
            specific_volume=1.667,
            enthalpy=2.7e6,
            entropy=7500.0,
            internal_energy=2.5e6,
            cp=2000.0,
            cv=1500.0,
            speed_of_sound=470.0,
            thermal_conductivity=0.025,
            dynamic_viscosity=1.2e-5,
            kinematic_viscosity=2.0e-5,
            quality=1.0,
            phase="vapor",
        )
        assert props.temperature == pytest.approx(400.0)
        assert props.phase == "vapor"

    def test_optional_fields_default_none(self) -> None:
        """Optional fields should default to None."""
        from src.shared.python.upstream_drift_tools.calculators.thermo.steam_engine import (
            SteamProperties,
        )

        props = SteamProperties(
            temperature=400.0,
            pressure=101325.0,
            density=0.6,
            specific_volume=1.667,
            enthalpy=2.7e6,
            entropy=7500.0,
            internal_energy=2.5e6,
            cp=2000.0,
            cv=1500.0,
            speed_of_sound=470.0,
            thermal_conductivity=0.025,
            dynamic_viscosity=1.2e-5,
            kinematic_viscosity=2.0e-5,
            quality=1.0,
            phase="vapor",
        )
        assert props.compressibility_factor is None
        assert props.prandtl_number is None
        assert props.specific_heat_ratio is None
