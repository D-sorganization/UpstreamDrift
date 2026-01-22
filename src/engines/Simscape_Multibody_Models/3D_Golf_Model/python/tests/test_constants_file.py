"""Tests for constants_file module."""

import math
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import constants after adding to path
# Use try/except to handle different path configurations
try:
    from src.constants import (
        AIR_DENSITY_SEA_LEVEL_KG_M3,
        BUNKER_DEPTH_MM,
        DRIVER_LOFT_DEG,
        GOLF_BALL_DIAMETER_M,
        GOLF_BALL_DRAG_COEFFICIENT,
        GOLF_BALL_MASS_KG,
        GRAVITY_M_S2,
        GREEN_SPEED_STIMP,
        HUMIDITY_PERCENT,
        IRON_7_LOFT_DEG,
        PI,
        PRESSURE_HPA,
        PUTTER_LOFT_DEG,
        ROUGH_HEIGHT_MM,
        SPEED_OF_LIGHT_M_S,
        TEMPERATURE_C,
        E,
    )
except ImportError:
    # Fallback for when python/src is directly in path (e.g. via conftest.py)
    from constants import (
        AIR_DENSITY_SEA_LEVEL_KG_M3,
        BUNKER_DEPTH_MM,
        DRIVER_LOFT_DEG,
        GOLF_BALL_DIAMETER_M,
        GOLF_BALL_DRAG_COEFFICIENT,
        GOLF_BALL_MASS_KG,
        GRAVITY_M_S2,
        GREEN_SPEED_STIMP,
        HUMIDITY_PERCENT,
        IRON_7_LOFT_DEG,
        PI,
        PRESSURE_HPA,
        PUTTER_LOFT_DEG,
        ROUGH_HEIGHT_MM,
        SPEED_OF_LIGHT_M_S,
        TEMPERATURE_C,
        E,
    )


class TestMathematicalConstants:
    """Tests for mathematical constants."""

    def test_pi_value(self) -> None:
        """Test that PI constant equals math.pi."""
        assert PI == math.pi
        # Verify it's the expected mathematical constant
        assert PI > 3.14 and PI < 3.15

    def test_e_value(self) -> None:
        """Test that E constant equals Euler's number."""
        assert E == math.e
        assert E == pytest.approx(2.71828182845904523536)


class TestPhysicalConstants:
    """Tests for physical constants."""

    def test_gravity_value(self) -> None:
        """Test standard gravity constant."""
        # ISO 80000-3:2006 standard gravity
        assert GRAVITY_M_S2 == 9.80665
        assert GRAVITY_M_S2 > 0
        # Verify reasonable range for Earth's gravity
        assert 9.5 < GRAVITY_M_S2 < 10.0

    def test_speed_of_light_value(self) -> None:
        """Test speed of light constant."""
        # Exact by SI definition
        assert SPEED_OF_LIGHT_M_S == 299792458
        assert isinstance(SPEED_OF_LIGHT_M_S, int | float)

    def test_air_density_value(self) -> None:
        """Test air density at sea level."""
        # ISA standard at 15°C
        assert AIR_DENSITY_SEA_LEVEL_KG_M3 == 1.225
        assert AIR_DENSITY_SEA_LEVEL_KG_M3 > 0
        assert 1.0 < AIR_DENSITY_SEA_LEVEL_KG_M3 < 1.5


class TestGolfBallConstants:
    """Tests for golf ball physical properties."""

    def test_golf_ball_mass(self) -> None:
        """Test golf ball mass constant."""
        # USGA Rule 5-1: max 1.620 oz = 0.04593 kg
        assert GOLF_BALL_MASS_KG == 0.04593
        assert GOLF_BALL_MASS_KG > 0
        # Should be less than maximum allowed
        assert GOLF_BALL_MASS_KG <= 0.04593
        # Convert to ounces and check
        mass_oz = GOLF_BALL_MASS_KG / 0.0283495
        assert mass_oz == pytest.approx(1.620, rel=1e-3)

    def test_golf_ball_diameter(self) -> None:
        """Test golf ball diameter constant."""
        # USGA Rule 5-2: min 1.680 inches = 0.04267 m
        assert GOLF_BALL_DIAMETER_M == 0.04267
        assert GOLF_BALL_DIAMETER_M > 0
        # Should be at least minimum allowed
        assert GOLF_BALL_DIAMETER_M >= 0.04267
        # Convert to inches and check
        diameter_inches = GOLF_BALL_DIAMETER_M / 0.0254
        assert diameter_inches == pytest.approx(1.680, rel=1e-3)

    def test_golf_ball_drag_coefficient(self) -> None:
        """Test golf ball drag coefficient."""
        # Typical value for smooth ball at Re~150,000
        assert GOLF_BALL_DRAG_COEFFICIENT == 0.25
        assert GOLF_BALL_DRAG_COEFFICIENT > 0
        # Should be reasonable range for spherical objects
        assert 0.1 < GOLF_BALL_DRAG_COEFFICIENT < 1.0


class TestClubSpecifications:
    """Tests for golf club specifications."""

    def test_driver_loft(self) -> None:
        """Test driver loft angle."""
        assert DRIVER_LOFT_DEG == 10.5
        assert DRIVER_LOFT_DEG > 0
        # Typical driver loft range: 8-12 degrees
        assert 8.0 <= DRIVER_LOFT_DEG <= 12.0

    def test_iron_7_loft(self) -> None:
        """Test 7-iron loft angle."""
        assert IRON_7_LOFT_DEG == 34.0
        assert IRON_7_LOFT_DEG > 0
        # Typical 7-iron loft range: 30-36 degrees
        assert 30.0 <= IRON_7_LOFT_DEG <= 36.0

    def test_putter_loft(self) -> None:
        """Test putter loft angle."""
        assert PUTTER_LOFT_DEG == 3.0
        assert PUTTER_LOFT_DEG > 0
        # Typical putter loft range: 2-5 degrees
        assert 2.0 <= PUTTER_LOFT_DEG <= 5.0

    def test_club_loft_progression(self) -> None:
        """Test that club lofts are in correct order."""
        # Driver should have least loft, 7-iron more, putter minimal
        assert DRIVER_LOFT_DEG < IRON_7_LOFT_DEG
        assert PUTTER_LOFT_DEG < DRIVER_LOFT_DEG


class TestCourseConditions:
    """Tests for golf course conditions."""

    def test_green_speed(self) -> None:
        """Test green speed stimpmeter reading."""
        assert GREEN_SPEED_STIMP == 10.0
        assert GREEN_SPEED_STIMP > 0
        # Fast greens typically 9-13 feet
        assert 9.0 <= GREEN_SPEED_STIMP <= 13.0

    def test_rough_height(self) -> None:
        """Test rough height."""
        assert ROUGH_HEIGHT_MM == 25.0
        assert ROUGH_HEIGHT_MM > 0
        # Medium rough typically 20-40mm
        assert 20.0 <= ROUGH_HEIGHT_MM <= 40.0

    def test_bunker_depth(self) -> None:
        """Test bunker sand depth."""
        assert BUNKER_DEPTH_MM == 100.0
        assert BUNKER_DEPTH_MM > 0
        # Typical bunker depth 50-150mm
        assert 50.0 <= BUNKER_DEPTH_MM <= 150.0


class TestAtmosphericConditions:
    """Tests for atmospheric conditions."""

    def test_temperature(self) -> None:
        """Test standard temperature."""
        assert TEMPERATURE_C == 20.0
        # Should be reasonable room temperature
        assert 15.0 <= TEMPERATURE_C <= 25.0

    def test_pressure(self) -> None:
        """Test standard atmospheric pressure."""
        assert PRESSURE_HPA == 1013.25
        # Standard atmospheric pressure at sea level
        assert 1010.0 <= PRESSURE_HPA <= 1020.0

    def test_humidity(self) -> None:
        """Test standard humidity."""
        assert HUMIDITY_PERCENT == 50.0
        # Should be valid percentage
        assert 0.0 <= HUMIDITY_PERCENT <= 100.0


class TestConstantTypes:
    """Tests for constant data types."""

    def test_all_constants_are_numeric(self) -> None:
        """Test that all constants are numeric types."""
        constants = [
            PI,
            E,
            GRAVITY_M_S2,
            SPEED_OF_LIGHT_M_S,
            AIR_DENSITY_SEA_LEVEL_KG_M3,
            GOLF_BALL_MASS_KG,
            GOLF_BALL_DIAMETER_M,
            GOLF_BALL_DRAG_COEFFICIENT,
            DRIVER_LOFT_DEG,
            IRON_7_LOFT_DEG,
            PUTTER_LOFT_DEG,
            GREEN_SPEED_STIMP,
            ROUGH_HEIGHT_MM,
            BUNKER_DEPTH_MM,
            TEMPERATURE_C,
            PRESSURE_HPA,
            HUMIDITY_PERCENT,
        ]
        for constant in constants:
            assert isinstance(constant, int | float)

    def test_all_constants_are_finite(self) -> None:
        """Test that all constants are finite (not inf or nan)."""
        constants = [
            PI,
            E,
            GRAVITY_M_S2,
            SPEED_OF_LIGHT_M_S,
            AIR_DENSITY_SEA_LEVEL_KG_M3,
            GOLF_BALL_MASS_KG,
            GOLF_BALL_DIAMETER_M,
            GOLF_BALL_DRAG_COEFFICIENT,
            DRIVER_LOFT_DEG,
            IRON_7_LOFT_DEG,
            PUTTER_LOFT_DEG,
            GREEN_SPEED_STIMP,
            ROUGH_HEIGHT_MM,
            BUNKER_DEPTH_MM,
            TEMPERATURE_C,
            PRESSURE_HPA,
            HUMIDITY_PERCENT,
        ]
        for constant in constants:
            assert math.isfinite(constant)
