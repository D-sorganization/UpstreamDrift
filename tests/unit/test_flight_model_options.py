"""Unit tests for flight model options and configuration."""

import math

import pytest

from src.shared.python.flight_model_options import (
    DEFAULT_OPTIONS,
    FlightModelOptions,
    compute_air_density_at_altitude,
    compute_spin_decay,
)


class TestFlightModelOptions:
    """Tests for FlightModelOptions dataclass."""

    def test_defaults(self):
        """Test default values."""
        options = FlightModelOptions()
        assert options.enable_wind is False
        assert options.enable_spin_decay is False
        assert options.spin_decay_rate == 0.05
        assert options.enable_altitude_correction is False
        assert options.altitude_m == 0.0

    def test_custom_values(self):
        """Test custom initialization."""
        options = FlightModelOptions(
            enable_wind=True,
            enable_spin_decay=True,
            spin_decay_rate=0.1,
            enable_altitude_correction=True,
            altitude_m=1000.0,
        )
        assert options.enable_wind is True
        assert options.enable_spin_decay is True
        assert options.spin_decay_rate == 0.1
        assert options.enable_altitude_correction is True
        assert options.altitude_m == 1000.0

    def test_default_instance(self):
        """Test global default instance."""
        assert isinstance(DEFAULT_OPTIONS, FlightModelOptions)
        assert DEFAULT_OPTIONS.enable_wind is False


class TestComputeSpinDecay:
    """Tests for spin decay computation."""

    def test_no_decay_at_t0(self):
        """At t=0, spin should remain initial."""
        omega_0 = 100.0
        omega = compute_spin_decay(omega_0, time=0.0, decay_rate=0.05)
        assert omega == omega_0

    def test_decay_over_time(self):
        """Test decay calculation."""
        omega_0 = 100.0
        decay_rate = 0.05
        time = 10.0

        # Expected: 100 * exp(-0.05 * 10) = 100 * exp(-0.5)
        expected = 100.0 * math.exp(-0.5)
        omega = compute_spin_decay(omega_0, time, decay_rate)

        assert omega == pytest.approx(expected)
        assert omega < omega_0

    def test_zero_decay_rate(self):
        """Test with zero decay rate."""
        omega_0 = 100.0
        omega = compute_spin_decay(omega_0, time=10.0, decay_rate=0.0)
        assert omega == omega_0

    def test_zero_initial_spin(self):
        """Test with zero initial spin."""
        omega = compute_spin_decay(0.0, time=10.0, decay_rate=0.05)
        assert omega == 0.0


class TestComputeAirDensity:
    """Tests for air density altitude correction."""

    def test_sea_level(self):
        """At 0m, density should be sea level density."""
        rho_0 = 1.225
        rho = compute_air_density_at_altitude(rho_0, 0.0)
        assert rho == rho_0

    def test_at_altitude(self):
        """Test density at altitude."""
        rho_0 = 1.225
        altitude = 1000.0
        scale_height = 8500.0

        expected = rho_0 * math.exp(-altitude / scale_height)
        rho = compute_air_density_at_altitude(rho_0, altitude)

        assert rho == pytest.approx(expected)
        assert rho < rho_0

    def test_high_altitude(self):
        """Test at high altitude (Mount Everest ~8848m)."""
        rho_0 = 1.225
        rho = compute_air_density_at_altitude(rho_0, 8848.0)
        # Density should be roughly 1/e of sea level since 8848 ~ 8500
        assert rho < rho_0 * 0.4
        assert rho > 0.0
