"""Comprehensive tests for src.shared.python.physics package.

Covers equipment (CLUB_CONFIGS, get_club_config), flight_model_options
(FlightModelOptions, compute_spin_decay, compute_air_density_at_altitude),
energy_monitor (EnergySnapshot, ConservationMonitor — standalone parts),
and physics_validation dataclasses.
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.physics.energy_monitor import (
    ENERGY_DRIFT_CRITICAL_PCT,
    ENERGY_DRIFT_TOLERANCE_PCT,
    EnergySnapshot,
    IntegrationFailureError,
)
from src.shared.python.physics.equipment import (
    CLUB_CONFIGS,
    get_club_config,
)
from src.shared.python.physics.flight_model_options import (
    DEFAULT_OPTIONS,
    FlightModelOptions,
    compute_air_density_at_altitude,
    compute_spin_decay,
)
from src.shared.python.physics.physics_validation import (
    EnergyValidationResult,
    JacobianValidationResult,
)

# ============================================================================
# Tests for equipment module
# ============================================================================


class TestEquipment:
    """Tests for golf equipment specifications."""

    def test_club_configs_keys(self) -> None:
        assert "driver" in CLUB_CONFIGS
        assert "iron_7" in CLUB_CONFIGS
        assert "wedge" in CLUB_CONFIGS

    def test_get_club_config_driver(self) -> None:
        config = get_club_config("driver")
        assert isinstance(config, dict)
        assert "head_mass" in config
        assert "shaft_length" in config
        assert "club_loft" in config
        assert config["head_mass"] > 0

    def test_get_club_config_all_types(self) -> None:
        for club_type in ["driver", "iron_7", "wedge"]:
            config = get_club_config(club_type)
            assert config["head_mass"] > 0
            assert config["shaft_length"] > 0
            assert config["total_length"] > 0

    def test_get_club_config_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid club_type"):
            get_club_config("putter")

    def test_club_physical_reasonability(self) -> None:
        """Club configs should have physically reasonable values."""
        for ctype in CLUB_CONFIGS:
            cfg = CLUB_CONFIGS[ctype]
            assert 0.05 < float(cfg["shaft_length"]) < 2.0  # type: ignore[arg-type]
            assert 0 < float(cfg["head_mass"]) < 1.0  # type: ignore[arg-type]

    def test_driver_longer_than_wedge(self) -> None:
        driver = get_club_config("driver")
        wedge = get_club_config("wedge")
        assert driver["total_length"] > wedge["total_length"]

    def test_wedge_heavier_head_than_driver(self) -> None:
        driver = get_club_config("driver")
        wedge = get_club_config("wedge")
        assert wedge["head_mass"] > driver["head_mass"]

    def test_wedge_higher_loft_than_driver(self) -> None:
        driver = get_club_config("driver")
        wedge = get_club_config("wedge")
        assert wedge["club_loft"] > driver["club_loft"]


# ============================================================================
# Tests for FlightModelOptions
# ============================================================================


class TestFlightModelOptions:
    """Tests for flight model configuration."""

    def test_defaults_off(self) -> None:
        opts = FlightModelOptions()
        assert not opts.enable_wind
        assert not opts.enable_spin_decay
        assert not opts.enable_altitude_correction
        assert opts.altitude_m == 0.0

    def test_default_options_global(self) -> None:
        assert not DEFAULT_OPTIONS.enable_wind
        assert not DEFAULT_OPTIONS.enable_spin_decay

    def test_custom_options(self) -> None:
        opts = FlightModelOptions(
            enable_wind=True,
            enable_spin_decay=True,
            spin_decay_rate=0.1,
            enable_altitude_correction=True,
            altitude_m=1500.0,
        )
        assert opts.enable_wind
        assert opts.spin_decay_rate == 0.1
        assert opts.altitude_m == 1500.0


class TestComputeSpinDecay:
    """Tests for exponential spin decay function."""

    def test_no_decay_at_time_zero(self) -> None:
        omega = compute_spin_decay(300.0, time=0.0, decay_rate=0.05)
        assert omega == pytest.approx(300.0)

    def test_decays_over_time(self) -> None:
        omega_0 = compute_spin_decay(300.0, time=0.0, decay_rate=0.05)
        omega_1 = compute_spin_decay(300.0, time=5.0, decay_rate=0.05)
        omega_2 = compute_spin_decay(300.0, time=10.0, decay_rate=0.05)
        assert omega_1 < omega_0
        assert omega_2 < omega_1

    def test_exponential_decay_formula(self) -> None:
        """ω(t) = ω₀ * exp(-λ*t)."""
        omega = compute_spin_decay(200.0, time=2.0, decay_rate=0.1)
        expected = 200.0 * math.exp(-0.1 * 2.0)
        assert omega == pytest.approx(expected)

    def test_zero_initial_spin(self) -> None:
        omega = compute_spin_decay(0.0, time=5.0, decay_rate=0.05)
        assert omega == 0.0

    def test_zero_decay_rate(self) -> None:
        """With zero decay rate, spin is constant."""
        omega = compute_spin_decay(300.0, time=100.0, decay_rate=0.0)
        assert omega == pytest.approx(300.0)

    def test_always_nonnegative(self) -> None:
        for t in [0, 1, 10, 100, 1000]:
            omega = compute_spin_decay(300.0, time=float(t), decay_rate=0.05)
            assert omega >= 0


class TestComputeAirDensityAtAltitude:
    """Tests for barometric altitude correction."""

    def test_sea_level_unchanged(self) -> None:
        rho = compute_air_density_at_altitude(1.225, altitude_m=0.0)
        assert rho == pytest.approx(1.225)

    def test_decreases_with_altitude(self) -> None:
        rho_0 = compute_air_density_at_altitude(1.225, altitude_m=0.0)
        rho_1000 = compute_air_density_at_altitude(1.225, altitude_m=1000.0)
        rho_3000 = compute_air_density_at_altitude(1.225, altitude_m=3000.0)
        assert rho_1000 < rho_0
        assert rho_3000 < rho_1000

    def test_barometric_formula(self) -> None:
        """ρ(h) = ρ₀ * exp(-h / H) where H ≈ 8500m."""
        rho = compute_air_density_at_altitude(1.225, altitude_m=1000.0)
        expected = 1.225 * math.exp(-1000.0 / 8500.0)
        assert rho == pytest.approx(expected)

    def test_always_positive(self) -> None:
        for h in [0, 500, 1000, 5000, 10000]:
            rho = compute_air_density_at_altitude(1.225, altitude_m=float(h))
            assert rho > 0

    def test_high_altitude_realistic(self) -> None:
        """At 5000m (Denver-like), density should be ~60% of sea level."""
        rho = compute_air_density_at_altitude(1.225, altitude_m=5000.0)
        ratio = rho / 1.225
        assert 0.4 < ratio < 0.7


# ============================================================================
# Tests for EnergySnapshot
# ============================================================================


class TestEnergySnapshot:
    """Tests for EnergySnapshot dataclass."""

    def test_total_energy(self) -> None:
        snap = EnergySnapshot(time=0.5, kinetic=100.0, potential=50.0)
        assert snap.total == pytest.approx(150.0)

    def test_zero_energy(self) -> None:
        snap = EnergySnapshot(time=0.0, kinetic=0.0, potential=0.0)
        assert snap.total == 0.0

    def test_attributes(self) -> None:
        snap = EnergySnapshot(time=1.0, kinetic=200.0, potential=100.0)
        assert snap.time == 1.0
        assert snap.kinetic == 200.0
        assert snap.potential == 100.0


class TestEnergyConstants:
    """Tests for energy monitoring constants."""

    def test_tolerance_values(self) -> None:
        assert ENERGY_DRIFT_TOLERANCE_PCT == 1.0
        assert ENERGY_DRIFT_CRITICAL_PCT == 5.0
        assert ENERGY_DRIFT_CRITICAL_PCT > ENERGY_DRIFT_TOLERANCE_PCT


class TestIntegrationFailureError:
    """Tests for IntegrationFailureError exception."""

    def test_is_exception(self) -> None:
        err = IntegrationFailureError("Energy blew up")
        assert isinstance(err, Exception)
        assert "Energy blew up" in str(err)


# ============================================================================
# Tests for physics_validation dataclasses
# ============================================================================


class TestEnergyValidationResult:
    """Tests for EnergyValidationResult."""

    def test_passing_result(self) -> None:
        r = EnergyValidationResult(
            energy_error=0.001,
            relative_error=0.0001,
            passes=True,
            kinetic_energy_initial=100.0,
            kinetic_energy_final=99.9,
            potential_energy_initial=50.0,
            potential_energy_final=50.1,
            work_applied=0.0,
            message="OK",
        )
        assert r.passes
        assert r.relative_error < 0.001

    def test_failing_result(self) -> None:
        r = EnergyValidationResult(
            energy_error=10.0,
            relative_error=0.1,
            passes=False,
            kinetic_energy_initial=100.0,
            kinetic_energy_final=110.0,
            potential_energy_initial=50.0,
            potential_energy_final=50.0,
            work_applied=0.0,
            message="FAIL: too much drift",
        )
        assert not r.passes

    def test_str_representation(self) -> None:
        r = EnergyValidationResult(
            energy_error=0.01,
            relative_error=0.001,
            passes=True,
            kinetic_energy_initial=100.0,
            kinetic_energy_final=100.0,
            potential_energy_initial=50.0,
            potential_energy_final=50.0,
            work_applied=0.0,
            message="OK",
        )
        assert len(str(r)) > 0


class TestJacobianValidationResult:
    """Tests for JacobianValidationResult."""

    def test_passing(self) -> None:
        r = JacobianValidationResult(
            jacobian_error=1e-8,
            passes=True,
            body_id=1,
            message="OK",
        )
        assert r.passes
        assert r.body_id == 1

    def test_str(self) -> None:
        r = JacobianValidationResult(
            jacobian_error=1e-3,
            passes=False,
            body_id=2,
            message="Jacobian mismatch",
        )
        assert "Jacobian" in str(r) or len(str(r)) > 0
