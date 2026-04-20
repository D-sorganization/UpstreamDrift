"""Aerodynamics calibration validation tests.

Validates the ISA altitude model (EnvironmentalConditions.from_altitude)
and velocity-dependent spin decay (AerodynamicsEngine.compute_spin_decay).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics.ball_launch_conditions import EnvironmentalConditions


class TestISAAltitudeModel:
    def test_sea_level_density(self):
        ec = EnvironmentalConditions.from_altitude(0.0)
        assert abs(ec.air_density - 1.225) < 0.01

    def test_denver_density(self):
        ec = EnvironmentalConditions.from_altitude(1609.0)
        assert abs(ec.air_density - 1.045) < 0.015

    def test_density_monotone_decrease(self):
        rhos = [
            EnvironmentalConditions.from_altitude(float(h)).air_density
            for h in range(0, 5000, 500)
        ]
        assert all(rhos[i] > rhos[i + 1] for i in range(len(rhos) - 1))

    def test_temperature_decrease_with_altitude(self):
        ec_low = EnvironmentalConditions.from_altitude(0.0)
        ec_high = EnvironmentalConditions.from_altitude(5000.0)
        assert ec_high.temperature < ec_low.temperature

    def test_altitude_stored(self):
        ec = EnvironmentalConditions.from_altitude(2000.0)
        assert ec.altitude == pytest.approx(2000.0)

    def test_negative_altitude_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            EnvironmentalConditions.from_altitude(-1.0)

    def test_wind_velocity_passed_through(self):
        wind = np.array([3.0, 0.0, 0.0])
        ec = EnvironmentalConditions.from_altitude(500.0, wind_velocity=wind)
        np.testing.assert_array_almost_equal(ec.wind_velocity, wind)

    def test_default_wind_is_zero(self):
        ec = EnvironmentalConditions.from_altitude(500.0)
        np.testing.assert_array_equal(ec.wind_velocity, np.zeros(3))


class TestVelocityDependentSpinDecay:
    @pytest.fixture()
    def engine(self):
        from src.shared.python.physics.aerodynamics._config import AerodynamicsConfig
        from src.shared.python.physics.aerodynamics._engine import AerodynamicsEngine

        cfg = AerodynamicsConfig(spin_decay_rate=0.01)
        return AerodynamicsEngine(cfg)

    def test_reference_speed_matches_constant_rate(self, engine):
        spin = np.array([0.0, -100.0, 0.0])
        dt = 0.1
        result_const = engine.compute_spin_decay(spin, dt)
        result_vref = engine.compute_spin_decay(spin, dt, velocity_magnitude=70.0)
        np.testing.assert_array_almost_equal(result_const, result_vref, decimal=6)

    def test_high_speed_decays_faster(self, engine):
        spin = np.array([0.0, -100.0, 0.0])
        dt = 0.1
        result_ref = engine.compute_spin_decay(spin, dt, velocity_magnitude=70.0)
        result_fast = engine.compute_spin_decay(spin, dt, velocity_magnitude=140.0)
        assert np.linalg.norm(result_fast) < np.linalg.norm(result_ref)

    def test_zero_velocity_no_decay(self, engine):
        spin = np.array([0.0, -100.0, 0.0])
        result = engine.compute_spin_decay(spin, 1.0, velocity_magnitude=0.0)
        np.testing.assert_array_almost_equal(result, spin, decimal=6)

    def test_negative_velocity_raises(self, engine):
        spin = np.array([0.0, -100.0, 0.0])
        with pytest.raises(ValueError, match="non-negative"):
            engine.compute_spin_decay(spin, 0.1, velocity_magnitude=-1.0)
