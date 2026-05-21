"""Tests for upstream_drift_tools.process_calculators.thermal_profile_predictor.

Covers predict_temperature_profile and fit_heating_parameters.
"""

from __future__ import annotations

import numpy as np
import pytest
from upstream_drift_tools.process_calculators.thermal_profile_predictor import (
    _heating_ode,
    fit_heating_parameters,
    predict_temperature_profile,
)


def constant_power(t: float) -> float:
    return 1000.0  # W


def zero_power(t: float) -> float:
    return 0.0


class TestHeatingOde:
    def test_heating_with_power(self):
        """With power input, temperature should increase."""
        dTdt = _heating_ode(
            t=0.0,
            y=[25.0],
            thermal_mass=500.0,
            heat_loss_coeff=10.0,
            ambient_temp=20.0,
            power_func=constant_power,
        )
        # dTdt = (1000 - 10*(25-20)) / 500 = (1000 - 50) / 500 = 1.9
        assert dTdt[0] == pytest.approx(1.9, rel=1e-4)

    def test_cooling_without_power(self):
        """Without power, above ambient temp → cooling (dTdt < 0)."""
        dTdt = _heating_ode(
            t=0.0,
            y=[50.0],
            thermal_mass=500.0,
            heat_loss_coeff=10.0,
            ambient_temp=20.0,
            power_func=zero_power,
        )
        assert dTdt[0] < 0

    def test_equilibrium_at_ambient(self):
        """At ambient temp with no power, dTdt ~ 0."""
        dTdt = _heating_ode(
            t=0.0,
            y=[20.0],
            thermal_mass=500.0,
            heat_loss_coeff=10.0,
            ambient_temp=20.0,
            power_func=zero_power,
        )
        assert dTdt[0] == pytest.approx(0.0, abs=1e-10)


class TestPredictTemperatureProfile:
    def test_returns_time_and_temp_arrays(self):
        t_eval = np.linspace(0, 100, 20)
        t_arr, temp_arr = predict_temperature_profile(
            t_span=(0.0, 100.0),
            t_eval=t_eval,
            initial_temp=20.0,
            thermal_mass=500.0,
            heat_loss_coeff=5.0,
            ambient_temp=20.0,
            power_func=constant_power,
        )
        assert len(t_arr) == 20
        assert len(temp_arr) == 20

    def test_temperature_increases_with_power(self):
        t_eval = np.linspace(0, 100, 10)
        _, temp_arr = predict_temperature_profile(
            t_span=(0.0, 100.0),
            t_eval=t_eval,
            initial_temp=20.0,
            thermal_mass=500.0,
            heat_loss_coeff=1.0,  # very low loss
            ambient_temp=20.0,
            power_func=constant_power,
        )
        # Final temp should be higher than initial
        assert temp_arr[-1] > temp_arr[0]

    def test_temperature_constant_at_equilibrium(self):
        """With no power and initial temp at ambient, temp stays constant."""
        t_eval = np.linspace(0, 50, 10)
        _, temp_arr = predict_temperature_profile(
            t_span=(0.0, 50.0),
            t_eval=t_eval,
            initial_temp=20.0,
            thermal_mass=500.0,
            heat_loss_coeff=5.0,
            ambient_temp=20.0,
            power_func=zero_power,
        )
        np.testing.assert_array_almost_equal(temp_arr, 20.0, decimal=6)

    def test_initial_temp_is_first_value(self):
        t_eval = np.linspace(0, 10, 5)
        _, temp_arr = predict_temperature_profile(
            t_span=(0.0, 10.0),
            t_eval=t_eval,
            initial_temp=35.0,
            thermal_mass=500.0,
            heat_loss_coeff=5.0,
            ambient_temp=20.0,
            power_func=zero_power,
        )
        assert temp_arr[0] == pytest.approx(35.0, rel=1e-4)


class TestFitHeatingParameters:
    def test_fit_recovers_known_parameters(self):
        """Fit should recover approximately the known thermal parameters."""
        # Generate synthetic data with known parameters
        true_mass = 500.0
        true_loss = 5.0
        t_eval = np.linspace(1, 200, 30)
        _, observed = predict_temperature_profile(
            t_span=(1.0, 200.0),
            t_eval=t_eval,
            initial_temp=20.0,
            thermal_mass=true_mass,
            heat_loss_coeff=true_loss,
            ambient_temp=20.0,
            power_func=constant_power,
        )

        # Add tiny noise so fit is not degenerate
        observed_noisy = observed + np.random.default_rng(42).normal(
            0, 0.01, len(observed)
        )

        fitted_mass, fitted_loss = fit_heating_parameters(
            times=t_eval,
            observed_temps=observed_noisy,
            initial_temp=20.0,
            thermal_mass_guess=600.0,
            heat_loss_guess=4.0,
            ambient_temp=20.0,
            power_func=constant_power,
        )
        # Should be within 5% of true values
        assert abs(fitted_mass - true_mass) / true_mass < 0.05
        assert abs(fitted_loss - true_loss) / true_loss < 0.05
