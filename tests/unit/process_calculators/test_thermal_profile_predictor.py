"""Tests for sidekick.process_calculators.thermal_profile_predictor (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from sidekick.process_calculators.thermal_profile_predictor import (
    fit_heating_parameters,
    predict_temperature_profile,
)


def _const_power(t: float) -> float:
    return 1000.0  # 1 kW constant power


def _zero_power(t: float) -> float:
    return 0.0


class TestPredictTemperatureProfile:
    def test_thermal_profile_predictor_returns_two_arrays(self) -> None:
        t_eval = np.linspace(0, 100, 11)
        times, temps = predict_temperature_profile(
            t_span=(0.0, 100.0),
            t_eval=t_eval,
            initial_temp=20.0,
            thermal_mass=5000.0,
            heat_loss_coeff=10.0,
            ambient_temp=20.0,
            power_func=_const_power,
        )
        assert isinstance(times, np.ndarray)
        assert isinstance(temps, np.ndarray)

    def test_output_length_matches_eval_points(self) -> None:
        t_eval = np.linspace(0, 100, 11)
        times, temps = predict_temperature_profile(
            t_span=(0.0, 100.0),
            t_eval=t_eval,
            initial_temp=20.0,
            thermal_mass=5000.0,
            heat_loss_coeff=10.0,
            ambient_temp=20.0,
            power_func=_const_power,
        )
        assert len(times) == len(t_eval)
        assert len(temps) == len(t_eval)

    def test_initial_temperature_at_t0(self) -> None:
        t_eval = np.linspace(0, 100, 11)
        _, temps = predict_temperature_profile(
            t_span=(0.0, 100.0),
            t_eval=t_eval,
            initial_temp=20.0,
            thermal_mass=5000.0,
            heat_loss_coeff=10.0,
            ambient_temp=20.0,
            power_func=_const_power,
        )
        assert temps[0] == pytest.approx(20.0, abs=0.5)

    def test_temp_rises_with_constant_power(self) -> None:
        t_eval = np.linspace(0, 3600, 50)
        _, temps = predict_temperature_profile(
            t_span=(0.0, 3600.0),
            t_eval=t_eval,
            initial_temp=20.0,
            thermal_mass=5000.0,
            heat_loss_coeff=1.0,
            ambient_temp=20.0,
            power_func=_const_power,
        )
        assert temps[-1] > temps[0]

    def test_no_power_stays_at_ambient(self) -> None:
        t_eval = np.linspace(0, 100, 11)
        _, temps = predict_temperature_profile(
            t_span=(0.0, 100.0),
            t_eval=t_eval,
            initial_temp=20.0,
            thermal_mass=5000.0,
            heat_loss_coeff=10.0,
            ambient_temp=20.0,
            power_func=_zero_power,
        )
        # No power and no temperature gradient → constant at ambient
        assert all(abs(t - 20.0) < 0.1 for t in temps)


class TestFitHeatingParameters:
    def test_returns_two_floats(self) -> None:
        # Generate synthetic data from known parameters
        thermal_mass_true = 5000.0
        heat_loss_true = 10.0
        times = np.linspace(0, 500, 20)
        _, obs = predict_temperature_profile(
            t_span=(0.0, 500.0),
            t_eval=times,
            initial_temp=20.0,
            thermal_mass=thermal_mass_true,
            heat_loss_coeff=heat_loss_true,
            ambient_temp=20.0,
            power_func=_const_power,
        )
        tm, hl = fit_heating_parameters(
            times=times,
            observed_temps=obs,
            initial_temp=20.0,
            thermal_mass_guess=4000.0,
            heat_loss_guess=8.0,
            ambient_temp=20.0,
            power_func=_const_power,
        )
        assert isinstance(tm, float)
        assert isinstance(hl, float)

    def test_recovers_approximate_parameters(self) -> None:
        thermal_mass_true = 5000.0
        heat_loss_true = 10.0
        times = np.linspace(0, 500, 20)
        _, obs = predict_temperature_profile(
            t_span=(0.0, 500.0),
            t_eval=times,
            initial_temp=20.0,
            thermal_mass=thermal_mass_true,
            heat_loss_coeff=heat_loss_true,
            ambient_temp=20.0,
            power_func=_const_power,
        )
        tm, hl = fit_heating_parameters(
            times=times,
            observed_temps=obs,
            initial_temp=20.0,
            thermal_mass_guess=4500.0,
            heat_loss_guess=9.0,
            ambient_temp=20.0,
            power_func=_const_power,
        )
        assert tm == pytest.approx(thermal_mass_true, rel=0.1)
        assert hl == pytest.approx(heat_loss_true, rel=0.1)
