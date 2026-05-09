"""Tests for electrode_advancement_calculator and thermal_profile_predictor (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
from src.shared.python.upstream_drift_tools.process_calculators.electrode_advancement_calculator import (
    ElectrodeAdvancementCalculator,
)
from src.shared.python.upstream_drift_tools.process_calculators.thermal_profile_predictor import (
    predict_temperature_profile,
)

# ---------------------------------------------------------------------------
# ElectrodeAdvancementCalculator
# ---------------------------------------------------------------------------


class TestElectrodeAdvancementCalculator:
    _CALC = ElectrodeAdvancementCalculator()

    def test_default_consumption_rate_positive(self) -> None:
        assert self._CALC.consumption_rate > 0.0

    def test_calculate_consumption_positive(self) -> None:
        result = self._CALC.calculate_consumption(current_ka=10.0, time_hrs=1.0)
        assert result > 0.0

    def test_consumption_proportional_to_current(self) -> None:
        low = self._CALC.calculate_consumption(current_ka=5.0, time_hrs=1.0)
        high = self._CALC.calculate_consumption(current_ka=10.0, time_hrs=1.0)
        assert abs(high - 2.0 * low) < 1e-10

    def test_consumption_proportional_to_time(self) -> None:
        one_hr = self._CALC.calculate_consumption(current_ka=10.0, time_hrs=1.0)
        two_hr = self._CALC.calculate_consumption(current_ka=10.0, time_hrs=2.0)
        assert abs(two_hr - 2.0 * one_hr) < 1e-10

    def test_zero_time_zero_consumption(self) -> None:
        result = self._CALC.calculate_consumption(current_ka=10.0, time_hrs=0.0)
        assert result == 0.0

    def test_zero_current_zero_consumption(self) -> None:
        result = self._CALC.calculate_consumption(current_ka=0.0, time_hrs=5.0)
        assert result == 0.0


# ---------------------------------------------------------------------------
# predict_temperature_profile
# ---------------------------------------------------------------------------


def _const_power(t: float) -> float:
    return 1000.0  # W — constant 1 kW heat input


class TestPredictTemperatureProfile:
    def test_electrode_and_thermal_returns_two_arrays(self) -> None:
        t_eval = np.linspace(0, 60, 10)
        times, temps = predict_temperature_profile(
            t_span=(0.0, 60.0),
            t_eval=t_eval,
            initial_temp=300.0,
            thermal_mass=5000.0,
            heat_loss_coeff=10.0,
            ambient_temp=293.0,
            power_func=_const_power,
        )
        assert len(times) == len(t_eval)
        assert len(temps) == len(t_eval)

    def test_initial_temp_matches(self) -> None:
        t_eval = np.linspace(0, 60, 5)
        _, temps = predict_temperature_profile(
            t_span=(0.0, 60.0),
            t_eval=t_eval,
            initial_temp=300.0,
            thermal_mass=5000.0,
            heat_loss_coeff=10.0,
            ambient_temp=293.0,
            power_func=_const_power,
        )
        assert abs(temps[0] - 300.0) < 0.1

    def test_temperature_increases_with_heating(self) -> None:
        t_eval = np.linspace(0, 300, 20)
        _, temps = predict_temperature_profile(
            t_span=(0.0, 300.0),
            t_eval=t_eval,
            initial_temp=300.0,
            thermal_mass=5000.0,
            heat_loss_coeff=2.0,
            ambient_temp=293.0,
            power_func=_const_power,
        )
        # Temperature should rise with constant heating
        assert temps[-1] > temps[0]
