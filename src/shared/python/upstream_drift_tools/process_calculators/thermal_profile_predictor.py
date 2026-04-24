"""thermal_profile_predictor.py module."""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

__all__ = ["fit_heating_parameters", "predict_temperature_profile"]


def _heating_ode(
    t: float,
    y: Sequence[float],
    thermal_mass: float,
    heat_loss_coeff: float,
    ambient_temp: float,
    power_func: Callable[[float], float],
) -> Sequence[float]:
    """ODE for simple vessel heating."""
    if t is None:
        raise ValueError("t must be provided")
    q_in = power_func(t)
    dTdt = (q_in - heat_loss_coeff * (y[0] - ambient_temp)) / thermal_mass
    return [dTdt]


def predict_temperature_profile(
    t_span: tuple[float, float],
    t_eval: Sequence[float],
    initial_temp: float,
    thermal_mass: float,
    heat_loss_coeff: float,
    ambient_temp: float,
    power_func: Callable[[float], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict temperature profile for a heated vessel."""

    if t_span is None:
        raise ValueError("t_span must be provided")

    def rhs(t: float, y: Any) -> Any:
        return _heating_ode(
            t, y, thermal_mass, heat_loss_coeff, ambient_temp, power_func
        )

    sol = solve_ivp(rhs, t_span, [initial_temp], t_eval=t_eval, vectorized=False)
    return sol.t, sol.y[0]


def fit_heating_parameters(
    times: Sequence[float],
    observed_temps: Sequence[float],
    initial_temp: float,
    thermal_mass_guess: float,
    heat_loss_guess: float,
    ambient_temp: float,
    power_func: Callable[[float], float],
) -> tuple[float, float]:
    """Fit thermal_mass and heat_loss_coeff to observed data."""

    if times is None:
        raise ValueError("times must be provided")

    def model(t: Any, thermal_mass: float, heat_loss_coeff: float) -> np.ndarray:
        """Model method.

        Returns:
            None
        """
        if thermal_mass is None:
            raise ValueError("thermal_mass must be provided")
        _, temps = predict_temperature_profile(
            (t[0], t[-1]),
            t,
            initial_temp,
            thermal_mass,
            heat_loss_coeff,
            ambient_temp,
            power_func,
        )
        return temps

    popt, _ = curve_fit(
        model,
        np.asarray(times),
        np.asarray(observed_temps),
        p0=[thermal_mass_guess, heat_loss_guess],
    )
    return popt[0], popt[1]
