import numpy as np
import pytest
from sidekick.process_calculators.thermal_profile_predictor import (
    _heating_ode,
    fit_heating_parameters,
    predict_temperature_profile,
)


def power_func(t: float) -> float:
    return 1000.0  # Constant 1000 W


def test_heating_ode() -> None:
    res = _heating_ode(0.0, [20.0], 500.0, 10.0, 20.0, power_func)
    assert len(res) == 1
    assert res[0] == 2.0  # (1000 - 10*(20-20))/500 = 2.0

    res2 = _heating_ode(10.0, [120.0], 500.0, 10.0, 20.0, power_func)
    assert res2[0] == 0.0  # (1000 - 10*(100))/500 = 0.0


def test_predict_temperature_profile() -> None:
    t_eval = np.linspace(0, 100, 10)
    t, y = predict_temperature_profile(
        t_span=(0, 100),
        t_eval=t_eval,
        initial_temp=20.0,
        thermal_mass=500.0,
        heat_loss_coeff=10.0,
        ambient_temp=20.0,
        power_func=power_func,
    )

    assert len(t) == 10
    assert len(y) == 10
    assert y[0] == pytest.approx(20.0)
    assert y[-1] > 20.0


def test_fit_heating_parameters() -> None:
    t_eval = np.linspace(0, 50, 20)
    _, true_y = predict_temperature_profile(
        t_span=(0, 50),
        t_eval=t_eval,
        initial_temp=20.0,
        thermal_mass=800.0,
        heat_loss_coeff=15.0,
        ambient_temp=20.0,
        power_func=power_func,
    )

    noisy_y = true_y + np.random.normal(0, 0.5, size=len(true_y))

    mass_fit, loss_fit = fit_heating_parameters(
        times=t_eval,
        observed_temps=noisy_y,
        initial_temp=20.0,
        thermal_mass_guess=500.0,
        heat_loss_guess=10.0,
        ambient_temp=20.0,
        power_func=power_func,
    )

    assert mass_fit == pytest.approx(800.0, rel=0.1)
    assert loss_fit == pytest.approx(15.0, rel=0.1)
