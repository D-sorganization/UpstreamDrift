"""Issue #15045 muscle-model acceptance contract (SCI-07 / SCI-08)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.activation_dynamics import (
    MAX_ACTIVATION_INTEGRATION_DT,
    ActivationDynamics,
)
from src.shared.python.biomechanics.hill_muscle import HillMuscleModel, MuscleParameters

pytestmark = pytest.mark.unit

HILL_A = 0.25


def _concentric_hill(v: float) -> float:
    return (1.0 + v) / (1.0 - v / HILL_A)


def test_f_v_zero_at_and_beyond_max_shortening() -> None:
    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    assert model.force_velocity(-1.0) == pytest.approx(0.0, abs=1e-15)
    assert model.force_velocity(-1.5) == pytest.approx(0.0, abs=1e-15)
    assert _concentric_hill(-1.5) < 0.0


def test_f_v_c1_kink_at_zero_documented() -> None:
    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    h = 1e-8
    left = (model.force_velocity(-h) - model.force_velocity(-2 * h)) / h
    right = (model.force_velocity(h) - model.force_velocity(0.0)) / h
    assert model.force_velocity(0.0) == pytest.approx(1.0)
    assert abs(left - right) > 1e-3


def test_f_t_toe_junction_c0() -> None:
    model = HillMuscleModel(MuscleParameters(F_max=1.0, l_opt=1.0, l_slack=1.0))
    eps_0 = HillMuscleModel.DEFAULT_TENDON_STRAIN_AT_FMAX
    eps_toe = 0.609 * eps_0
    l_toe = 1.0 + eps_toe
    h = 1e-8
    assert model.tendon_force(l_toe - h) == pytest.approx(
        model.tendon_force(l_toe + h), abs=1e-6
    )


def _floor_exponential_step(u: float, a0: float, dt: float) -> float:
    tau = 0.010 * (0.5 + 1.5 * a0)
    return u + (a0 - u) * float(np.exp(-dt / tau))


def test_activation_one_ms_matches_exponential_formula() -> None:
    dyn = ActivationDynamics(tau_act=0.010, tau_deact=0.040, min_activation=0.001)
    dt = 0.001
    got = dyn.update(1.0, dyn.min_activation, dt)
    expected = _floor_exponential_step(1.0, dyn.min_activation, dt)
    assert got == pytest.approx(expected, abs=1e-9)


def test_activation_stable_at_mocap_step_sizes() -> None:
    dyn = ActivationDynamics(tau_act=0.010, tau_deact=0.040, min_activation=0.001)
    u = 1.0
    a0 = dyn.min_activation
    for dt in (0.001, 0.010, 0.020):
        got = dyn.update(u, a0, dt)
        assert dyn.min_activation <= got < 1.0


def test_activation_avoids_euler_overshoot_at_20ms() -> None:
    dyn = ActivationDynamics(tau_act=0.010, tau_deact=0.040, min_activation=0.001)
    dt = 0.020
    dadt = dyn.compute_derivative(1.0, dyn.min_activation)
    naive = float(
        np.clip(dyn.min_activation + dadt * dt, dyn.min_activation, 1.0)
    )
    integrated = dyn.update(1.0, dyn.min_activation, dt)
    assert naive == pytest.approx(1.0)
    assert integrated < 1.0 - 1e-6
    assert integrated < naive - 0.1


def test_max_activation_integration_dt_half_floor_tau() -> None:
    tau_floor = 0.010 * (0.5 + 1.5 * 0.001)
    assert pytest.approx(0.5 * tau_floor) == MAX_ACTIVATION_INTEGRATION_DT
