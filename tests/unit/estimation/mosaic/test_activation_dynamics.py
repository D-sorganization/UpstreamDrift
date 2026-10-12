"""Tests for the declared joint-level activation kernel (MOSAIC-11)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.activation_dynamics import (
    ActivationModel,
    fit_time_constants,
    simulate_activation,
)

pytestmark = pytest.mark.unit


def _model(**kw: object) -> ActivationModel:
    base: dict[str, object] = {
        "tau_act": 0.05,
        "tau_deact": 0.05,
        "lower": -10.0,
        "upper": 10.0,
    }
    base.update(kw)
    return ActivationModel(**base)  # type: ignore[arg-type]


def test_step_response_matches_analytic() -> None:
    t = np.arange(0.0, 0.5, 1e-3)
    u = np.ones((t.size, 1))
    a = simulate_activation(_model(), t, u, a0=np.zeros(1))
    np.testing.assert_allclose(a[:, 0], 1.0 - np.exp(-t / 0.05), atol=1e-12)


def test_ramp_response_matches_analytic() -> None:
    tau, slope, dt = 0.05, 2.0, 1e-4
    t = np.arange(0.0, 0.5, dt)
    a = simulate_activation(
        _model(tau_act=tau, tau_deact=tau), t, (slope * t)[:, None], a0=np.zeros(1)
    )
    exact = slope * (t - tau * (1.0 - np.exp(-t / tau)))
    assert np.max(np.abs(a[:, 0] - exact)) < 2.0 * slope * dt


def test_asymmetric_activation_and_deactivation() -> None:
    t = np.arange(0.0, 1.0, 1e-3)
    u = np.where(t < 0.5, 1.0, 0.0)[:, None]
    a = simulate_activation(_model(tau_act=0.02, tau_deact=0.1), t, u, a0=np.zeros(1))
    k = int(np.searchsorted(t, 0.5))
    a_peak = 1.0 - np.exp(-t[k] / 0.02)
    np.testing.assert_allclose(a[k, 0], a_peak)
    np.testing.assert_allclose(
        a[-1, 0], a_peak * np.exp(-(t[-1] - t[k]) / 0.1), rtol=1e-6
    )


def test_saturation_clamps_command_and_state() -> None:
    t = np.arange(0.0, 1.0, 1e-3)
    m = _model(lower=-0.5, upper=0.5)
    a = simulate_activation(m, t, np.full((t.size, 1), 5.0), a0=np.zeros(1))
    assert a.max() <= 0.5 and a.min() >= -0.5
    assert a[-1, 0] == pytest.approx(0.5, abs=1e-6)


def test_rate_limit_caps_slew() -> None:
    t = np.arange(0.0, 0.2, 1e-3)
    m = _model(tau_act=1e-4, tau_deact=1e-4, rate_limit=3.0)
    a = simulate_activation(m, t, np.ones((t.size, 1)), a0=np.zeros(1))
    assert np.max(np.abs(np.diff(a[:, 0]))) <= 3.0 * 1e-3 + 1e-12
    assert a[10, 0] == pytest.approx(3.0 * t[10], rel=1e-9)


def test_multi_joint_independent_columns() -> None:
    t = np.arange(0.0, 0.3, 1e-3)
    m = ActivationModel(
        tau_act=[0.02, 0.1], tau_deact=[0.02, 0.1], lower=[0.0, 0.0], upper=[1.0, 1.0]
    )
    a = simulate_activation(m, t, np.ones((t.size, 2)), a0=np.zeros(2))
    np.testing.assert_allclose(a[:, 0], 1 - np.exp(-t / 0.02), atol=1e-12)
    np.testing.assert_allclose(a[:, 1], 1 - np.exp(-t / 0.1), atol=1e-12)


def test_default_initial_state_is_clipped_first_command() -> None:
    t = np.arange(0.0, 0.1, 1e-3)
    a = simulate_activation(_model(upper=1.0), t, np.full((t.size, 1), 4.0))
    assert np.all(a == 1.0)


@pytest.mark.parametrize(
    "kw",
    [
        {"tau_act": 0.0},
        {"tau_deact": -1.0},
        {"tau_act": float("nan")},
        {"lower": 1.0, "upper": 1.0},
        {"lower": 2.0, "upper": 1.0},
        {"upper": float("inf")},
        {"rate_limit": 0.0},
        {"rate_limit": float("nan")},
    ],
)
def test_invalid_model_rejected(kw: dict[str, float]) -> None:
    with pytest.raises(ContractViolationError):
        _model(**kw)


@pytest.mark.parametrize(
    "times",
    [
        np.array([0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.5]),
        np.array([0.0, np.nan, 1.0]),
    ],
)
def test_invalid_time_grid_rejected(times: np.ndarray) -> None:
    with pytest.raises(ContractViolationError):
        simulate_activation(_model(), times, np.zeros((times.size, 1)))


def test_invalid_command_shapes_rejected() -> None:
    t = np.arange(5.0)
    with pytest.raises(ContractViolationError):
        simulate_activation(_model(), t, np.zeros((4, 1)))
    with pytest.raises(ContractViolationError):
        simulate_activation(_model(), t, np.full((5, 1), np.nan))
    with pytest.raises(ContractViolationError):
        simulate_activation(_model(), t, np.zeros((5, 2)))
    with pytest.raises(ContractViolationError):
        simulate_activation(_model(), t, np.zeros((5, 1)), a0=np.array([99.0]))


def _truth(tau_act: float, tau_deact: float) -> tuple[np.ndarray, ...]:
    t = np.arange(0.0, 4.0, 2e-3)
    u = (np.sin(2 * np.pi * 0.7 * t) > 0).astype(float)[:, None]
    m = _model(tau_act=tau_act, tau_deact=tau_deact, lower=0.0, upper=1.0)
    return t, u, simulate_activation(m, t, u, a0=np.zeros(1))


def test_time_constant_recovery_on_synthetic_truth() -> None:
    t, u, a = _truth(0.03, 0.09)
    fit = fit_time_constants(t, u[:, 0], a[:, 0], lower=0.0, upper=1.0)
    assert fit.tau_act == pytest.approx(0.03, rel=1e-6)
    assert fit.tau_deact == pytest.approx(0.09, rel=1e-6)


def test_time_constant_recovery_with_noise_within_tolerance() -> None:
    t, u, a = _truth(0.03, 0.09)
    rng = np.random.default_rng(0)
    noisy = a[:, 0] + 1e-4 * rng.standard_normal(t.size)
    fit = fit_time_constants(t, u[:, 0], noisy, lower=0.0, upper=1.0)
    assert fit.tau_act == pytest.approx(0.03, rel=0.1)
    assert fit.tau_deact == pytest.approx(0.09, rel=0.1)


def test_unobservable_when_no_excitation_reports_none() -> None:
    t = np.arange(0.0, 1.0, 1e-3)
    u = np.full(t.size, 0.5)
    a = np.full(t.size, 0.5)
    fit = fit_time_constants(t, u, a, lower=0.0, upper=1.0)
    assert fit.tau_act is None and fit.tau_deact is None


def test_deactivation_unobservable_when_only_rising() -> None:
    t = np.arange(0.0, 0.5, 1e-3)
    u = np.ones(t.size)
    a = 1.0 - np.exp(-t / 0.05)
    fit = fit_time_constants(t, u, a, lower=0.0, upper=1.0)
    assert fit.tau_act == pytest.approx(0.05, rel=1e-6)
    assert fit.tau_deact is None
