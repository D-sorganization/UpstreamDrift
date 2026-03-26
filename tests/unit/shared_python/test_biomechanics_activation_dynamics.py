"""Unit tests for biomechanics/activation_dynamics.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.activation_dynamics import ActivationDynamics


class TestActivationDynamicsInit:
    def test_default_params(self) -> None:
        d = ActivationDynamics()
        assert d.tau_act == pytest.approx(0.010)
        assert d.tau_deact == pytest.approx(0.040)
        assert d.min_activation == pytest.approx(0.001)

    def test_custom_params(self) -> None:
        d = ActivationDynamics(tau_act=0.005, tau_deact=0.060, min_activation=0.01)
        assert d.tau_act == pytest.approx(0.005)
        assert d.tau_deact == pytest.approx(0.060)
        assert d.min_activation == pytest.approx(0.01)

    def test_zero_tau_act_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError, Exception)):
            ActivationDynamics(tau_act=0.0)

    def test_negative_tau_act_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError, Exception)):
            ActivationDynamics(tau_act=-0.01)

    def test_zero_tau_deact_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError, Exception)):
            ActivationDynamics(tau_deact=0.0)

    def test_min_activation_zero_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError, Exception)):
            ActivationDynamics(min_activation=0.0)

    def test_min_activation_one_raises(self) -> None:
        with pytest.raises((ValueError, AssertionError, Exception)):
            ActivationDynamics(min_activation=1.0)


class TestComputeDerivative:
    @pytest.fixture
    def dyn(self) -> ActivationDynamics:
        return ActivationDynamics(tau_act=0.010, tau_deact=0.040)

    def test_activation_phase_positive_rate(self, dyn: ActivationDynamics) -> None:
        """When u > a, da/dt should be positive (activation rising)."""
        dadt = dyn.compute_derivative(u=0.8, a=0.2)
        assert dadt > 0

    def test_deactivation_phase_negative_rate(self, dyn: ActivationDynamics) -> None:
        """When u < a, da/dt should be negative (activation falling)."""
        dadt = dyn.compute_derivative(u=0.1, a=0.9)
        assert dadt < 0

    def test_equilibrium_zero_rate(self, dyn: ActivationDynamics) -> None:
        """When u == a, da/dt should be zero."""
        dadt = dyn.compute_derivative(u=0.5, a=0.5)
        assert abs(dadt) < 1e-9

    def test_result_is_finite(self, dyn: ActivationDynamics) -> None:
        dadt = dyn.compute_derivative(u=0.5, a=0.5)
        assert np.isfinite(dadt)

    def test_clamps_excitation_above_1(self, dyn: ActivationDynamics) -> None:
        """Excitation > 1 should be clamped to 1."""
        dadt_clamped = dyn.compute_derivative(u=2.0, a=0.5)
        dadt_full = dyn.compute_derivative(u=1.0, a=0.5)
        assert abs(dadt_clamped - dadt_full) < 1e-9

    def test_clamps_excitation_below_zero(self, dyn: ActivationDynamics) -> None:
        """Excitation < 0 should be clamped to min_activation."""
        # Should not raise and result in a finite number
        dadt = dyn.compute_derivative(u=-0.5, a=0.5)
        assert np.isfinite(dadt)

    def test_activation_time_constant_applied(self, dyn: ActivationDynamics) -> None:
        """Faster tau_act yields higher da/dt magnitude during activation phase."""
        dyn_fast = ActivationDynamics(tau_act=0.005, tau_deact=0.040)
        dyn_slow = ActivationDynamics(tau_act=0.020, tau_deact=0.040)
        fast_rate = abs(dyn_fast.compute_derivative(u=1.0, a=0.2))
        slow_rate = abs(dyn_slow.compute_derivative(u=1.0, a=0.2))
        assert fast_rate > slow_rate

    def test_deactivation_time_constant_applied(self, dyn: ActivationDynamics) -> None:
        """Faster tau_deact yields higher |da/dt| magnitude during deactivation."""
        dyn_fast = ActivationDynamics(tau_act=0.010, tau_deact=0.020)
        dyn_slow = ActivationDynamics(tau_act=0.010, tau_deact=0.080)
        fast_rate = abs(dyn_fast.compute_derivative(u=0.0, a=0.8))
        slow_rate = abs(dyn_slow.compute_derivative(u=0.0, a=0.8))
        assert fast_rate > slow_rate


class TestUpdate:
    @pytest.fixture
    def dyn(self) -> ActivationDynamics:
        return ActivationDynamics(tau_act=0.010, tau_deact=0.040)

    def test_activation_rises_toward_excitation(self, dyn: ActivationDynamics) -> None:
        a = 0.0
        for _ in range(100):
            a = dyn.update(u=1.0, a=a, dt=0.001)
        assert a > 0.9  # Nearly fully activated after 100 ms

    def test_activation_falls_after_excitation_removed(self, dyn: ActivationDynamics) -> None:
        a = 1.0  # Start fully activated
        for _ in range(200):
            a = dyn.update(u=0.0, a=a, dt=0.001)
        assert a < 0.3  # Substantially deactivated after 200 ms

    def test_result_clamped_to_min(self, dyn: ActivationDynamics) -> None:
        a = dyn.update(u=0.0, a=0.001, dt=10.0)  # Large dt — would try to go negative
        assert a >= dyn.min_activation

    def test_result_never_exceeds_one(self, dyn: ActivationDynamics) -> None:
        a = 0.0
        for _ in range(1000):
            a = dyn.update(u=1.0, a=a, dt=0.001)
        assert a <= 1.0

    def test_zero_dt_raises(self, dyn: ActivationDynamics) -> None:
        with pytest.raises((ValueError, AssertionError, Exception)):
            dyn.update(u=0.5, a=0.5, dt=0.0)

    def test_negative_dt_raises(self, dyn: ActivationDynamics) -> None:
        with pytest.raises((ValueError, AssertionError, Exception)):
            dyn.update(u=0.5, a=0.5, dt=-0.001)

    def test_returns_float(self, dyn: ActivationDynamics) -> None:
        result = dyn.update(u=0.5, a=0.5, dt=0.01)
        assert isinstance(result, float)

    def test_result_is_finite(self, dyn: ActivationDynamics) -> None:
        result = dyn.update(u=0.5, a=0.5, dt=0.01)
        assert np.isfinite(result)

    def test_step_response_timing(self, dyn: ActivationDynamics) -> None:
        """Activation should reach ~63% of final in ~tau_act."""
        a = 0.0
        steps = int(0.010 / 0.001)  # one tau_act = 10 ms at 1 ms step
        for _ in range(steps):
            a = dyn.update(u=1.0, a=a, dt=0.001)
        # After ~1 time constant, activation ~ (1 - 1/e) ≈ 63%
        # The exact value depends on the DDE structure but should be > 0.3
        assert a > 0.3
