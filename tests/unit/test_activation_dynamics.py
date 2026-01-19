"""Tests for activation dynamics module.

Tests the neural excitation to muscle activation dynamics modeling,
including first-order differential equations with asymmetric time constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from shared.python.activation_dynamics import ActivationDynamics


class TestActivationDynamicsInitialization:
    """Test ActivationDynamics initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        dynamics = ActivationDynamics()
        assert dynamics.tau_act == 0.010  # 10 ms default
        assert dynamics.tau_deact == 0.040  # 40 ms default
        assert dynamics.min_activation == 0.001

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        dynamics = ActivationDynamics(
            tau_act=0.015,
            tau_deact=0.050,
            min_activation=0.005,
        )
        assert dynamics.tau_act == 0.015
        assert dynamics.tau_deact == 0.050
        assert dynamics.min_activation == 0.005

    def test_negative_tau_act_raises_error(self):
        """Test that negative tau_act raises ValueError."""
        with pytest.raises(ValueError, match="Time constants must be positive"):
            ActivationDynamics(tau_act=-0.010, tau_deact=0.040)

    def test_negative_tau_deact_raises_error(self):
        """Test that negative tau_deact raises ValueError."""
        with pytest.raises(ValueError, match="Time constants must be positive"):
            ActivationDynamics(tau_act=0.010, tau_deact=-0.040)

    def test_zero_tau_act_raises_error(self):
        """Test that zero tau_act raises ValueError."""
        with pytest.raises(ValueError, match="Time constants must be positive"):
            ActivationDynamics(tau_act=0.0, tau_deact=0.040)

    def test_zero_tau_deact_raises_error(self):
        """Test that zero tau_deact raises ValueError."""
        with pytest.raises(ValueError, match="Time constants must be positive"):
            ActivationDynamics(tau_act=0.010, tau_deact=0.0)


class TestComputeDerivative:
    """Test compute_derivative method."""

    @pytest.fixture
    def dynamics(self):
        """Create standard activation dynamics instance."""
        return ActivationDynamics(tau_act=0.010, tau_deact=0.040)

    def test_activation_regime_positive_derivative(self, dynamics):
        """Test that da/dt > 0 when u > a (activation)."""
        u = 0.8
        a = 0.2
        dadt = dynamics.compute_derivative(u, a)
        assert dadt > 0, "Derivative should be positive when u > a"

    def test_deactivation_regime_negative_derivative(self, dynamics):
        """Test that da/dt < 0 when u < a (deactivation)."""
        u = 0.2
        a = 0.8
        dadt = dynamics.compute_derivative(u, a)
        assert dadt < 0, "Derivative should be negative when u < a"

    def test_equilibrium_zero_derivative(self, dynamics):
        """Test that da/dt ≈ 0 when u = a (equilibrium)."""
        u = 0.5
        a = 0.5
        dadt = dynamics.compute_derivative(u, a)
        assert abs(dadt) < 1e-10, "Derivative should be zero at equilibrium"

    def test_activation_time_constant_formula(self, dynamics):
        """Test activation time constant formula: τ = τ_act * (0.5 + 1.5*a)."""
        u = 0.8  # u > a (activation regime)
        a = 0.2

        # Expected: τ = 0.010 * (0.5 + 1.5 * 0.2) = 0.010 * 0.8 = 0.008
        # da/dt = (u - a) / τ = (0.8 - 0.2) / 0.008 = 0.6 / 0.008 = 75.0
        expected_tau = dynamics.tau_act * (0.5 + 1.5 * a)
        expected_dadt = (u - a) / expected_tau

        dadt = dynamics.compute_derivative(u, a)
        np.testing.assert_allclose(dadt, expected_dadt, rtol=1e-10)

    def test_deactivation_time_constant_formula(self, dynamics):
        """Test deactivation time constant formula: τ = τ_deact / (0.5 + 1.5*a)."""
        u = 0.2  # u < a (deactivation regime)
        a = 0.8

        # Expected: τ = 0.040 / (0.5 + 1.5 * 0.8) = 0.040 / 1.7 ≈ 0.02353
        # da/dt = (u - a) / τ = (0.2 - 0.8) / 0.02353 ≈ -25.49
        expected_tau = dynamics.tau_deact / (0.5 + 1.5 * a)
        expected_dadt = (u - a) / expected_tau

        dadt = dynamics.compute_derivative(u, a)
        np.testing.assert_allclose(dadt, expected_dadt, rtol=1e-10)

    def test_input_clamping_high(self, dynamics):
        """Test that inputs > 1.0 are clamped to 1.0."""
        u = 1.5  # Above maximum
        a = 0.5

        # Should be treated as u = 1.0
        expected_dadt = dynamics.compute_derivative(1.0, a)
        actual_dadt = dynamics.compute_derivative(u, a)

        np.testing.assert_allclose(actual_dadt, expected_dadt, rtol=1e-10)

    def test_input_clamping_low(self, dynamics):
        """Test that inputs < min_activation are clamped to min_activation."""
        u = -0.1  # Below minimum
        a = 0.5

        # Should be treated as u = min_activation
        expected_dadt = dynamics.compute_derivative(dynamics.min_activation, a)
        actual_dadt = dynamics.compute_derivative(u, a)

        np.testing.assert_allclose(actual_dadt, expected_dadt, rtol=1e-10)

    def test_activation_clamping_low(self, dynamics):
        """Test that activation < min_activation is clamped."""
        u = 0.5
        a = -0.1  # Below minimum

        # Should be treated as a = min_activation
        expected_dadt = dynamics.compute_derivative(u, dynamics.min_activation)
        actual_dadt = dynamics.compute_derivative(u, a)

        np.testing.assert_allclose(actual_dadt, expected_dadt, rtol=1e-10)

    @pytest.mark.parametrize("a_value", [0.0, 0.2, 0.5, 0.8, 1.0])
    def test_activation_increases_with_activation_level(self, dynamics, a_value):
        """Test that activation time constant increases with activation level.

        Formula: τ_act(a) = τ_act * (0.5 + 1.5*a)
        So higher a -> higher τ -> slower activation response
        """
        if a_value == 0.0:
            a_value = dynamics.min_activation  # Avoid exact zero

        u = 1.0  # Full excitation

        # Skip equilibrium case (u = a) where dadt = 0
        if abs(u - a_value) < 1e-10:
            pytest.skip("Equilibrium case, dadt = 0")

        # Compute time constant from derivative
        dadt = dynamics.compute_derivative(u, a_value)
        tau_computed = (u - a_value) / dadt

        expected_tau = dynamics.tau_act * (0.5 + 1.5 * a_value)
        np.testing.assert_allclose(tau_computed, expected_tau, rtol=1e-6)


class TestUpdate:
    """Test update method."""

    @pytest.fixture
    def dynamics(self):
        """Create standard activation dynamics instance."""
        return ActivationDynamics(tau_act=0.010, tau_deact=0.040)

    def test_euler_integration(self, dynamics):
        """Test that update implements Euler integration: a(t+dt) = a(t) + da/dt * dt."""
        u = 0.8
        a = 0.2
        dt = 0.001

        dadt = dynamics.compute_derivative(u, a)
        expected_a_new = a + dadt * dt

        actual_a_new = dynamics.update(u, a, dt)

        np.testing.assert_allclose(actual_a_new, expected_a_new, rtol=1e-10)

    def test_output_clamping_high(self, dynamics):
        """Test that output is clamped to maximum of 1.0."""
        u = 1.0
        a = 0.99
        dt = 0.100  # Large time step that would overshoot

        a_new = dynamics.update(u, a, dt)
        assert a_new <= 1.0, "Activation should be clamped to 1.0"

    def test_output_clamping_low(self, dynamics):
        """Test that output is clamped to min_activation."""
        u = 0.0
        a = 0.01
        dt = 0.100  # Large time step that might go negative

        a_new = dynamics.update(u, a, dt)
        assert (
            a_new >= dynamics.min_activation
        ), "Activation should be clamped to min_activation"

    def test_single_step_increases_activation(self, dynamics):
        """Test that a single step increases activation when u > a."""
        u = 1.0
        a = 0.0
        dt = 0.001

        a_new = dynamics.update(u, a, dt)
        assert a_new > a, "Activation should increase when u > a"

    def test_single_step_decreases_activation(self, dynamics):
        """Test that a single step decreases activation when u < a."""
        u = 0.0
        a = 1.0
        dt = 0.001

        a_new = dynamics.update(u, a, dt)
        assert a_new < a, "Activation should decrease when u < a"

    def test_zero_time_step_no_change(self, dynamics):
        """Test that zero time step produces no change."""
        u = 1.0
        a = 0.5
        dt = 0.0

        a_new = dynamics.update(u, a, dt)
        np.testing.assert_allclose(a_new, a, rtol=1e-10)


class TestStepResponse:
    """Test step response behavior."""

    @pytest.fixture
    def dynamics(self):
        """Create standard activation dynamics instance."""
        return ActivationDynamics(tau_act=0.010, tau_deact=0.040)

    def test_activation_rise_time(self, dynamics):
        """Test activation rise time with step input 0 -> 1.

        For first-order system with varying time constant, we expect
        approximately 63% response at τ and 95% at ~3τ.

        Note: time constant increases with activation (τ = τ_act * (0.5 + 1.5*a)),
        so convergence is slower than simple first-order system.
        """
        u = 1.0
        a = 0.0
        dt = 0.0001  # Small time step for accuracy

        # Simulate for 50 ms (5x tau_act)
        duration = 0.050
        n_steps = int(duration / dt)

        for _ in range(n_steps):
            a = dynamics.update(u, a, dt)

        # After 50ms should be > 0.95 (relaxed from 0.99 due to varying time constant)
        assert a > 0.95, f"After 50ms, activation should be > 0.95, got {a:.4f}"

    def test_deactivation_fall_time(self, dynamics):
        """Test deactivation fall time with step input 1 -> 0.

        Deactivation should be slower than activation (40ms vs 10ms).
        """
        u = 0.0
        a = 1.0
        dt = 0.0001  # Small time step

        # Simulate for 200 ms (5x tau_deact)
        duration = 0.200
        n_steps = int(duration / dt)

        for _ in range(n_steps):
            a = dynamics.update(u, a, dt)

        # After 200ms (5x tau_deact), should be very close to min_activation
        assert a < 0.05, f"After 200ms, activation should be < 0.05, got {a:.4f}"

    def test_activation_faster_than_deactivation(self, dynamics):
        """Test that activation is faster than deactivation."""
        dt = 0.0001

        # Test activation (0 -> 1)
        u_act = 1.0
        a_act = 0.0
        duration = 0.030  # 30ms
        n_steps = int(duration / dt)

        for _ in range(n_steps):
            a_act = dynamics.update(u_act, a_act, dt)

        # Test deactivation (1 -> 0)
        u_deact = 0.0
        a_deact = 1.0

        for _ in range(n_steps):
            a_deact = dynamics.update(u_deact, a_deact, dt)

        # After same duration, activation should have progressed more
        # a_act should be closer to 1.0 than a_deact is to 0.0
        activation_progress = a_act - 0.0
        deactivation_progress = 1.0 - a_deact

        assert activation_progress > deactivation_progress, (
            f"Activation should be faster: "
            f"activation={activation_progress:.4f}, deactivation={deactivation_progress:.4f}"
        )

    def test_step_response_monotonic_increase(self, dynamics):
        """Test that activation increases monotonically with constant u > a."""
        u = 1.0
        a = 0.0
        dt = 0.001

        a_prev = a
        for _ in range(100):
            a = dynamics.update(u, a, dt)
            assert a >= a_prev, "Activation should increase monotonically"
            a_prev = a

    def test_step_response_monotonic_decrease(self, dynamics):
        """Test that activation decreases monotonically with constant u < a."""
        u = 0.0
        a = 1.0
        dt = 0.001

        a_prev = a
        for _ in range(100):
            a = dynamics.update(u, a, dt)
            assert a <= a_prev, "Activation should decrease monotonically"
            a_prev = a


class TestPhysiologicalRealism:
    """Test physiological realism of the model."""

    def test_typical_time_constants(self):
        """Test that typical time constants produce physiologically realistic responses.

        Based on literature (Thelen 2003):
        - τ_act ≈ 10 ms (fast calcium release)
        - τ_deact ≈ 40 ms (slow calcium pump)
        """
        dynamics = ActivationDynamics(tau_act=0.010, tau_deact=0.040)

        # After 20ms with full excitation, should have significant activation
        u = 1.0
        a = 0.0
        dt = 0.001
        duration = 0.020  # 20ms

        for _ in range(int(duration / dt)):
            a = dynamics.update(u, a, dt)

        # After 2x tau_act, should be > 80% activated
        assert a > 0.80, f"After 20ms, activation should be > 0.80, got {a:.4f}"

    def test_asymmetric_response(self):
        """Test asymmetric activation/deactivation is physiologically realistic.

        Activation (calcium release) is faster than deactivation (calcium pump).
        This is a key physiological characteristic of muscle dynamics.
        """
        dynamics = ActivationDynamics(tau_act=0.010, tau_deact=0.040)

        assert (
            dynamics.tau_deact > dynamics.tau_act
        ), "Deactivation should be slower than activation (physiological realism)"
        assert (
            dynamics.tau_deact / dynamics.tau_act == 4.0
        ), "Typical ratio is 4:1 (deactivation:activation)"

    def test_minimum_activation_prevents_division_by_zero(self):
        """Test that min_activation prevents numerical issues."""
        dynamics = ActivationDynamics(min_activation=0.001)

        # Even with zero excitation, deactivation time constant should be finite
        u = 0.0
        a = 0.001  # At minimum

        # Should not raise any errors
        dadt = dynamics.compute_derivative(u, a)
        assert np.isfinite(dadt), "Derivative should be finite at minimum activation"

    @pytest.mark.parametrize(
        "tau_act,tau_deact",
        [
            (0.010, 0.040),  # Typical
            (0.015, 0.050),  # Slower muscle
            (0.008, 0.030),  # Faster muscle
        ],
    )
    def test_different_muscle_types(self, tau_act, tau_deact):
        """Test that model works with different muscle fiber types.

        Fast-twitch and slow-twitch muscles have different time constants.
        """
        dynamics = ActivationDynamics(tau_act=tau_act, tau_deact=tau_deact)

        # Simulate step response
        u = 1.0
        a = 0.0
        dt = 0.0001
        duration = 0.100  # 100ms

        for _ in range(int(duration / dt)):
            a = dynamics.update(u, a, dt)

        # All muscle types should reach high activation eventually
        assert a > 0.95, f"Activation should reach > 0.95, got {a:.4f}"


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    @pytest.fixture
    def dynamics(self):
        """Create standard activation dynamics instance."""
        return ActivationDynamics(tau_act=0.010, tau_deact=0.040)

    def test_large_time_step_stability(self, dynamics):
        """Test that large time steps don't cause instability."""
        u = 1.0
        a = 0.0
        dt = 0.010  # Large time step (equal to tau_act)

        a_new = dynamics.update(u, a, dt)

        assert 0.0 <= a_new <= 1.0, "Activation should remain in bounds"
        assert np.isfinite(a_new), "Activation should be finite"

    def test_very_small_time_step(self, dynamics):
        """Test that very small time steps work correctly."""
        u = 1.0
        a = 0.5
        dt = 1e-8  # Very small time step

        a_new = dynamics.update(u, a, dt)

        # Change should be very small but finite
        assert np.isfinite(a_new), "Activation should be finite"
        assert abs(a_new - a) < 1e-6, "Change should be very small"

    def test_repeated_updates_stability(self, dynamics):
        """Test stability over many repeated updates."""
        u = 0.5
        a = 0.5
        dt = 0.001

        # Run for 1000 steps (1 second) - sufficient for stability verification
        for _ in range(1000):
            a = dynamics.update(u, a, dt)

        # At equilibrium (u = a = 0.5), should stay at 0.5
        np.testing.assert_allclose(a, 0.5, atol=0.01)

    def test_oscillating_input(self, dynamics):
        """Test response to oscillating excitation."""
        dt = 0.001
        a = 0.0

        # Simulate square wave at 10 Hz for 0.5 seconds
        for i in range(500):
            # 10 Hz = 100ms period, 50ms on, 50ms off
            if (i % 100) < 50:
                u = 1.0
            else:
                u = 0.0

            a = dynamics.update(u, a, dt)

            # Should remain bounded
            assert 0.0 <= a <= 1.0, f"Activation out of bounds at step {i}: {a}"
            assert np.isfinite(a), f"Activation not finite at step {i}: {a}"

    def test_return_type_is_float(self, dynamics):
        """Test that update and compute_derivative return Python float."""
        u = 0.5
        a = 0.3
        dt = 0.001

        dadt = dynamics.compute_derivative(u, a)
        a_new = dynamics.update(u, a, dt)

        assert isinstance(dadt, float), "compute_derivative should return float"
        assert isinstance(a_new, float), "update should return float"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def dynamics(self):
        """Create standard activation dynamics instance."""
        return ActivationDynamics(tau_act=0.010, tau_deact=0.040)

    def test_zero_excitation_zero_activation(self, dynamics):
        """Test behavior when both excitation and activation are zero."""
        u = 0.0
        a = 0.0
        dt = 0.001

        a_new = dynamics.update(u, a, dt)

        # Should stay at minimum activation
        np.testing.assert_allclose(a_new, dynamics.min_activation, rtol=1e-10)

    def test_full_excitation_full_activation(self, dynamics):
        """Test behavior when both excitation and activation are 1.0."""
        u = 1.0
        a = 1.0
        dt = 0.001

        a_new = dynamics.update(u, a, dt)

        # Should stay at 1.0 (equilibrium)
        np.testing.assert_allclose(a_new, 1.0, atol=1e-6)

    def test_excitation_equals_activation_equilibrium(self, dynamics):
        """Test that u = a represents an equilibrium point."""
        for equilibrium_point in [0.2, 0.5, 0.8]:
            u = equilibrium_point
            a = equilibrium_point
            dt = 0.001

            a_new = dynamics.update(u, a, dt)

            # Should remain very close to equilibrium
            np.testing.assert_allclose(a_new, equilibrium_point, atol=1e-6)

    def test_extreme_time_constants(self):
        """Test with extreme but valid time constants."""
        # Very fast dynamics
        fast = ActivationDynamics(tau_act=0.001, tau_deact=0.004)
        u, a, dt = 1.0, 0.0, 0.0001

        for _ in range(100):  # 10ms
            a = fast.update(u, a, dt)

        assert a > 0.95, "Fast dynamics should activate quickly"

        # Very slow dynamics
        slow = ActivationDynamics(tau_act=0.100, tau_deact=0.400)
        a = 0.0

        for _ in range(100):  # 10ms
            a = slow.update(u, a, dt)

        assert a < 0.20, "Slow dynamics should activate slowly"

    def test_custom_min_activation(self):
        """Test with custom minimum activation value."""
        dynamics = ActivationDynamics(
            tau_act=0.010,
            tau_deact=0.040,
            min_activation=0.01,  # Higher minimum
        )

        u = 0.0
        a = 0.5
        dt = 0.001

        # Deactivate for long time
        for _ in range(1000):  # 1 second
            a = dynamics.update(u, a, dt)

        # Should not go below min_activation
        assert a >= dynamics.min_activation
        np.testing.assert_allclose(a, dynamics.min_activation, atol=1e-6)
