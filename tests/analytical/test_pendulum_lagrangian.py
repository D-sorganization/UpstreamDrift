"""Analytical benchmark tests for simple pendulum dynamics.

Assessment B Finding B-001 (BLOCKER) - Mathematical Ground Truth Validation

This module tests physics engines against ANALYTICAL (closed-form) solutions,
not just cross-engine agreement. This proves the equations of motion are
mathematically correct, not just that engines agree with each other.

Analytical Solution for Simple Pendulum:
----------------------------------------
Lagrangian: L = 0.5·m·l²·θ̇² - m·g·l·(1 - cos θ)

Equation of Motion (Euler-Lagrange):
    τ = m·l²·θ̈ + m·g·l·sin(θ)

For small angles (linearized):
    τ ≈ m·l²·θ̈ + m·g·l·θ

Source: Classical Mechanics (Goldstein, 3rd ed.), Section 1.5
"""

import numpy as np
import pytest

# Try to import pendulum engine, skip tests if not available
pytest.importorskip("engines.physics_engines.pendulum")

from engines.physics_engines.pendulum.python.pendulum_physics_engine import (
    PendulumPhysicsEngine,
)


class TestPendulumAnalyticalDynamics:
    """Test pendulum engine against closed-form analytical solutions.

    These are NOT cross-engine comparisons. These test against MATHEMATICAL TRUTH.
    """

    def setup_method(self) -> None:
        """Initialize pendulum for each test."""
        self.engine = PendulumPhysicsEngine()

        # Simple pendulum parameters (from DoublePendulumDynamics defaults)
        # Link 1: m1 = 1.0 kg, l1 = 1.0 m
        # Link 2: m2 = 1.0 kg, l2 = 1.0 m
        # Gravity: g = 9.80665 m/s² (NIST standard)

        self.m1 = 1.0  # kg
        self.l1 = 1.0  # m
        self.m2 = 1.0  # kg
        self.l2 = 1.0  # m
        self.g = 9.80665  # m/s² (NIST CODATA 2018)

    def analytical_single_pendulum_torque(
        self, theta: float, theta_dot: float, theta_ddot: float
    ) -> float:
        """Closed-form inverse dynamics for simple pendulum (link 1 only).

        Equation of motion: τ = I·θ̈ + m·g·l·sin(θ)
        where I = m·l² (point mass inertia)

        Args:
            theta: Angle [rad]
            theta_dot: Angular velocity [rad/s]
            theta_ddot: Angular acceleration [rad/s²]

        Returns:
            Required torque [N·m]

        Note:
            This ignores link 2 (treats as single pendulum).
            For double pendulum, coupling terms appear.
        """
        I_inertia = self.m1 * self.l1**2  # Inertia [kg·m²]
        gravity_torque = self.m1 * self.g * self.l1 * np.sin(theta)
        return I_inertia * theta_ddot + gravity_torque

    def test_inverse_dynamics_vertical_down(self) -> None:
        """Test ID at vertical down (θ=0): pure inertial torque.

        Configuration: θ = 0 (vertical down), v = 0, a = 1.0 rad/s²

        Analytical solution:
            sin(0) = 0 → gravity torque = 0
            τ = I·a = (m·l²)·1.0 = 1.0·1.0²·1.0 = 1.0 N·m

        This is a PURE INERTIAL case (no gravity component).
        """
        theta = 0.0
        v = 0.0
        theta_ddot = 1.0  # rad/s²

        # Set state
        self.engine.set_state(np.array([theta, 0.0]), np.array([v, 0.0]))

        # Compute engine torque
        tau_engine = self.engine.compute_inverse_dynamics(np.array([theta_ddot, 0.0]))

        # Analytical torque (link 1 only)
        tau_analytical = self.analytical_single_pendulum_torque(theta, v, theta_ddot)

        # CRITICAL: Must match analytical solution
        np.testing.assert_allclose(
            tau_engine[0],
            tau_analytical,
            atol=1e-8,
            err_msg=f"Engine inverse dynamics DEVIATES from analytical solution!\\n"
            f"  θ = {theta:.6f} rad (vertical down)\\n"
            f"  a = {theta_ddot:.6f} rad/s²\\n"
            f"  τ_engine = {tau_engine[0]:.8e} N·m\\n"
            f"  τ_analytical = {tau_analytical:.8e} N·m\\n"
            f"  This is a BLOCKER: equations of motion are incorrect.",
        )

    def test_inverse_dynamics_small_angle(self) -> None:
        """Test ID at small angle (θ=0.1 rad ≈ 5.7°): linearized regime.

        Configuration: θ = 0.1 rad, v = 0, a = 0 rad/s²

        Analytical solution (static equilibrium, a=0):
            τ = m·g·l·sin(0.1) ≈ 1.0·9.80665·1.0·0.0998 ≈ 0.978 N·m

        This tests GRAVITY COMPONENT accuracy in small-angle regime.
        """
        theta = 0.1  # rad (≈ 5.7°)
        v = 0.0
        theta_ddot = 0.0  # Static equilibrium

        self.engine.set_state(np.array([theta, 0.0]), np.array([v, 0.0]))

        tau_engine = self.engine.compute_inverse_dynamics(np.array([theta_ddot, 0.0]))
        tau_analytical = self.analytical_single_pendulum_torque(theta, v, theta_ddot)

        np.testing.assert_allclose(
            tau_engine[0],
            tau_analytical,
            atol=1e-6,  # Slightly larger tolerance for numerical sin()
            err_msg=f"Engine gravity torque DEVIATES from analytical!\\n"
            f"  θ = {theta:.6f} rad\\n"
            f"  τ_engine = {tau_engine[0]:.8e} N·m\\n"
            f"  τ_analytical = {tau_analytical:.8e} N·m\\n"
            f"  Deviation: {abs(tau_engine[0] - tau_analytical):.2e} N·m",
        )

    def test_inverse_dynamics_horizontal(self) -> None:
        """Test ID at horizontal (θ=π/2): maximum gravity torque.

        Configuration: θ = π/2 (horizontal), v = 0, a = 0

        Analytical solution:
            sin(π/2) = 1.0 → max gravity torque
            τ = m·g·l·1.0 = 1.0·9.80665·1.0 ≈ 9.807 N·m

        This tests MAXIMUM GRAVITY configuration.
        """
        theta = np.pi / 2  # Horizontal
        v = 0.0
        theta_ddot = 0.0

        self.engine.set_state(np.array([theta, 0.0]), np.array([v, 0.0]))

        tau_engine = self.engine.compute_inverse_dynamics(np.array([theta_ddot, 0.0]))
        tau_analytical = self.analytical_single_pendulum_torque(theta, v, theta_ddot)

        np.testing.assert_allclose(
            tau_engine[0],
            tau_analytical,
            atol=1e-6,
            err_msg=f"Engine max gravity torque DEVIATES!\\n"
            f"  θ = π/2 (horizontal)\\n"
            f"  τ_engine = {tau_engine[0]:.8e} N·m\\n"
            f"  τ_analytical = {tau_analytical:.8e} N·m (expected ≈ m·g·l = {self.m1*self.g*self.l1:.6f})",
        )

    def test_inverse_dynamics_combined_inertial_gravity(self) -> None:
        """Test ID with both inertial and gravity components.

        Configuration: θ = 0.3 rad, v = 0, a = 2.0 rad/s²

        Analytical solution:
            I·a = 1.0·1.0²·2.0 = 2.0 N·m (inertial)
            m·g·l·sin(0.3) = 1.0·9.80665·1.0·0.2955 ≈ 2.898 N·m (gravity)
            τ_total = 2.0 + 2.898 = 4.898 N·m

        This tests SUPERPOSITION of inertial + gravity terms.
        """
        theta = 0.3  # rad
        v = 0.0
        theta_ddot = 2.0  # rad/s²

        self.engine.set_state(np.array([theta, 0.0]), np.array([v, 0.0]))

        tau_engine = self.engine.compute_inverse_dynamics(np.array([theta_ddot, 0.0]))
        tau_analytical = self.analytical_single_pendulum_torque(theta, v, theta_ddot)

        np.testing.assert_allclose(
            tau_engine[0],
            tau_analytical,
            atol=1e-6,
            err_msg=f"Engine combined dynamics DEVIATE from analytical!\\n"
            f"  θ = {theta:.6f} rad, a = {theta_ddot:.6f} rad/s²\\n"
            f"  τ_engine = {tau_engine[0]:.8e} N·m\\n"
            f"  τ_analytical = {tau_analytical:.8e} N·m\\n"
            f"  Components: I·a = {self.m1 * self.l1**2 * theta_ddot:.6f} N·m, "
            f"m·g·l·sin(θ) = {self.m1 * self.g * self.l1 * np.sin(theta):.6f} N·m",
        )

    def test_drift_acceleration_zero_control(self) -> None:
        """Test drift acceleration matches analytical τ=0 case.

        Configuration: θ = 0.2 rad, v = 0.5 rad/s, τ = 0

        Analytical solution (with Coriolis=0 for single pendulum, v negligible):
            M·a_drift = -g(θ) → a_drift = -m·g·l·sin(θ) / (m·l²)
            a_drift = -g·sin(θ)/l = -9.80665·sin(0.2)/1.0 ≈ -1.946 rad/s²

        This tests DRIFT-CONTROL DECOMPOSITION correctness.
        """
        theta = 0.2  # rad
        v = 0.5  # rad/s (small, Coriolis negligible for single pendulum)

        self.engine.set_state(np.array([theta, 0.0]), np.array([v, 0.0]))

        # Drift acceleration (τ=0)
        a_drift_engine = self.engine.compute_drift_acceleration()

        # Analytical drift (gravity restoration only, no Coriolis for single pendulum at small v)
        # M·a_drift + g(θ) = 0 → a_drift = -M⁻¹·g(θ)
        # For single pendulum: a_drift = -g·sin(θ)/l
        a_drift_analytical = -self.g * np.sin(theta) / self.l1

        np.testing.assert_allclose(
            a_drift_engine[0],
            a_drift_analytical,
            atol=1e-6,
            err_msg=f"Drift acceleration DEVIATES from analytical!\\n"
            f"  θ = {theta:.6f} rad, v = {v:.6f} rad/s\\n"
            f"  a_drift_engine = {a_drift_engine[0]:.8e} rad/s²\\n"
            f"  a_drift_analytical = {a_drift_analytical:.8e} rad/s²\\n"
            f"  This tests τ=0 passive dynamics (Guideline F).",
        )

    def test_ztcf_matches_drift(self) -> None:
        """Test ZTCF counterfactual matches drift acceleration.

        By definition, ZTCF(q,v) = drift acceleration at (q,v).

        This is a CONSISTENCY CHECK, not analytical validation,
        but ensures counterfactual implementation is correct.
        """
        theta = 0.15
        v = 0.3

        q = np.array([theta, 0.0])
        vel = np.array([v, 0.0])

        self.engine.set_state(q, vel)

        a_drift = self.engine.compute_drift_acceleration()
        a_ztcf = self.engine.compute_ztcf(q, vel)

        np.testing.assert_allclose(
            a_ztcf,
            a_drift,
            atol=1e-12,
            err_msg=f"ZTCF DOES NOT MATCH drift acceleration!\\n"
            f"  This violates Guideline G1 definition.\\n"
            f"  a_drift = {a_drift}\\n"
            f"  a_ztcf = {a_ztcf}",
        )


class TestPendulumEnergyConservation:
    """Test energy conservation using analytical solutions.

    For conservative pendulum (no damping), total energy E = KE + PE must be constant.
    """

    def setup_method(self) -> None:
        """Initialize pendulum."""
        self.engine = PendulumPhysicsEngine()
        self.m1 = 1.0  # kg
        self.l1 = 1.0  # m
        self.g = 9.80665  # m/s²

    def compute_total_energy(self, theta: float, theta_dot: float) -> float:
        """Analytical total energy for simple pendulum.

        E = KE + PE = 0.5·I·ω² + m·g·l·(1 - cos(θ))
        where I = m·l²

        Args:
            theta: Angle [rad]
            theta_dot: Angular velocity [rad/s]

        Returns:
            Total mechanical energy [J]
        """
        I_inertia = self.m1 * self.l1**2
        KE = 0.5 * I_inertia * theta_dot**2
        PE = self.m1 * self.g * self.l1 * (1 - np.cos(theta))
        return KE + PE

    def test_energy_conservation_free_swing(self) -> None:
        """Test energy is conserved during passive swing (no torque).

        Initial: θ = 0.3 rad, ω = 0 rad/s
        Simulate 1 second with τ = 0 (passive dynamics)
        Check: ΔE/E₀ < 1% (Guideline O3)

        This is a NUMERICAL INTEGRATION test, not just analytical.
        But we know E_analytical stays constant, so engine must too.
        """
        theta0 = 0.3  # rad
        v0 = 0.0

        self.engine.set_state(np.array([theta0, 0.0]), np.array([v0, 0.0]))
        self.engine.set_control(np.array([0.0, 0.0]))  # No torque

        E_initial = self.compute_total_energy(theta0, v0)

        # Simulate 1 second with dt = 0.001 s
        dt = 0.001
        n_steps = 1000
        for _ in range(n_steps):
            self.engine.step(dt)

        # Final energy
        q_final, v_final = self.engine.get_state()
        E_final = self.compute_total_energy(q_final[0], v_final[0])

        # Energy drift percentage
        drift_pct = abs(E_final - E_initial) / abs(E_initial) * 100

        assert (
            drift_pct < 1.0
        ), f"Energy drift EXCEEDS 1% tolerance (Guideline O3)!\\n  E_initial = {E_initial:.6f} J\\n  E_final = {E_final:.6f} J\\n  Drift: {drift_pct:.2f}% (tolerance: 1.0%)\\n  This indicates integration error or damping leakage."


# Run tests if executed as script
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
