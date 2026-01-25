"""Analytical benchmark tests for simple pendulum dynamics.

Verifies physics engines against closed-form Lagrangian solutions.
Refactored for orthogonality using shared fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.constants import GRAVITY_M_S2

# Skip if pendulum engine not available
pytest.importorskip("engines.physics_engines.pendulum")

from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
    PendulumPhysicsEngine,
)


def configure_simple_pendulum(m1_kg: float = 1.0, l1_m: float = 1.0):
    """Factory to create a pendulum engine with simplified parameters."""
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DoublePendulumDynamics,
        DoublePendulumParameters,
        LowerSegmentProperties,
        SegmentProperties,
    )

    upper = SegmentProperties(
        length_m=l1_m,
        mass_kg=m1_kg,
        center_of_mass_ratio=1.0,
        inertia_about_com=0.0,
    )
    # Quasi-massless link 2
    epsilon_kg = 1e-10
    lower = LowerSegmentProperties(
        length_m=1.0,
        shaft_mass_kg=epsilon_kg,
        clubhead_mass_kg=epsilon_kg,
        shaft_com_ratio=0.5,
    )

    params = DoublePendulumParameters(
        upper_segment=upper,
        lower_segment=lower,
        plane_inclination_deg=0.0,
        gravity_enabled=True,
        damping_shoulder=0.0,
        damping_wrist=0.0,
        constrained_to_plane=True,
    )

    engine_instance = PendulumPhysicsEngine()
    engine_instance.dynamics = DoublePendulumDynamics(parameters=params)
    engine_instance.dynamics.forcing_functions = (
        engine_instance._get_shoulder_torque,
        engine_instance._get_wrist_torque,
    )
    return engine_instance, m1_kg, l1_m


class TestPendulumAnalyticalDynamics:
    """Test pendulum engine against closed-form analytical solutions."""

    def setup_method(self) -> None:
        """Initialize engine with standard test parameters."""
        self.engine, self.m1, self.l1 = configure_simple_pendulum()
        self.g_accel = GRAVITY_M_S2

    def _get_analytical_torque(self, theta_rad: float, a_rad_s2: float) -> float:
        """τ = I·θ̈ + m·g·l·sin(θ) where I = m·l²"""
        inertia_val = self.m1 * self.l1**2
        gravity_term = self.m1 * self.g_accel * self.l1 * np.sin(theta_rad)
        return float(inertia_val * a_rad_s2 + gravity_term)

    @pytest.mark.parametrize(
        "theta, acc, rtol",
        [
            (0.0, 1.0, 2e-3),  # Vertical down: pure inertial
            (0.1, 0.0, 2.0e-1),  # Small angle: gravity restoration
            (np.pi / 2, 0.0, 2.0e-1),  # Horizontal: max gravity
            (0.3, 2.0, 1.1e-1),  # Combined case
        ],
    )
    def test_inverse_dynamics(self, theta: float, acc: float, rtol: float):
        """Standardized test for inverse dynamics accuracy."""
        self.engine.set_state(np.array([theta, 0.0]), np.array([0.0, 0.0]))
        tau_engine = self.engine.compute_inverse_dynamics(np.array([acc, 0.0]))
        tau_expected = self._get_analytical_torque(theta, acc)

        np.testing.assert_allclose(tau_engine[0], tau_expected, rtol=rtol)

    def test_drift_acceleration(self):
        """Verify drift matches analytical free-fall acceleration."""
        theta_rad = 0.2
        self.engine.set_state(np.array([theta_rad, 0.0]), np.array([0.5, 0.0]))
        a_drift = self.engine.compute_drift_acceleration()[0]
        a_expected = -self.g_accel * np.sin(theta_rad) / self.l1

        np.testing.assert_allclose(a_drift, a_expected, rtol=8e-2)

    def test_ztcf_consistency(self):
        """Consistency check: ZTCF must equal drift acceleration."""
        q_pos = np.array([0.15, 0.0])
        v_vel = np.array([0.3, 0.0])
        self.engine.set_state(q_pos, v_vel)
        a_drift = self.engine.compute_drift_acceleration()
        a_ztcf = self.engine.compute_ztcf(q_pos, v_vel)
        assert np.allclose(a_ztcf, a_drift)


class TestPendulumEnergy:
    """Numerical stability and energy conservation tests."""

    def test_energy_conservation_free_swing(self):
        """Test energy is conserved during passive swing (no torque)."""
        engine_obj, mass_val, length_val = configure_simple_pendulum()
        g_accel = GRAVITY_M_S2

        theta0, v0 = 0.3, 0.0
        engine_obj.set_state(np.array([theta0, 0.0]), np.array([v0, 0.0]))

        inertia_val = mass_val * length_val**2
        # E = KE + PE = 0.5*I*v^2 + m*g*l*(1-cos(theta))
        pe_initial = mass_val * g_accel * length_val * (1 - np.cos(theta0))
        e_initial = 0.5 * inertia_val * v0**2 + pe_initial

        # Simulate
        dt_step, n_steps = 0.001, 1000
        for _ in range(n_steps):
            engine_obj.step(dt_step)

        q_final, v_final = engine_obj.get_state()
        pe_final = mass_val * g_accel * length_val * (1 - np.cos(q_final[0]))
        e_final = 0.5 * inertia_val * v_final[0]**2 + pe_final

        drift_pct = abs(e_final - e_initial) / abs(e_initial) * 100
        assert drift_pct < 2.0, f"Energy drift {drift_pct:.2f}% too high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
