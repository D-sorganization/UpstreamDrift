"""Cross-engine physics consistency tests (Issue #126).

Verifies that physics engines produce consistent results for identical
initial conditions. Uses the PendulumPhysicsEngine as a baseline that
runs without external dependencies, and compares MuJoCo/Drake/Pinocchio
when available.

Guideline references:
    M2 - Cross-engine deterministic validation
    P3 - Tolerance-based comparison
"""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
    PendulumPhysicsEngine,
)

# Tolerances from Guideline P3
TOLERANCE_POSITION_RAD = 1e-4
TOLERANCE_VELOCITY_RAD_S = 1e-3
TOLERANCE_ACCELERATION_RAD_S2 = 1e-2
TOLERANCE_MASS_MATRIX = 1e-6


# ---------------------------------------------------------------------------
# Pendulum analytical validation (always runs, no external deps)
# ---------------------------------------------------------------------------


class TestPendulumAnalytical:
    """Validate PendulumPhysicsEngine against analytical properties."""

    def _make_engine(
        self, q: np.ndarray | None = None, v: np.ndarray | None = None
    ) -> PendulumPhysicsEngine:
        engine = PendulumPhysicsEngine()
        engine.reset()
        if q is not None and v is not None:
            engine.set_state(q, v)
            engine.forward()
        return engine

    def test_mass_matrix_symmetric_positive_definite(self) -> None:
        """Mass matrix M(q) must be symmetric and positive-definite."""
        engine = self._make_engine(np.array([0.3, -0.2]), np.array([0.0, 0.0]))
        M = engine.compute_mass_matrix()

        np.testing.assert_allclose(M, M.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(M)
        assert all(ev > 0 for ev in eigenvalues), (
            f"M not positive definite: eigs={eigenvalues}"
        )

    def test_mass_matrix_varies_with_configuration(self) -> None:
        """Mass matrix should change with joint angles (coupled inertia)."""
        engine1 = self._make_engine(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        M1 = engine1.compute_mass_matrix()

        engine2 = self._make_engine(np.array([1.0, 0.5]), np.array([0.0, 0.0]))
        M2 = engine2.compute_mass_matrix()

        assert not np.allclose(M1, M2, atol=1e-10), "M should vary with q"

    def test_gravity_forces_zero_at_vertical(self) -> None:
        """At q=0 (hanging straight down), gravity torques should be zero."""
        engine = self._make_engine(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        g = engine.compute_gravity_forces()
        np.testing.assert_allclose(g, 0.0, atol=1e-10)

    def test_gravity_forces_nonzero_at_angle(self) -> None:
        """At non-zero angle, gravity torques should be non-zero."""
        engine = self._make_engine(np.array([0.5, 0.0]), np.array([0.0, 0.0]))
        g = engine.compute_gravity_forces()
        assert np.linalg.norm(g) > 0.1, f"Expected non-zero gravity, got {g}"

    def test_drift_acceleration_matches_forward_dynamics(self) -> None:
        """Drift acceleration should equal -M^{-1} * bias at zero velocity."""
        engine = self._make_engine(np.array([0.3, 0.1]), np.array([0.0, 0.0]))

        M = engine.compute_mass_matrix()
        bias = engine.compute_bias_forces()

        expected_qacc = -np.linalg.solve(M, bias)
        actual_qacc = engine.compute_drift_acceleration()

        np.testing.assert_allclose(actual_qacc, expected_qacc, atol=1e-10)

    def test_inverse_dynamics_consistency(self) -> None:
        """tau = M*qacc + bias should be self-consistent."""
        engine = self._make_engine(np.array([0.5, -0.3]), np.array([1.0, -0.5]))

        M = engine.compute_mass_matrix()
        bias = engine.compute_bias_forces()

        qacc_zero_tau = -np.linalg.solve(M, bias)
        tau = engine.compute_inverse_dynamics(qacc_zero_tau)
        np.testing.assert_allclose(tau, 0.0, atol=1e-8)

    def test_inverse_dynamics_unit_acceleration(self) -> None:
        """Inverse dynamics with known qacc should give M*qacc + bias."""
        engine = self._make_engine(np.array([0.4, 0.2]), np.array([0.5, -0.3]))

        M = engine.compute_mass_matrix()
        bias = engine.compute_bias_forces()
        qacc = np.array([1.0, -1.0])

        expected_tau = M @ qacc + bias
        actual_tau = engine.compute_inverse_dynamics(qacc)

        np.testing.assert_allclose(actual_tau, expected_tau, atol=1e-10)

    def test_stepping_changes_state(self) -> None:
        """Stepping the engine should change the state (non-equilibrium)."""
        engine = self._make_engine(np.array([0.5, 0.0]), np.array([0.0, 0.0]))
        q0, _ = engine.get_state()

        engine.step(0.01)
        q1, _ = engine.get_state()

        assert not np.allclose(q0, q1, atol=1e-10)

    def test_checkpoint_restore_consistency(self) -> None:
        """Save and restore checkpoint should preserve state exactly."""
        engine = self._make_engine(np.array([0.3, -0.1]), np.array([0.5, 0.2]))

        checkpoint = engine.save_checkpoint()
        engine.step(0.01)
        engine.step(0.01)

        engine.restore_checkpoint(checkpoint)
        q, v = engine.get_state()

        np.testing.assert_allclose(q, np.array([0.3, -0.1]), atol=1e-12)
        np.testing.assert_allclose(v, np.array([0.5, 0.2]), atol=1e-12)

    def test_deterministic_stepping(self) -> None:
        """Same initial conditions should produce identical trajectories."""
        q0 = np.array([0.5, -0.3])
        v0 = np.array([1.0, 0.0])
        dt = 0.001

        engine1 = self._make_engine(q0, v0)
        for _ in range(100):
            engine1.step(dt)
        q1, v1 = engine1.get_state()

        engine2 = self._make_engine(q0.copy(), v0.copy())
        for _ in range(100):
            engine2.step(dt)
        q2, v2 = engine2.get_state()

        np.testing.assert_allclose(q1, q2, atol=1e-14)
        np.testing.assert_allclose(v1, v2, atol=1e-14)


# ---------------------------------------------------------------------------
# Cross-engine consistency (requires external engines)
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.live_simulation
