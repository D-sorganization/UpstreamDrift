"""Acceptance tests for ZTCF/ZVCF Counterfactual Experiments.

Guideline G1 (ZTCF) and G2 (ZVCF) Implementation Tests.

These tests verify that counterfactual acceleration computations work correctly
across all physics engines that implement them.

Tests:
1. ZTCF (Zero-Torque Counterfactual): Verify drift equals ZTCF at current state
2. ZVCF (Zero-Velocity Counterfactual): Verify gravity-dominated acceleration at v=0
3. Causal Decomposition: Verify a_full = a_drift + a_control
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from shared.python.interfaces import PhysicsEngine

LOGGER = logging.getLogger(__name__)

# Tolerance for numerical comparisons
ACCELERATION_TOLERANCE = 1e-6  # [rad/s² or m/s²]


class TestZTCFCounterfactual:
    """Tests for Zero-Torque Counterfactual (Guideline G1)."""

    @pytest.fixture
    def pendulum_engine(self) -> PhysicsEngine:
        """Create a pendulum engine for testing."""
        from engines.physics_engines.pendulum.python.pendulum_physics_engine import (
            PendulumPhysicsEngine,
        )

        engine = PendulumPhysicsEngine()
        return engine

    def test_ztcf_equals_drift_at_current_state(
        self, pendulum_engine: PhysicsEngine
    ) -> None:
        """ZTCF at current state should equal drift acceleration.

        When computing ZTCF at the engine's current (q, v) with τ=0,
        the result should match compute_drift_acceleration().
        """
        engine = pendulum_engine

        # Set a non-trivial state
        q = np.array([np.pi / 4, np.pi / 6])  # 45° and 30°
        v = np.array([1.0, -0.5])  # Some angular velocities
        engine.set_state(q, v)

        # Compute drift acceleration (τ=0 at current state)
        a_drift = engine.compute_drift_acceleration()

        # Compute ZTCF at the same state
        a_ztcf = engine.compute_ztcf(q, v)

        # They should be identical
        np.testing.assert_allclose(
            a_ztcf,
            a_drift,
            atol=ACCELERATION_TOLERANCE,
            err_msg="ZTCF should equal drift acceleration at current state",
        )

    def test_ztcf_at_different_state(self, pendulum_engine: PhysicsEngine) -> None:
        """ZTCF can be computed at arbitrary (q, v), not just current state.

        This tests that ZTCF correctly handles state different from current.
        """
        engine = pendulum_engine

        # Set current state
        engine.set_state(np.array([0.0, 0.0]), np.array([0.0, 0.0]))

        # Compute ZTCF at a different state
        q_test = np.array([np.pi / 3, np.pi / 4])
        v_test = np.array([2.0, 1.0])
        a_ztcf = engine.compute_ztcf(q_test, v_test)

        # Verify non-zero result (pendulum with gravity)
        assert a_ztcf.size > 0, "ZTCF should return non-empty result"
        assert not np.allclose(
            a_ztcf, 0, atol=1e-10
        ), "ZTCF should be non-zero with gravity"

    def test_ztcf_preserves_engine_state(self, pendulum_engine: PhysicsEngine) -> None:
        """Computing ZTCF should not modify the engine's internal state."""
        engine = pendulum_engine

        # Set initial state
        q_init = np.array([0.1, 0.2])
        v_init = np.array([0.3, 0.4])
        engine.set_state(q_init, v_init)

        # Compute ZTCF at a DIFFERENT state
        q_test = np.array([1.0, 1.5])
        v_test = np.array([2.0, 2.5])
        _ = engine.compute_ztcf(q_test, v_test)

        # Verify state is unchanged
        q_after, v_after = engine.get_state()
        np.testing.assert_allclose(
            q_after, q_init, atol=1e-12, err_msg="ZTCF should not modify position state"
        )
        np.testing.assert_allclose(
            v_after, v_init, atol=1e-12, err_msg="ZTCF should not modify velocity state"
        )


class TestZVCFCounterfactual:
    """Tests for Zero-Velocity Counterfactual (Guideline G2)."""

    @pytest.fixture
    def pendulum_engine(self) -> PhysicsEngine:
        """Create a pendulum engine for testing."""
        from engines.physics_engines.pendulum.python.pendulum_physics_engine import (
            PendulumPhysicsEngine,
        )

        engine = PendulumPhysicsEngine()
        return engine

    def test_zvcf_removes_coriolis_effects(
        self, pendulum_engine: PhysicsEngine
    ) -> None:
        """ZVCF should have no Coriolis/centrifugal contribution.

        With v=0, the Coriolis matrix C(q,v)·v = 0, so ZVCF isolates
        gravity and control effects only.
        """
        engine = pendulum_engine

        # Set state with non-zero velocity
        q = np.array([np.pi / 4, np.pi / 6])
        v = np.array([5.0, 3.0])  # High velocities for significant Coriolis
        engine.set_state(q, v)
        engine.set_control(np.array([0.0, 0.0]))  # Zero control

        # Compute full ZTCF (drift with current v)
        a_ztcf = engine.compute_ztcf(q, v)

        # Compute ZVCF (v=0, so no Coriolis)
        a_zvcf = engine.compute_zvcf(q)

        # ZVCF should be different from ZTCF due to removed Coriolis
        # (unless by coincidence they cancel, which is unlikely)
        assert not np.allclose(
            a_ztcf, a_zvcf, atol=0.01
        ), "ZVCF should differ from ZTCF when velocity is non-zero"

    def test_zvcf_at_rest_configuration(self, pendulum_engine: PhysicsEngine) -> None:
        """ZVCF at vertical (θ=0) should show gravity effect.

        A pendulum at rest vertically should have gravitational acceleration
        trying to swing it back if perturbed, or zero if perfectly balanced.
        """
        engine = pendulum_engine

        # Vertical configuration (both links pointing down)
        q = np.array([0.0, 0.0])
        engine.set_state(q, np.array([0.0, 0.0]))
        engine.set_control(np.array([0.0, 0.0]))

        a_zvcf = engine.compute_zvcf(q)

        # At equilibrium, acceleration is zero (marginally stable)
        np.testing.assert_allclose(
            a_zvcf,
            0.0,
            atol=1e-8,
            err_msg="ZVCF at vertical equilibrium should be ~zero",
        )

    def test_zvcf_preserves_engine_state(self, pendulum_engine: PhysicsEngine) -> None:
        """Computing ZVCF should not modify the engine's internal state."""
        engine = pendulum_engine

        # Set initial state with non-zero velocity
        q_init = np.array([0.1, 0.2])
        v_init = np.array([0.5, 0.6])
        engine.set_state(q_init, v_init)

        # Compute ZVCF at a DIFFERENT configuration
        q_test = np.array([1.0, 1.5])
        _ = engine.compute_zvcf(q_test)

        # Verify state is unchanged
        q_after, v_after = engine.get_state()
        np.testing.assert_allclose(
            q_after, q_init, atol=1e-12, err_msg="ZVCF should not modify position state"
        )
        np.testing.assert_allclose(
            v_after, v_init, atol=1e-12, err_msg="ZVCF should not modify velocity state"
        )


class TestCausalDecomposition:
    """Tests for Section F Drift-Control Superposition."""

    @pytest.fixture
    def pendulum_engine(self) -> PhysicsEngine:
        """Create a pendulum engine for testing."""
        from engines.physics_engines.pendulum.python.pendulum_physics_engine import (
            PendulumPhysicsEngine,
        )

        engine = PendulumPhysicsEngine()
        return engine

    def test_superposition_drift_plus_control_equals_full(
        self, pendulum_engine: PhysicsEngine
    ) -> None:
        """Verify a_full = a_drift + a_control (Section F requirement).

        The total acceleration should decompose into drift and control components.
        """
        engine = pendulum_engine

        # Set a non-trivial state with non-zero control
        q = np.array([np.pi / 4, np.pi / 6])
        v = np.array([1.0, -0.5])
        tau = np.array([2.0, 1.5])  # Applied torques

        engine.set_state(q, v)
        engine.set_control(tau)

        # Compute individual components
        a_drift = engine.compute_drift_acceleration()
        a_control = engine.compute_control_acceleration(tau)

        # Compute full acceleration (with current control)
        # For pendulum, full = drift + control
        a_full = a_drift + a_control

        # Verify superposition using inverse dynamics
        # tau = ID(q, v, a) => a = M^-1 * (tau - bias)
        M = engine.compute_mass_matrix()
        bias = engine.compute_bias_forces()
        a_expected = np.linalg.solve(M, tau - bias)

        np.testing.assert_allclose(
            a_full,
            a_expected,
            atol=1e-6,
            err_msg="Drift + Control should equal full acceleration",
        )

    def test_causal_attribution_ztcf(self, pendulum_engine: PhysicsEngine) -> None:
        """Verify Δa_control = a_full - a_ZTCF.

        The causal effect of control is the difference between full and ZTCF.
        """
        engine = pendulum_engine

        # Set state with control
        q = np.array([np.pi / 4, np.pi / 6])
        v = np.array([1.0, -0.5])
        tau = np.array([2.0, 1.5])

        engine.set_state(q, v)
        engine.set_control(tau)

        # Compute ZTCF (drift only)
        a_ztcf = engine.compute_ztcf(q, v)

        # Compute full acceleration
        M = engine.compute_mass_matrix()
        bias = engine.compute_bias_forces()
        a_full = np.linalg.solve(M, tau - bias)

        # Control-attributed acceleration
        delta_a_control = a_full - a_ztcf

        # This should equal M^-1 * tau
        a_control_expected = engine.compute_control_acceleration(tau)

        np.testing.assert_allclose(
            delta_a_control,
            a_control_expected,
            atol=1e-6,
            err_msg="Δa_control = a_full - a_ZTCF should equal M^-1 * τ",
        )


class TestCounterfactualCrossEngine:
    """Cross-engine validation for counterfactual computations."""

    @pytest.fixture
    def engines_with_ztcf(self) -> list[tuple[str, PhysicsEngine]]:
        """Get all engines that implement ZTCF/ZVCF."""
        engines: list[tuple[str, PhysicsEngine]] = []

        # Always include Pendulum (reference implementation)
        try:
            from engines.physics_engines.pendulum.python.pendulum_physics_engine import (
                PendulumPhysicsEngine,
            )

            engines.append(("Pendulum", PendulumPhysicsEngine()))
        except ImportError:
            pass

        return engines

    def test_ztcf_dimensionality(
        self, engines_with_ztcf: list[tuple[str, PhysicsEngine]]
    ) -> None:
        """ZTCF output should have correct dimensionality (n_v,)."""
        for name, engine in engines_with_ztcf:
            q, v = engine.get_state()
            a_ztcf = engine.compute_ztcf(q, v)

            assert a_ztcf.ndim == 1, f"{name}: ZTCF should be 1D array"
            assert len(a_ztcf) == len(v), f"{name}: ZTCF dim should match n_v"

    def test_zvcf_dimensionality(
        self, engines_with_ztcf: list[tuple[str, PhysicsEngine]]
    ) -> None:
        """ZVCF output should have correct dimensionality (n_v,)."""
        for name, engine in engines_with_ztcf:
            q, v = engine.get_state()
            a_zvcf = engine.compute_zvcf(q)

            assert a_zvcf.ndim == 1, f"{name}: ZVCF should be 1D array"
            assert len(a_zvcf) == len(v), f"{name}: ZVCF dim should match n_v"
