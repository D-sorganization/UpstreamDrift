"""Tests for drift-control decomposition (Section F).

Verifies that drift + control = full dynamics for all physics engines.
Refactored for DRY compliance using parameterized engine tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# TOLERANCE for superposition test
SUPERPOSITION_TOLERANCE = 1e-5


def _get_engine(engine_name: str) -> Any:
    """Factory to get the requested physics engine, skipping if not available."""
    if engine_name == "pinocchio":
        try:
            import pinocchio as pin

            if not hasattr(pin, "__version__"):
                pytest.skip("Pinocchio mocked")
            from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                PinocchioPhysicsEngine,
            )

            return PinocchioPhysicsEngine()
        except ImportError:
            pytest.skip("Pinocchio not installed")

    elif engine_name == "mujoco":
        try:
            from src.engines.physics_engines.mujoco_humanoid_golf.physics_engine import (
                MuJoCoPhysicsEngine,
            )

            return MuJoCoPhysicsEngine()
        except ImportError:
            pytest.skip("MuJoCo not installed")

    pytest.skip(f"Engine {engine_name} not available")


class TestDriftControlDecomposition:
    """Acceptance tests for drift-control superposition (Section F).

    These tests verify that the causal decomposition holds:
    a_full = a_drift + a_control

    where:
    - a_drift = acceleration with zero control (gravity, Coriolis, centrifugal)
    - a_control = acceleration from control inputs only (M^-1 * tau)
    """

    @pytest.fixture(params=["pinocchio", "mujoco"])
    def engine(self, request: pytest.FixtureRequest) -> Any:
        """Parameterized fixture to test across different physics engines."""
        return _get_engine(request.param)

    def test_drift_acceleration_non_empty(self, engine: Any) -> None:
        """Drift acceleration should return a valid non-empty array."""
        # Set a non-trivial state
        q = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nq]
        v = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nv]
        engine.set_state(q, v)

        a_drift = engine.compute_drift_acceleration()

        assert a_drift.size > 0, "Drift acceleration should be non-empty"
        assert a_drift.ndim == 1, "Drift acceleration should be 1D"
        assert len(a_drift) == engine.nv, "Drift dim should match nv"

    def test_control_acceleration_non_empty(self, engine: Any) -> None:
        """Control acceleration should return a valid non-empty array."""
        # Set a non-trivial state
        q = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nq]
        v = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nv]
        tau = np.ones(engine.nv) * 0.5  # Small control input

        engine.set_state(q, v)
        a_control = engine.compute_control_acceleration(tau)

        assert a_control.size > 0, "Control acceleration should be non-empty"
        assert a_control.ndim == 1, "Control acceleration should be 1D"
        assert len(a_control) == engine.nv, "Control dim should match nv"

    def test_superposition_drift_plus_control_equals_full(
        self, engine: Any
    ) -> None:
        """Verify a_full = a_drift + a_control (superposition principle).

        This is the core acceptance test for the drift-control decomposition.
        The total acceleration should equal the sum of drift and control components.
        """
        # Set a non-trivial state with non-zero control
        q = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nq]
        v = np.array([0.5, -0.3, 0.0, 0.0, 0.0, 0.0])[: engine.nv]
        tau = np.ones(engine.nv) * 1.0

        engine.set_state(q, v)
        engine.set_control(tau)

        # Compute individual components
        a_drift = engine.compute_drift_acceleration()
        a_control = engine.compute_control_acceleration(tau)

        # Compute full acceleration using inverse dynamics
        M = engine.compute_mass_matrix()
        bias = engine.compute_bias_forces()
        a_full = np.linalg.solve(M, tau - bias)

        # Verify superposition
        a_decomposed = a_drift + a_control

        np.testing.assert_allclose(
            a_decomposed,
            a_full,
            atol=SUPERPOSITION_TOLERANCE,
            err_msg="Drift + Control should equal full acceleration (superposition)",
        )

    def test_zero_control_equals_drift(self, engine: Any) -> None:
        """With zero control, full acceleration should equal drift."""
        # Set a non-trivial state
        q = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nq]
        v = np.array([0.5, -0.3, 0.0, 0.0, 0.0, 0.0])[: engine.nv]
        tau = np.zeros(engine.nv)

        engine.set_state(q, v)
        engine.set_control(tau)

        # Compute drift and full acceleration
        a_drift = engine.compute_drift_acceleration()

        M = engine.compute_mass_matrix()
        bias = engine.compute_bias_forces()
        a_full = np.linalg.solve(M, -bias)  # tau = 0

        np.testing.assert_allclose(
            a_full,
            a_drift,
            atol=SUPERPOSITION_TOLERANCE,
            err_msg="With zero control, full acceleration should equal drift",
        )

    def test_drift_preserves_engine_state(self, engine: Any) -> None:
        """Computing drift acceleration should not modify engine state."""
        # Set initial state
        q_init = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nq]
        v_init = np.array([0.3, 0.4, 0.0, 0.0, 0.0, 0.0])[: engine.nv]
        engine.set_state(q_init, v_init)

        # Compute drift
        _ = engine.compute_drift_acceleration()

        # Verify state is unchanged
        q_after, v_after = engine.get_state()
        np.testing.assert_allclose(
            q_after,
            q_init,
            atol=1e-12,
            err_msg="Drift computation should not modify position state",
        )
        np.testing.assert_allclose(
            v_after,
            v_init,
            atol=1e-12,
            err_msg="Drift computation should not modify velocity state",
        )

    def test_control_preserves_engine_state(self, engine: Any) -> None:
        """Computing control acceleration should not modify engine state."""
        # Set initial state
        q_init = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nq]
        v_init = np.array([0.3, 0.4, 0.0, 0.0, 0.0, 0.0])[: engine.nv]
        engine.set_state(q_init, v_init)

        # Compute control acceleration
        tau = np.ones(engine.nv) * 0.5
        _ = engine.compute_control_acceleration(tau)

        # Verify state is unchanged
        q_after, v_after = engine.get_state()
        np.testing.assert_allclose(
            q_after,
            q_init,
            atol=1e-12,
            err_msg="Control computation should not modify position state",
        )
        np.testing.assert_allclose(
            v_after,
            v_init,
            atol=1e-12,
            err_msg="Control computation should not modify velocity state",
        )

    def test_control_linearity(self, engine: Any) -> None:
        """Control acceleration should be linear in tau.

        a_control(tau1 + tau2) = a_control(tau1) + a_control(tau2)
        """
        q = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nq]
        v = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])[: engine.nv]
        engine.set_state(q, v)

        tau1 = np.ones(engine.nv) * 0.5
        tau2 = np.ones(engine.nv) * 0.3

        a1 = engine.compute_control_acceleration(tau1)
        a2 = engine.compute_control_acceleration(tau2)
        a_combined = engine.compute_control_acceleration(tau1 + tau2)

        np.testing.assert_allclose(
            a_combined,
            a1 + a2,
            atol=SUPERPOSITION_TOLERANCE,
            err_msg="Control acceleration should be linear in tau",
        )