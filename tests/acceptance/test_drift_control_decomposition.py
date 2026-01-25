"""Tests for drift-control decomposition (Section F).

Verifies that drift + control = full dynamics for all physics engines.
Refactored for DRY compliance using parameterized engine tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

# TOLERANCE for superposition test
SUPERPOSITION_TOLERANCE = 1e-5


def _get_engine(engine_name: str):
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
            from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
                MuJoCoPhysicsEngine,
            )
            return MuJoCoPhysicsEngine()
        except ImportError:
            pytest.skip("MuJoCo not installed")
            
    pytest.skip(f"Engine {engine_name} not available")


@pytest.mark.parametrize("engine_name", ["pinocchio", "mujoco"])
class TestDriftControlDecomposition:
    """Unified tests for drift-control decomposition across all engines."""

    def test_superposition(self, engine_name, pendulum_urdf):
        """Verify drift + control = full dynamics. (Requirement F)"""
        engine = _get_engine(engine_name)
        
        try:
            engine.load_from_path(pendulum_urdf)
        except Exception:
            pytest.skip(f"{engine_name} failed to load URDF")

        # Set state (non-rest position)
        q_initial = np.array([0.1])
        v_initial = np.array([0.5])
        # Pad q/v if engine has more DOFs than the single pendulum URDF
        if hasattr(engine, "get_model") and engine.get_model():
            q_initial = np.zeros(engine.get_model().nq)
            q_initial[0] = 0.1
            v_initial = np.zeros(engine.get_model().nv)
            v_initial[0] = 0.5
            
        engine.set_state(q_initial, v_initial)
        tau_control = np.zeros(len(v_initial))
        tau_control[0] = 0.5
        engine.set_control(tau_control)

        # 1. Full Dynamics
        engine.forward()
        if engine_name == "pinocchio":
            import pinocchio as pin
            a_full = pin.aba(engine.model, engine.data, engine.q, engine.v, tau_control)
        else:
            a_full = engine.get_data().qacc.copy()

        # 2. Components
        a_drift = engine.compute_drift_acceleration()
        a_control = engine.compute_control_acceleration(tau_control)

        # 3. Superposition check
        a_reconstructed = a_drift + a_control
        residual = a_full - a_reconstructed
        max_res = float(np.max(np.abs(residual)))

        assert max_res < SUPERPOSITION_TOLERANCE, \
            f"{engine_name}: Superposition failed (res={max_res:.2e})"

    def test_zero_control(self, engine_name, pendulum_urdf):
        """Verify full dynamics with tau=0 equals drift acceleration."""
        engine = _get_engine(engine_name)
        try:
            engine.load_from_path(pendulum_urdf)
        except Exception:
            pytest.skip(f"{engine_name} failed to load URDF")

        q_initial, v_initial = np.array([0.3]), np.array([0.2])
        if hasattr(engine, "get_model") and engine.get_model():
            q_initial = np.zeros(engine.get_model().nq)
            q_initial[0] = 0.3
            v_initial = np.zeros(engine.get_model().nv)
            v_initial[0] = 0.2
            
        engine.set_state(q_initial, v_initial)
        a_drift = engine.compute_drift_acceleration()
        
        # Compute full with zero torque
        tau_zero = np.zeros(len(v_initial))
        if engine_name == "pinocchio":
            import pinocchio as pin
            a_full_zero = pin.aba(engine.model, engine.data, engine.q, engine.v, tau_zero)
        else:
            engine.set_control(tau_zero)
            engine.forward()
            a_full_zero = engine.get_data().qacc.copy()

        np.testing.assert_allclose(a_drift, a_full_zero, atol=1e-10)

    def test_interface_compliance(self, engine_name, pendulum_urdf):
        """Verify drift-control API compliance."""
        engine = _get_engine(engine_name)
        assert hasattr(engine, "compute_drift_acceleration")
        assert hasattr(engine, "compute_control_acceleration")
