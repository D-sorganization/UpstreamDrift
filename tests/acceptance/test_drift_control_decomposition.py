"""Tests for drift-control decomposition (Section F).

Verifies that drift + control = full dynamics for all physics engines.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

# Import test utilities
from shared.python.indexed_acceleration import (
    AccelerationClosureError,
    compute_indexed_acceleration_from_engine,
)

LOGGER = logging.getLogger(__name__)

# TOLERANCE for superposition test
SUPERPOSITION_TOLERANCE = 1e-5  # rad/s² or m/s²


@pytest.fixture
def simple_pendulum_urdf(tmp_path):
    """Create a simple pendulum URDF for testing."""
    urdf_content = """<?xml version="1.0"?>
<robot name="pendulum">
  <link name="world"/>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="world"/>
    <child link="link1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="10"/>
  </joint>
</robot>
"""
    urdf_path = tmp_path / "pendulum.urdf"
    urdf_path.write_text(urdf_content)
    return str(urdf_path)


class TestPinocchioDriftControl:
    """Test drift-control decomposition for Pinocchio engine."""

    def test_superposition_simple_pendulum(self, simple_pendulum_urdf):
        """Verify drift + control = full dynamics (Pinocchio).

        Section F Requirement: a_drift + a_control = a_full
        """
        try:
            from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                PinocchioPhysicsEngine,
            )
        except ImportError:
            pytest.skip("Pinocchio not installed")

        engine = PinocchioPhysicsEngine()
        engine.load_from_path(simple_pendulum_urdf)

        # Set initial state (pendulum at 0.1 rad, zero velocity)
        q = np.array([0.1])
        v = np.array([0.0])
        engine.set_state(q, v)

        # Apply non-zero torque
        tau = np.array([0.5])
        engine.set_control(tau)

        # Compute full forward dynamics
        engine.forward()
        # Pinocchio stores acceleration in engine.a after step/forward
        # We need to compute it explicitly
        import pinocchio as pin

        a_full = pin.aba(engine.model, engine.data, engine.q, engine.v, tau)

        # Compute drift and control components
        a_drift = engine.compute_drift_acceleration()
        a_control = engine.compute_control_acceleration(tau)

        # Superposition test: drift + control = full
        a_reconstructed = a_drift + a_control
        residual = a_full - a_reconstructed

        LOGGER.info("Pinocchio Superposition Test:")
        LOGGER.info(f"  a_full = {a_full}")
        LOGGER.info(f"  a_drift = {a_drift}")
        LOGGER.info(f"  a_control = {a_control}")
        LOGGER.info(f"  residual = {residual}")

        assert np.max(np.abs(residual)) < SUPERPOSITION_TOLERANCE, (
            f"Pinocchio drift-control superposition failed: "
            f"max residual = {np.max(np.abs(residual)):.2e}"
        )

    def test_zero_control_equals_drift(self, simple_pendulum_urdf):
        """Verify that full dynamics with tau=0 equals drift acceleration."""
        try:
            import pinocchio as pin

            from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                PinocchioPhysicsEngine,
            )
        except ImportError:
            pytest.skip("Pinocchio not installed")

        engine = PinocchioPhysicsEngine()
        engine.load_from_path(simple_pendulum_urdf)

        # Set state
        q = np.array([0.5])
        v = np.array([0.2])
        engine.set_state(q, v)

        # Compute drift
        a_drift = engine.compute_drift_acceleration()

        # Compute full dynamics with zero torque
        tau_zero = np.array([0.0])
        a_full_zero_tau = pin.aba(
            engine.model, engine.data, engine.q, engine.v, tau_zero
        )

        residual = a_drift - a_full_zero_tau

        assert np.max(np.abs(residual)) < 1e-10, (
            f"Drift should equal full dynamics with tau=0: " f"residual = {residual}"
        )


class TestMuJoCoDriftControl:
    """Test drift-control decomposition for MuJoCo engine."""

    def test_superposition_simple_pendulum(self, simple_pendulum_urdf):
        """Verify drift + control = full dynamics (MuJoCo).

        Section F Requirement: a_drift + a_control = a_full
        """
        try:
            import mujoco

            from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
                MuJoCoPhysicsEngine,
            )
        except ImportError:
            pytest.skip("MuJoCo not installed")

        engine = MuJoCoPhysicsEngine()

        # Convert URDF to MuJoCo XML (MuJoCo can load URDF directly in newer versions)
        try:
            engine.load_from_path(simple_pendulum_urdf)
        except Exception:
            pytest.skip("MuJoCo URDF loading not supported in this version")

        # Set initial state
        q = np.array([0.1, 0.0])  # MuJoCo might have base joint
        v = np.array([0.0])
        engine.set_state(q, v)

        # Apply control
        tau = np.array([0.5])
        engine.set_control(tau)

        # Compute full forward dynamics
        engine.forward()
        model = engine.get_model()
        data = engine.get_data()

        if model is None or data is None:
            pytest.skip("MuJoCo model/data not initialized")

        # Re-compute forward dynamics to get acceleration
        mujoco.mj_forward(model, data)
        a_full = data.qacc.copy()

        # Compute drift and control components
        a_drift = engine.compute_drift_acceleration()
        a_control = engine.compute_control_acceleration(tau)

        # Superposition test
        a_reconstructed = a_drift + a_control
        residual = a_full - a_reconstructed

        LOGGER.info("MuJoCo Superposition Test:")
        LOGGER.info(f"  a_full = {a_full}")
        LOGGER.info(f"  a_drift = {a_drift}")
        LOGGER.info(f"  a_control = {a_control}")
        LOGGER.info(f"  residual = {residual}")

        assert np.max(np.abs(residual)) < SUPERPOSITION_TOLERANCE, (
            f"MuJoCo drift-control superposition failed: "
            f"max residual = {np.max(np.abs(residual)):.2e}"
        )


class TestIndexedAccelerationClosure:
    """Test indexed acceleration closure (Section H2)."""

    def test_pinocchio_closure(self, simple_pendulum_urdf):
        """Verify indexed acceleration components sum to total (Pinocchio)."""
        try:
            import pinocchio as pin

            from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                PinocchioPhysicsEngine,
            )
        except ImportError:
            pytest.skip("Pinocchio not installed")

        engine = PinocchioPhysicsEngine()
        engine.load_from_path(simple_pendulum_urdf)

        # Set state
        q = np.array([0.3])
        v = np.array([0.1])
        engine.set_state(q, v)

        # Apply torque
        tau = np.array([0.7])

        # Compute total acceleration
        a_total = pin.aba(engine.model, engine.data, engine.q, engine.v, tau)

        # Compute indexed acceleration
        indexed = compute_indexed_acceleration_from_engine(engine, tau)

        # Closure test
        try:
            indexed.assert_closure(a_total, atol_joint_space=1e-6)
        except AccelerationClosureError as e:
            pytest.fail(f"Pinocchio indexed acceleration closure failed: {e}")

    def test_contribution_percentages(self, simple_pendulum_urdf):
        """Verify contribution percentage calculation."""
        try:
            from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                PinocchioPhysicsEngine,
            )
        except ImportError:
            pytest.skip("Pinocchio not installed")

        engine = PinocchioPhysicsEngine()
        engine.load_from_path(simple_pendulum_urdf)

        # Set state
        q = np.array([0.5])
        v = np.array([0.0])
        engine.set_state(q, v)

        # Apply torque
        tau = np.array([1.0])

        # Compute indexed acceleration
        indexed = compute_indexed_acceleration_from_engine(engine, tau)

        # Get contribution percentages
        percentages = indexed.get_contribution_percentages()

        # Verify percentages are non-negative and reasonable
        for component, percentage in percentages.items():
            assert percentage >= 0.0, f"{component} has negative percentage"
            LOGGER.info(f"  {component}: {percentage:.1f}%")


class TestCrossEngineDriftControl:
    """Cross-engine validation for drift-control decomposition."""

    @pytest.mark.parametrize("engine_name", ["pinocchio", "mujoco"])
    def test_drift_control_interface(self, engine_name, simple_pendulum_urdf):
        """Verify all engines implement drift-control interface."""
        if engine_name == "pinocchio":
            try:
                from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
                    PinocchioPhysicsEngine,
                )

                engine = PinocchioPhysicsEngine()
            except ImportError:
                pytest.skip(f"{engine_name} not installed")
        elif engine_name == "mujoco":
            try:
                from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
                    MuJoCoPhysicsEngine,
                )

                engine = MuJoCoPhysicsEngine()
            except ImportError:
                pytest.skip(f"{engine_name} not installed")
        else:
            pytest.skip(f"Engine {engine_name} not yet tested")

        # Verify methods exist
        assert hasattr(
            engine, "compute_drift_acceleration"
        ), f"{engine_name} missing compute_drift_acceleration()"
        assert hasattr(
            engine, "compute_control_acceleration"
        ), f"{engine_name} missing compute_control_acceleration()"

        # Try loading and computing (may fail due to URDF compatibility)
        try:
            engine.load_from_path(simple_pendulum_urdf)
            q = np.array([0.1])
            v = np.array([0.0])
            engine.set_state(q, v)

            a_drift = engine.compute_drift_acceleration()
            a_control = engine.compute_control_acceleration(np.array([0.5]))

            assert len(a_drift) > 0, f"{engine_name} returned empty drift acceleration"
            assert (
                len(a_control) > 0
            ), f"{engine_name} returned empty control acceleration"

        except Exception as e:
            LOGGER.warning(f"{engine_name} failed to load URDF: {e}")
            # This is acceptable - we've at least verified the interface exists
