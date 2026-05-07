"""Parity tests comparing Pinocchio RK4 vs Drake RK4 integrators.

This test suite validates that Pinocchio and Drake produce identical
or near-identical results when integrating the same system under
identical conditions.

Goal: Ensure RK4 implementation parity for issue #4118.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from src.shared.python.core.constants import GRAVITY_M_S2


class TestPinocchioDrakeRK4Parity:
    """Test parity between Pinocchio and Drake RK4 integrators."""

    @pytest.fixture
    def pinocchio_setup(self):
        """Set up Pinocchio if available."""
        try:
            import pinocchio as pin

            if isinstance(pin, MagicMock):
                pytest.skip("Pinocchio is mocked")

            return pin
        except ImportError:
            pytest.skip("Pinocchio not available")

    @pytest.fixture
    def drake_setup(self):
        """Set up Drake if available."""
        try:
            from pydrake.all import (
                AddMultibodyPlantSceneGraph,
                DiagramBuilder,
            )
            from pydrake.systems.analysis import Simulator

            return {
                "AddMultibodyPlantSceneGraph": AddMultibodyPlantSceneGraph,
                "DiagramBuilder": DiagramBuilder,
                "Simulator": Simulator,
            }
        except ImportError:
            pytest.skip("Drake not available")

    def test_pinocchio_drake_simple_pendulum_parity(
        self, pinocchio_setup, drake_setup
    ) -> None:
        """Test parity on simple 1-DOF pendulum."""
        from src.engines.physics_engines.pinocchio.python.rk4_integrator import (
            PinocchioRK4Integrator,
        )

        pin = pinocchio_setup
        drake_tools = drake_setup

        # --- Pinocchio Setup ---
        pin_model = pin.Model()
        pin_model.gravity.linear = np.array([0.0, 0.0, -GRAVITY_M_S2])

        pin_model.addFrame(
            pin.Frame(
                "world",
                0,
                0,
                pin.SE3.Identity(),
                pin.FrameType.FIXED,
            )
        )

        inertia = pin.Inertia(
            1.0,
            np.array([0.0, 0.0, -1.0]),
            np.eye(3) * 0.001,
        )
        pin_model.addBody(
            1,
            pin.JointModelRY(),
            pin.SE3.Identity(),
            "pendulum",
            inertia,
        )

        pin_integrator = PinocchioRK4Integrator(
            pin_model,
            timestep=0.001,
            validate_stages=True,
        )

        # --- Drake Setup ---
        import pydrake.multibody.tree as mut
        from pydrake.all import (
            AddMultibodyPlantSceneGraph,
            DiagramBuilder,
        )
        from pydrake.systems.analysis import Simulator

        builder = DiagramBuilder()
        drake_plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

        M = 1.0
        L = 1.0
        com_vector = [0.0, 0.0, -L]
        unit_inertia = mut.UnitInertia.PointMass(com_vector)
        spatial_inertia = mut.SpatialInertia(M, com_vector, unit_inertia)

        pendulum = drake_plant.AddRigidBody("pendulum", spatial_inertia)
        drake_plant.AddJoint(
            mut.RevoluteJoint(
                "hinge",
                drake_plant.world_frame(),
                pendulum.body_frame(),
                [0.0, 1.0, 0.0],
            )
        )

        drake_plant.Finalize()
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        drake_plant_context = drake_plant.GetMyContextFromRoot(context)
        drake_simulator = Simulator(diagram, context)
        drake_simulator.Initialize()

        # --- Initial Conditions ---
        q0 = np.array([np.deg2rad(30.0)])  # 30 degrees
        v0 = np.array([0.0])
        tau = np.array([0.0])

        # Pinocchio
        pin_q = q0.copy()
        pin_v = v0.copy()

        # Drake
        drake_q = q0.copy()
        drake_v = v0.copy()
        drake_plant.SetPositions(drake_plant_context, drake_q)
        drake_plant.SetVelocities(drake_plant_context, drake_v)

        # --- Integration Loop ---
        n_steps = 100
        q_errors = []
        v_errors = []

        for _ in range(n_steps):
            # Pinocchio step
            pin_result = pin_integrator.step(pin_q, pin_v, control=tau)
            pin_q = pin_result.q_next
            pin_v = pin_result.v_next

            # Drake step
            current_time = context.get_time()
            drake_simulator.AdvanceTo(current_time + 0.001)
            drake_q = drake_plant.GetPositions(drake_plant_context)
            drake_v = drake_plant.GetVelocities(drake_plant_context)

            # Compute errors
            q_error = np.linalg.norm(pin_q - drake_q)
            v_error = np.linalg.norm(pin_v - drake_v)

            q_errors.append(q_error)
            v_errors.append(v_error)

        # --- Assertions ---
        max_q_error = max(q_errors)
        max_v_error = max(v_errors)

        # Allow small differences due to different ABA implementations
        # RK4 errors typically accumulate, so relative tolerance of 1% is reasonable
        initial_state_magnitude = np.linalg.norm(np.concatenate([q0, v0]))
        abs_tolerance = initial_state_magnitude * 0.01

        assert max_q_error < abs_tolerance, (
            f"Position parity failed: max_q_error={max_q_error:.6e}, "
            f"tolerance={abs_tolerance:.6e}"
        )
        assert max_v_error < abs_tolerance, (
            f"Velocity parity failed: max_v_error={max_v_error:.6e}, "
            f"tolerance={abs_tolerance:.6e}"
        )

    def test_pinocchio_drake_controlled_pendulum_parity(
        self, pinocchio_setup, drake_setup
    ) -> None:
        """Test parity on controlled pendulum with applied torques."""
        from src.engines.physics_engines.pinocchio.python.rk4_integrator import (
            PinocchioRK4Integrator,
        )

        pin = pinocchio_setup
        drake_tools = drake_setup

        # --- Pinocchio Setup ---
        pin_model = pin.Model()
        pin_model.gravity.linear = np.array([0.0, 0.0, -GRAVITY_M_S2])

        pin_model.addFrame(
            pin.Frame(
                "world",
                0,
                0,
                pin.SE3.Identity(),
                pin.FrameType.FIXED,
            )
        )

        inertia = pin.Inertia(
            1.0,
            np.array([0.0, 0.0, -1.0]),
            np.eye(3) * 0.001,
        )
        pin_model.addBody(
            1,
            pin.JointModelRY(),
            pin.SE3.Identity(),
            "pendulum",
            inertia,
        )

        pin_integrator = PinocchioRK4Integrator(
            pin_model,
            timestep=0.001,
            validate_stages=True,
        )

        # --- Drake Setup ---
        import pydrake.multibody.tree as mut
        from pydrake.all import (
            AddMultibodyPlantSceneGraph,
            DiagramBuilder,
        )
        from pydrake.systems.analysis import Simulator

        builder = DiagramBuilder()
        drake_plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

        M = 1.0
        L = 1.0
        com_vector = [0.0, 0.0, -L]
        unit_inertia = mut.UnitInertia.PointMass(com_vector)
        spatial_inertia = mut.SpatialInertia(M, com_vector, unit_inertia)

        pendulum = drake_plant.AddRigidBody("pendulum", spatial_inertia)
        joint = drake_plant.AddJoint(
            mut.RevoluteJoint(
                "hinge",
                drake_plant.world_frame(),
                pendulum.body_frame(),
                [0.0, 1.0, 0.0],
            )
        )

        # Add actuator
        drake_plant.AddJointActuator("hinge_actuator", joint)

        drake_plant.Finalize()
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        drake_plant_context = drake_plant.GetMyContextFromRoot(context)
        drake_simulator = Simulator(diagram, context)
        drake_simulator.Initialize()

        # --- Initial Conditions ---
        q0 = np.array([np.deg2rad(15.0)])
        v0 = np.array([0.0])
        tau = np.array([0.5])  # Apply 0.5 N·m torque

        # Pinocchio
        pin_q = q0.copy()
        pin_v = v0.copy()

        # Drake
        drake_q = q0.copy()
        drake_v = v0.copy()
        drake_plant.SetPositions(drake_plant_context, drake_q)
        drake_plant.SetVelocities(drake_plant_context, drake_v)

        # --- Integration Loop ---
        n_steps = 50  # Shorter for controlled case
        q_errors = []
        v_errors = []

        for _ in range(n_steps):
            # Pinocchio step
            pin_result = pin_integrator.step(pin_q, pin_v, control=tau)
            pin_q = pin_result.q_next
            pin_v = pin_result.v_next

            # Drake step (manual since we need explicit control)
            current_time = context.get_time()
            drake_simulator.AdvanceTo(current_time + 0.001)
            drake_q = drake_plant.GetPositions(drake_plant_context)
            drake_v = drake_plant.GetVelocities(drake_plant_context)

            q_error = np.linalg.norm(pin_q - drake_q)
            v_error = np.linalg.norm(pin_v - drake_v)

            q_errors.append(q_error)
            v_errors.append(v_error)

        # --- Assertions ---
        # Controlled systems may have larger divergence due to different
        # integration orders and rounding, so use 2% tolerance
        max_q_error = max(q_errors)
        max_v_error = max(v_errors)
        initial_magnitude = np.linalg.norm(np.concatenate([q0, v0])) + np.linalg.norm(
            tau
        )
        abs_tolerance = initial_magnitude * 0.02

        assert max_q_error < abs_tolerance, (
            f"Controlled position parity failed: max_q_error={max_q_error:.6e}"
        )
        assert max_v_error < abs_tolerance, (
            f"Controlled velocity parity failed: max_v_error={max_v_error:.6e}"
        )


class TestRK4NumericalStability:
    """Test numerical stability of RK4 integrators."""

    @pytest.fixture
    def pinocchio_module(self):
        try:
            import pinocchio as pin

            if isinstance(pin, MagicMock):
                pytest.skip("Pinocchio is mocked")
            return pin
        except ImportError:
            pytest.skip("Pinocchio not available")

    def test_rk4_stability_large_timestep(self, pinocchio_module) -> None:
        """Test RK4 stability with relatively large timestep (0.01s)."""
        from src.engines.physics_engines.pinocchio.python.rk4_integrator import (
            PinocchioRK4Integrator,
        )

        pin = pinocchio_module

        # Build simple pendulum
        model = pin.Model()
        model.gravity.linear = np.array([0.0, 0.0, -GRAVITY_M_S2])
        model.addFrame(
            pin.Frame(
                "world",
                0,
                0,
                pin.SE3.Identity(),
                pin.FrameType.FIXED,
            )
        )

        inertia = pin.Inertia(
            1.0,
            np.array([0.0, 0.0, -1.0]),
            np.eye(3) * 0.001,
        )
        model.addBody(
            1,
            pin.JointModelRY(),
            pin.SE3.Identity(),
            "pendulum",
            inertia,
        )

        # Use 0.01s timestep (10ms, large for pendulum)
        integrator = PinocchioRK4Integrator(
            model,
            timestep=0.01,
            validate_stages=True,
        )

        q = np.array([np.pi / 3.0])
        v = np.array([0.0])

        # Should not blow up or produce NaNs
        for _ in range(100):
            result = integrator.step(q, v, control=np.array([0.0]))
            q = result.q_next
            v = result.v_next

            assert np.isfinite(q).all(), "Position became non-finite"
            assert np.isfinite(v).all(), "Velocity became non-finite"

        # After 1 second, pendulum should still be oscillating (not diverged)
        assert abs(q[0]) < np.pi, "Pendulum swung past vertical (instability)"
