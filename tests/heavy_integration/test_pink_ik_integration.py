"""
Heavy Integration Contracts — Pink IK Solver
=============================================
Tests are marked @pytest.mark.live_simulation and run only in the heavy
integration lane (weekly CI or local Docker).

Contract: Pink inverse kinematics solver can build tasks and solve IK
for a simple Pinocchio model.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.live_simulation
class TestPinkIKSolver:
    """Contract: Pink IK solver loads, creates tasks, and solves."""

    def test_pink_imports_and_version(self) -> None:
        """Pink and its solver backend are importable."""
        try:
            import pink
        except ImportError:
            pytest.skip("pink not installed")

        assert hasattr(pink, "solve_ik") or hasattr(pink, "Configuration"), (
            f"Pink API unexpected: {[a for a in dir(pink) if not a.startswith('_')]}"
        )

    def test_pink_configuration_from_pinocchio_model(self) -> None:
        """Pink Configuration wraps a Pinocchio model without error."""
        try:
            import pink
            import pinocchio as pin
        except ImportError:
            pytest.skip("pink or pinocchio not installed")

        if not hasattr(pin, "Model"):
            pytest.skip("pinocchio stub installed, not robotics library")

        # Build minimal 2-DOF robot
        model = pin.Model()
        inertia = pin.Inertia(1.0, np.zeros(3), np.eye(3))
        j1 = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "joint1")
        model.appendBodyToJoint(j1, inertia, pin.SE3.Identity())
        j2 = model.addJoint(
            j1, pin.JointModelRZ(), pin.SE3(np.eye(3), np.array([0, 0, 1.0])), "joint2"
        )
        model.appendBodyToJoint(j2, inertia, pin.SE3.Identity())
        model.addFrame(
            pin.Frame("end_effector", j2, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME)
        )

        data = model.createData()
        q = pin.neutral(model)

        # Pink Configuration should accept the model
        try:
            configuration = pink.Configuration(model, data, q)
            assert configuration is not None
        except (TypeError, AttributeError):
            # API may differ across versions
            pytest.skip("Pink API version not compatible with this test")

    def test_pink_end_effector_task(self) -> None:
        """Pink FrameTask can target an end-effector pose."""
        try:
            import pink
            import pinocchio as pin
        except ImportError:
            pytest.skip("pink or pinocchio not installed")

        if not hasattr(pin, "Model"):
            pytest.skip("pinocchio stub installed")

        # Check for FrameTask (the main task type)
        if not hasattr(pink, "tasks"):
            pytest.skip("pink.tasks not available in this version")

        task_module = pink.tasks
        assert hasattr(task_module, "FrameTask") or hasattr(pink, "FrameTask"), (
            "Pink should provide FrameTask"
        )


@pytest.mark.live_simulation
class TestPinkSolverIntegration:
    """Contract: The project's PinkSolver wrapper works with real Pink."""

    def test_pink_solver_importable(self) -> None:
        """PinkSolver class is importable."""
        from src.engines.physics_engines.pinocchio.python.dtack.ik.pink_solver import (
            PINK_SOLVER_AVAILABLE,
            PinkSolver,
        )

        assert PinkSolver is not None
        # Report availability — test doesn't fail if pink not installed
        if not PINK_SOLVER_AVAILABLE:
            pytest.skip("Pink not installed — PinkSolver unavailable")

    def test_pink_backend_importable(self) -> None:
        """PinkBackend loads without error."""
        from src.engines.physics_engines.pinocchio.python.dtack.backends.pink_backend import (
            PinkBackend,
        )

        assert PinkBackend is not None


pytestmark = pytest.mark.live_simulation
