"""Strict Protocol Tests for A+ Physics Engines.

Verifies that all physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite, Pendulum)
strictly adhere to the updated PhysicsEngine protocol.

This test file aggressively mocks all external physics libraries (mujoco, pydrake, pinocchio, opensim, gym)
to ensure it can run in any CI environment to verify LOGIC and PROTOCOL compliance without
needing heavy binary dependencies.
"""

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# --- Global Mocking Setup ---
# We must mock these libs BEFORE importing the engines, because some engines
# (like MuJoCo) import them at the top level without try/except guards.

# Create Mocks
mock_mujoco = MagicMock()
mock_pydrake = MagicMock()
mock_pinocchio = MagicMock()
mock_opensim = MagicMock()
mock_gym = MagicMock()
mock_myosuite = MagicMock()

# Setup specific mock behaviors required for import time or basic init
mock_mujoco.MjModel = MagicMock()
mock_mujoco.MjData = MagicMock()

# Apply patches to sys.modules
module_patches = {
    "mujoco": mock_mujoco,
    "mujoco._structs": mock_mujoco,  # Often needed for type checks
    "pydrake": mock_pydrake,
    "pydrake.all": mock_pydrake,
    "pydrake.multibody": mock_pydrake,
    "pydrake.multibody.parsing": mock_pydrake,
    "pydrake.multibody.plant": mock_pydrake,
    "pydrake.systems": mock_pydrake,
    "pydrake.systems.framework": mock_pydrake,
    "pydrake.systems.analysis": mock_pydrake,
    "pydrake.math": mock_pydrake,
    "pinocchio": mock_pinocchio,
    "opensim": mock_opensim,
    "gym": mock_gym,
    "myosuite": mock_myosuite,
}

# Ensure DiagramBuilder is available via two paths depending on how it's imported
# If 'from pydrake.systems.framework import DiagramBuilder', and 'pydrake.systems.framework' is mock_pydrake:
mock_pydrake.DiagramBuilder = MagicMock()
# If accessed via attribute:
mock_pydrake.systems.framework.DiagramBuilder = mock_pydrake.DiagramBuilder

# --- Imports with Patch Context ---
# Use a context manager so patches apply only during engine imports.
# NOTE: Do NOT use importlib.reload() here as it can cause numpy to be
# reloaded, corrupting pandas' C API bindings in later tests.

with patch.dict("sys.modules", module_patches):
    from engines.physics_engines.myosuite.python.myosuite_physics_engine import (  # noqa: E402
        MyoSuitePhysicsEngine,
    )
    from engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E402
        OpenSimPhysicsEngine,
    )
    from engines.physics_engines.pendulum.python.pendulum_physics_engine import (  # noqa: E402
        PendulumPhysicsEngine,
    )

TEST_LINEAR_VAL = 1.0
TEST_ANGULAR_VAL = 2.0


# --- Test Classes ---


class TestMuJoCoStrict:
    def setup_method(self):
        """Enforce strict patching via direct dependency injection."""
        # 1. Ensure the module is loaded (using whatever state sys.modules is in)
        import engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine as mod

        # 2. Force the module's 'mujoco' reference to be OUR mock
        # This bypasses reload() and avoids triggering real imports or identity mismatches
        self.original_mujoco = getattr(mod, "mujoco", None)
        setattr(mod, "mujoco", mock_mujoco)  # noqa: B010
        # 3. Capture the class from THIS specific module version
        self.MuJoCoPhysicsEngine = mod.MuJoCoPhysicsEngine
        self.mod = mod

    def teardown_method(self):
        # Restore original if needed (though we mostly don't care in strict/mocked env)
        if hasattr(self, "original_mujoco"):
            setattr(self.mod, "mujoco", self.original_mujoco)  # noqa: B010

    def test_jacobian_standardization_mocked(self):
        """Verify compute_jacobian returns standard suite format [Angular; Linear] for spatial."""
        # Use the class from the patched module
        engine = self.MuJoCoPhysicsEngine()
        engine.model = MagicMock()
        engine.data = MagicMock()
        # Ensure we attach the mocks to the same object the engine is using
        engine.model.nv = 6

        # Mock mj_jacBody to return known values
        def side_effect_jac(model, data, jac_linear, jac_angular, body_id):
            jac_linear.fill(TEST_LINEAR_VAL)  # Linear (MuJoCo spec: jacp)
            jac_angular.fill(TEST_ANGULAR_VAL)  # Angular (MuJoCo spec: jacr)

        mock_mujoco.mj_jacBody.side_effect = side_effect_jac

        # Ensure mj_name2id returns valid id
        mock_mujoco.mj_name2id.return_value = 1

        jac = engine.compute_jacobian("foo")

        assert jac is not None
        assert "linear" in jac
        assert "angular" in jac
        assert "spatial" in jac

        # Check Spatial Stacking: [Angular; Linear]
        spatial = jac["spatial"]
        assert spatial.shape == (6, 6)
        # Top 3 rows -> Angular (2.0)
        np.testing.assert_allclose(
            spatial[:3, :], TEST_ANGULAR_VAL, err_msg="Top rows must be angular"
        )
        # Bottom 3 rows -> Linear (1.0)
        np.testing.assert_allclose(
            spatial[3:, :], 1.0, err_msg="Bottom rows must be linear"
        )

    def test_get_sensors_implemented(self):
        engine = self.MuJoCoPhysicsEngine()
        assert hasattr(engine, "get_sensors"), "MuJoCo must implement get_sensors"

        engine.model = MagicMock()
        engine.data = MagicMock()
        engine.model.nsensor = 1
        mock_mujoco.mj_id2name.return_value = "sensor_0"
        engine.data.sensordata = [0.123]

        sensors = engine.get_sensors()
        assert sensors["sensor_0"] == 0.123





class TestOpenSimStrict:
    def test_inverse_dynamics_implemented(self):
        engine = OpenSimPhysicsEngine()
        engine._model = MagicMock()  # type: ignore
        engine._state = MagicMock()  # type: ignore

        # Provide correct speeds/coords
        engine._model.getNumSpeeds.return_value = 2  # type: ignore
        engine._model.getNumCoordinates.return_value = 2  # type: ignore

        # Mock vector interaction for udot input
        # No error raised

        # Mock solver
        mock_solver_inst = MagicMock()
        mock_vec_out = MagicMock()
        mock_vec_out.get.side_effect = [10.0, 20.0]
        mock_solver_inst.solve.return_value = mock_vec_out

        mock_opensim.InverseDynamicsSolver.return_value = mock_solver_inst

        tau = engine.compute_inverse_dynamics(np.array([1.0, 1.0]))

        assert len(tau) == 2
        # Check values
        assert tau[0] == 10.0
        assert tau[1] == 20.0


class TestMyoSuiteStrict:
    def test_loading_uses_gym(self):
        """MyoSuite should assume path is an Env ID and load via gym."""
        engine = MyoSuitePhysicsEngine()

        # Setup mock env
        mock_env = MagicMock()
        # Mock underlying sim
        mock_env.sim = MagicMock()
        mock_gym.make.return_value = mock_env

        engine.load_from_path("myoElbow-v0")

        mock_gym.make.assert_called_with("myoElbow-v0")
        mock_env.reset.assert_called()

    def test_step_uses_sim_if_available(self):
        """MyoSuite should prefer underlying sim.step() if accessible."""
        engine = MyoSuitePhysicsEngine()
        mock_env = MagicMock()
        mock_sim = MagicMock()
        mock_sim.model.opt.timestep = 0.01

        # Mock Env structure where env.sim exists
        mock_env.sim = mock_sim
        mock_gym.make.return_value = mock_env

        engine.load_from_path("foo")
        assert engine.sim == mock_sim

        engine.step(dt=0.2)  # Override dt

        # Should set timestep, step, restore
        assert mock_sim.step.called
        assert mock_sim.model.opt.timestep == 0.01  # Restored


class TestPendulumStrict:
    def test_protocol_methods(self):
        engine = PendulumPhysicsEngine()
        engine.reset()

        # Test basic step
        engine.step(0.01)

        # Test get_state format
        q, v = engine.get_state()
        assert isinstance(q, np.ndarray)
        assert len(q) == 2
