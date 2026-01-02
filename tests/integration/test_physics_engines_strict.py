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
mock_mujoco.MjModel = MagicMock
mock_mujoco.MjData = MagicMock

# Apply patches to sys.modules
module_patches = {
    "mujoco": mock_mujoco,
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

# --- Imports with Patch Context ---
# Use a context manager so patches apply only during engine imports,
# avoiding long-lived module-level patchers that can leak between tests.
with patch.dict("sys.modules", module_patches):
    from engines.physics_engines.drake.python.drake_physics_engine import (  # noqa: E402
        DrakePhysicsEngine,
    )
    from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (  # noqa: E402
        MuJoCoPhysicsEngine,
    )
    from engines.physics_engines.myosuite.python.myosuite_physics_engine import (  # noqa: E402
        MyoSuitePhysicsEngine,
    )
    from engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E402
        OpenSimPhysicsEngine,
    )
    from engines.physics_engines.pendulum.python.pendulum_physics_engine import (  # noqa: E402
        PendulumPhysicsEngine,
    )
    from engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (  # noqa: E402
        PinocchioPhysicsEngine,
    )

TEST_LINEAR_VAL = 1.0
TEST_ANGULAR_VAL = 2.0


# --- Test Classes ---


class TestMuJoCoStrict:
    def test_jacobian_standardization_mocked(self):
        """Verify compute_jacobian returns standard suite format [Angular; Linear] for spatial."""
        engine = MuJoCoPhysicsEngine()
        engine.model = MagicMock()
        engine.data = MagicMock()
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
        engine = MuJoCoPhysicsEngine()
        assert hasattr(engine, "get_sensors"), "MuJoCo must implement get_sensors"

        engine.model = MagicMock()
        engine.data = MagicMock()
        engine.model.nsensor = 1
        mock_mujoco.mj_id2name.return_value = "sensor_0"
        engine.data.sensordata = [0.123]

        sensors = engine.get_sensors()
        assert sensors["sensor_0"] == 0.123


class TestDrakeStrict:
    def test_jacobian_standardization_mocked(self):
        engine = DrakePhysicsEngine()
        # Mock internals set by AddMultibodyPlantSceneGraph
        engine.plant = MagicMock()
        engine.plant_context = MagicMock()

        # Mock output of CalcJacobianSpatialVelocity
        # Drake returns (w, v) -> Angular, Linear
        J_fake = np.zeros((6, 2))
        J_fake[:3, :] = TEST_ANGULAR_VAL  # Angular
        J_fake[3:, :] = TEST_LINEAR_VAL  # Linear
        engine.plant.CalcJacobianSpatialVelocity.return_value = J_fake
        # Ensure body lookup works
        engine.plant.GetBodyByName.return_value = MagicMock()

        jac = engine.compute_jacobian("foo")
        assert jac is not None

        spatial = jac["spatial"]
        # Drake engine should pass J through directly as it is already [Angular; Linear]
        np.testing.assert_allclose(spatial[:3, :], TEST_ANGULAR_VAL)
        np.testing.assert_allclose(spatial[3:, :], TEST_LINEAR_VAL)

    def test_reset_protection(self, caplog):
        """Drake reset should warn if uninitialized."""
        engine = DrakePhysicsEngine()
        engine.context = None  # Force uninitialized

        with caplog.at_level(logging.WARNING):
            engine.reset()

        assert "Attempted to reset Drake engine before initialization." in caplog.text


class TestPinocchioStrict:
    def test_jacobian_standardization_mocked(self):
        engine = PinocchioPhysicsEngine()
        engine.model = MagicMock()
        engine.data = MagicMock()

        # Mock frame lookup success
        engine.model.existFrame.return_value = True
        engine.model.getFrameId.return_value = 1

        # Pinocchio returns [Linear; Angular] natively from getFrameJacobian
        J_native = np.zeros((6, 2))
        J_native[:3, :] = TEST_LINEAR_VAL  # Linear (top)
        J_native[3:, :] = TEST_ANGULAR_VAL  # Angular (bottom)

        mock_pinocchio.getFrameJacobian.return_value = J_native

        jac = engine.compute_jacobian("foo")
        assert jac is not None

        # We upgraded Pinocchio to re-stack to [Angular; Linear] (MuJoCo/Drake standard)
        spatial = jac["spatial"]
        # Top 3 should now be Angular (2.0)
        np.testing.assert_allclose(
            spatial[:3, :],
            TEST_ANGULAR_VAL,
            err_msg="Pinocchio spatial top should be re-stacked to Angular",
        )
        # Bottom 3 should now be Linear (1.0)
        np.testing.assert_allclose(
            spatial[3:, :],
            TEST_LINEAR_VAL,
            err_msg="Pinocchio spatial bottom should be re-stacked to Linear",
        )

    def test_compute_jacobian_missing_frame_and_body(self):
        """Test behavior when neither frame nor body exists."""
        engine = PinocchioPhysicsEngine()
        engine.model = MagicMock()
        engine.model.existFrame.return_value = False
        engine.model.existBodyName.return_value = False

        jac = engine.compute_jacobian("missing_link")
        assert jac is None


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
