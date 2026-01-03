from __future__ import annotations

import importlib
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# We used to mock at module level, but that restores sys.modules before tests run.
# Now we will rely on setUp to patch and import the engine.


class TestMuJoCoProtocol(unittest.TestCase):
    def setUp(self):
        # 1. Patch sys.modules to mock 'mujoco'
        self.patcher = patch.dict("sys.modules", {"mujoco": MagicMock()})
        self.patcher.start()

        # 2. Setup the mock
        import mujoco

        self.mock_mujoco = mujoco
        self.mock_mujoco.MjModel.from_xml_path = MagicMock()
        self.mock_mujoco.MjData = MagicMock()

        # 3. Import (and force reload) the engine module to ensure it binds to our mock
        # We must reload because previous tests might have loaded the real/other version
        import engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine as physics_module

        importlib.reload(physics_module)

        # 4. Get the class
        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
            MuJoCoPhysicsEngine,
        )

        self.MuJoCoPhysicsEngine = MuJoCoPhysicsEngine

        # 5. Instantiate
        self.engine = self.MuJoCoPhysicsEngine()

        # Mock model and data structure
        self.mock_model = MagicMock()
        self.mock_model.nu = 2
        self.mock_model.nv = 3
        self.mock_model.opt = MagicMock()
        self.mock_model.opt.timestep = 0.01

        self.mock_data = MagicMock()
        self.mock_data.ctrl = np.zeros(2)
        self.mock_data.qpos = np.zeros(3)
        self.mock_data.qvel = np.zeros(3)

        self.engine.model = self.mock_model
        self.engine.data = self.mock_data

    def tearDown(self):
        self.patcher.stop()

    def test_set_control_size_check(self):
        # Correct size
        u_correct = np.array([1.0, 2.0])
        self.engine.set_control(u_correct)
        # Should not raise

        # Incorrect size
        u_bad = np.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            self.engine.set_control(u_bad)

    def test_set_state_calls_forward(self):
        q = np.array([1, 2, 3])
        v = np.array([4, 5, 6])

        # Spy on the 'mj_forward' of the mock we created in setUp
        # Since 'physics_engine.py' imported 'mujoco', and we patched 'mujoco',
        # they share the same mock object.
        self.mock_mujoco.mj_forward.reset_mock()

        self.engine.set_state(q, v)

        self.mock_mujoco.mj_forward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
