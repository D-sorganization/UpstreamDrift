from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# We mock mujoco before importing the engine
# This simulates the presence of the library
with patch.dict("sys.modules", {"mujoco": MagicMock()}):
    import mujoco

    # Setup mock behavior
    mujoco.MjModel.from_xml_path = MagicMock()
    mujoco.MjData = MagicMock()

    from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (
        MuJoCoPhysicsEngine,
    )


class TestMuJoCoProtocol(unittest.TestCase):
    def setUp(self):
        self.engine = MuJoCoPhysicsEngine()
        # Mock model and data
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

        with patch.object(mujoco, "mj_forward") as mock_forward:
            self.engine.set_state(q, v)
            mock_forward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
