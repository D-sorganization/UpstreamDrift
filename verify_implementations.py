
import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import os

# Mock modules if not available
sys.modules['gym'] = MagicMock()
sys.modules['myosuite'] = MagicMock()
sys.modules['opensim'] = MagicMock()

# Add repo root to path
sys.path.append(os.getcwd())

from engines.physics_engines.myosuite.python.myosuite_physics_engine import MyoSuitePhysicsEngine
from engines.physics_engines.opensim.python.opensim_physics_engine import OpenSimPhysicsEngine

class TestPhysicsEngines(unittest.TestCase):
    def test_myosuite_methods(self):
        engine = MyoSuitePhysicsEngine()
        # Mock internal sim
        engine.sim = MagicMock()
        engine.sim.data.qpos = np.zeros(10)
        engine.sim.data.qvel = np.zeros(10)
        engine.sim.data.ctrl = np.zeros(5)
        engine.sim.data.qacc = np.zeros(10) # acceleration
        engine.sim.model.nv = 10

        q = np.ones(10)
        v = np.ones(10)

        try:
            ztcf = engine.compute_ztcf(q, v)
            zvcf = engine.compute_zvcf(q)
            print("MyoSuite ZTCF/ZVCF implemented successfully")
        except NotImplementedError:
            self.fail("MyoSuite ZTCF/ZVCF raised NotImplementedError")

    def test_opensim_methods(self):
        engine = OpenSimPhysicsEngine()
        # Mock internal model/state
        engine._model = MagicMock()
        engine._state = MagicMock()
        engine._model.getNumCoordinates.return_value = 6
        engine._model.getNumSpeeds.return_value = 6
        engine._model.getNumControls.return_value = 3

        # Setup vectors for gets
        q_vec = MagicMock()
        q_vec.get.side_effect = lambda i: 0.0
        engine._state.getQ.return_value = q_vec

        u_vec = MagicMock()
        u_vec.get.side_effect = lambda i: 0.0
        engine._state.getU.return_value = u_vec

        udot_vec = MagicMock()
        udot_vec.get.side_effect = lambda i: 0.0
        engine._state.getUDot.return_value = udot_vec

        # Setup vector for updControls
        controls_vec = MagicMock()
        controls_vec.size.return_value = 3
        engine._model.updControls.return_value = controls_vec

        q = np.zeros(6)
        v = np.zeros(6)

        try:
            ztcf = engine.compute_ztcf(q, v)
            zvcf = engine.compute_zvcf(q)
            print("OpenSim ZTCF/ZVCF implemented successfully")
        except NotImplementedError:
            self.fail("OpenSim ZTCF/ZVCF raised NotImplementedError")

if __name__ == '__main__':
    unittest.main()
