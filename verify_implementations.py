"""Verify physics engine implementations for ZTCF/ZVCF methods."""

import logging
import os
import sys
import unittest
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np

# Mock modules if not available
sys.modules["gym"] = MagicMock()
sys.modules["myosuite"] = MagicMock()
sys.modules["opensim"] = MagicMock()

# Add repo root to path
sys.path.append(os.getcwd())

# Import after mocking - E402 is unavoidable here
from engines.physics_engines.myosuite.python.myosuite_physics_engine import (  # noqa: E402
    MyoSuitePhysicsEngine,
)
from engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E402
    OpenSimPhysicsEngine,
)

logger = logging.getLogger(__name__)


class TestPhysicsEngines(unittest.TestCase):
    """Test physics engine ZTCF/ZVCF implementations."""

    def test_myosuite_methods(self) -> None:
        """Test MyoSuite engine has working ZTCF/ZVCF methods."""
        engine = MyoSuitePhysicsEngine()
        # Mock internal sim
        engine.sim = MagicMock()
        # Use cast(Any, ...) to avoid mypy errors with MagicMock assignment
        sim_data = cast(Any, MagicMock())
        engine.sim.data = sim_data
        sim_data.qpos = np.zeros(10)
        sim_data.qvel = np.zeros(10)
        sim_data.ctrl = np.zeros(5)
        sim_data.qacc = np.zeros(10)  # acceleration

        sim_model = cast(Any, MagicMock())
        engine.sim.model = sim_model
        sim_model.nv = 10

        q = np.ones(10)
        v = np.ones(10)

        try:
            _ztcf = engine.compute_ztcf(q, v)
            _zvcf = engine.compute_zvcf(q)
            logger.info("MyoSuite ZTCF/ZVCF implemented successfully")
        except NotImplementedError:
            self.fail("MyoSuite ZTCF/ZVCF raised NotImplementedError")

    def test_opensim_methods(self) -> None:
        """Test OpenSim engine has working ZTCF/ZVCF methods."""
        engine = OpenSimPhysicsEngine()
        # Mock internal model/state
        # Use cast(Any, ...) to bypass type checking for mocked private attributes
        model_mock = cast(Any, MagicMock())
        engine._model = model_mock

        state_mock = cast(Any, MagicMock())
        engine._state = state_mock

        model_mock.getNumCoordinates.return_value = 6
        model_mock.getNumSpeeds.return_value = 6
        model_mock.getNumControls.return_value = 3

        # Setup vectors for gets
        q_vec = MagicMock()
        q_vec.get.side_effect = lambda i: 0.0
        state_mock.getQ.return_value = q_vec

        u_vec = MagicMock()
        u_vec.get.side_effect = lambda i: 0.0
        state_mock.getU.return_value = u_vec

        udot_vec = MagicMock()
        udot_vec.get.side_effect = lambda i: 0.0
        state_mock.getUDot.return_value = udot_vec

        # Setup vector for updControls
        controls_vec = MagicMock()
        controls_vec.size.return_value = 3
        model_mock.updControls.return_value = controls_vec

        q = np.zeros(6)
        v = np.zeros(6)

        try:
            _ztcf = engine.compute_ztcf(q, v)
            _zvcf = engine.compute_zvcf(q)
            logger.info("OpenSim ZTCF/ZVCF implemented successfully")
        except NotImplementedError:
            self.fail("OpenSim ZTCF/ZVCF raised NotImplementedError")


if __name__ == "__main__":
    unittest.main()
