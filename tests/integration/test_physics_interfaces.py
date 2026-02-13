"""Verify physics engine implementations for ZTCF/ZVCF methods."""

import sys
from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.logging_pkg.logging_config import get_logger

# Mock modules using patch.dict (auto-cleans) before importing engines.
# The mocks must also be present at test execution time because the engine
# modules reference gym/myosuite/opensim at runtime, not just import time.
_MOCKED_MODULES = {
    "gym": MagicMock(),
    "myosuite": MagicMock(),
    "opensim": MagicMock(),
}

with patch.dict(sys.modules, _MOCKED_MODULES):
    from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (  # noqa: E402
        MyoSuitePhysicsEngine,
    )
    from src.engines.physics_engines.opensim.python import (  # noqa: E402
        opensim_physics_engine as _osim_module,
    )
    from src.engines.physics_engines.opensim.python.opensim_physics_engine import (  # noqa: E402
        OpenSimPhysicsEngine,
    )

# Re-install mocks for test execution (auto-cleaned by patch.dict on stop).
# Also patch the module-level opensim attribute on the engine module so
# runtime references to opensim.Vector etc. work.
_runtime_patcher = patch.dict(sys.modules, _MOCKED_MODULES)
_mock_opensim = _MOCKED_MODULES["opensim"]


def setup_module(module):
    _runtime_patcher.start()
    _osim_module.opensim = _mock_opensim


def teardown_module(module):
    _runtime_patcher.stop()
    _osim_module.opensim = None


logger = get_logger(__name__)


class TestPhysicsEngines:
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
            pytest.fail("MyoSuite ZTCF/ZVCF raised NotImplementedError")

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
            pytest.fail("OpenSim ZTCF/ZVCF raised NotImplementedError")
