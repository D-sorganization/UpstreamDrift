"""Integration tests for Phase 1 Drake engine functionality.

This module tests the complete Drake engine integration including:
- Engine loading and initialization
- State management (reset, forward, step)
- Error handling and logging
- Integration with the engine manager

Refactored to use shared engine availability module (DRY principle).
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.shared.python.core.contracts import PreconditionError
from src.shared.python.engine_core.engine_availability import DRAKE_AVAILABLE
from src.shared.python.engine_core.engine_manager import EngineManager, EngineType

if DRAKE_AVAILABLE:
    from pydrake.all import DiagramBuilder, Parser
    from pydrake.geometry import SceneGraph
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import Context, Diagram

# MultibodyPlant uses some undocumented Drake APIs (SetDefaultVelocities,
# MakeMultibodyForces) via type-ignore in the production code.  We enumerate
# the attributes the tests rely on so that the mock is constrained yet still
# permits those extra calls.
_PLANT_SPEC_ATTRS = [
    "Finalize",
    "time_step",
    "GetMyContextFromRoot",
    "SetDefaultPositions",
    "SetDefaultVelocities",
    "GetPositions",
    "GetVelocities",
    "CalcMassMatrixViaInverseDynamics",
    "CalcInverseDynamics",
    "num_velocities",
    "MakeMultibodyForces",
]


if __name__ == "__main__":
    unittest.main()

pytestmark = pytest.mark.live_simulation
