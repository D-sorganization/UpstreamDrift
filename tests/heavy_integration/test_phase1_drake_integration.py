"""Integration tests for Phase 1 Drake engine functionality.

This module tests the complete Drake engine integration including:
- Engine loading and initialization
- State management (reset, forward, step)
- Error handling and logging
- Integration with the engine manager

Refactored to use shared engine availability module (DRY principle).
"""

import unittest

import pytest
from src.shared.python.engine_core.engine_availability import DRAKE_AVAILABLE

if DRAKE_AVAILABLE:
    pass

# Only attributes that exist on the real pydrake MultibodyPlant may be listed
# (enforced against real pydrake by tests/engines/drake/test_drake_real_api_parity.py).
_PLANT_SPEC_ATTRS = [
    "Finalize",
    "time_step",
    "GetMyContextFromRoot",
    "SetDefaultState",
    "SetPositions",
    "SetVelocities",
    "GetPositions",
    "GetVelocities",
    "CalcMassMatrixViaInverseDynamics",
    "CalcInverseDynamics",
    "num_velocities",
]


if __name__ == "__main__":
    unittest.main()

pytestmark = pytest.mark.live_simulation
