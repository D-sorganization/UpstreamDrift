"""Integration tests for Phase 1 Drake engine functionality.

This module tests the complete Drake engine integration including:
- Engine loading and initialization
- State management (reset, forward, step)
- Error handling and logging
- Integration with the engine manager

Refactored to use shared engine availability module (DRY principle).
"""

import unittest

from src.shared.python.engine_core.engine_availability import DRAKE_AVAILABLE

if DRAKE_AVAILABLE:
    pass

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
