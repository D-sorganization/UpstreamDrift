"""Tests for OpenSim integration (Section J).

Verifies:
- Hill-type muscle model functionality
- Activation → force →torque pipeline
- Muscle-induced acceleration analysis
- Grip wrapping geometry

Refactored to use shared engine availability module (DRY principle).
"""

from __future__ import annotations

from typing import Any

import pytest
from src.shared.python.engine_core.engine_availability import (
    OPENSIM_AVAILABLE,
    skip_if_unavailable,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# Skip entire module if OpenSim not available
pytestmark = skip_if_unavailable("opensim")

if OPENSIM_AVAILABLE:
    import opensim


@pytest.fixture
def simple_arm_model() -> tuple[Any, Any]:
    """Create a simple arm model with muscles for testing."""
    if not OPENSIM_AVAILABLE:
        pytest.skip("OpenSim not installed")

    # Create a simple arm model
    model = opensim.Model()
    model.setName("SimpleArm")

    # Ground body
    ground = model.getGround()

    # Upper arm body
    upper_arm = opensim.Body(
        "upperarm",
        1.0,  # mass [kg]
        opensim.Vec3(0, -0.15, 0),  # COM
        opensim.Inertia(0.01, 0.01, 0.01),  # Inertia
    )

    # Shoulder joint (revolute)
    shoulder_loc = opensim.Vec3(0, 0, 0)
    shoulder_joint = opensim.PinJoint(
        "shoulder",
        ground,
        shoulder_loc,
        opensim.Vec3(0, 0, 0),
        upper_arm,
        shoulder_loc,
        opensim.Vec3(0, 0, 0),
    )

    model.addBody(upper_arm)
    model.addJoint(shoulder_joint)

    # Add a simple muscle (Thelen2003Muscle - Hill-type)
    muscle = opensim.Thelen2003Muscle()
    muscle.setName("biceps")
    muscle.setMaxIsometricForce(500.0)  # [N]
    muscle.setOptimalFiberLength(0.08)  # [m]
    muscle.setTendonSlackLength(0.2)  # [m]

    # Muscle path: origin on ground, insertion on upperarm
    muscle.addNewPathPoint("origin", ground, opensim.Vec3(0, 0.05, 0))
    muscle.addNewPathPoint("insertion", upper_arm, opensim.Vec3(0, -0.1, 0))

    model.addForce(muscle)

    # Finalize
    state = model.initSystem()

    return model, state

    # This is acceptable - we're testing the interface exists
