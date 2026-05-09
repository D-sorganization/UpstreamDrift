"""Heavy integration tests for the motion training / IK pipeline (fixes #1990).

Tests DualHandIKSolver instantiation with a Pinocchio model, IK solving
for a reachable target pose, and MotionVisualizer headless recording.
All tests skip gracefully when pinocchio, pink, or meshcat are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLFER_URDF = (
    Path(__file__).parents[2]
    / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)


def _pinocchio_available() -> bool:
    try:
        import pinocchio as pin  # noqa: F401

        return True
    except ImportError:
        return False


def _pink_available() -> bool:
    try:
        import pink  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.live_simulation
