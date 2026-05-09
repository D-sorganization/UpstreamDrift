"""Unit tests for BallRollPhysics module.

TDD Tests - These tests define the expected behavior of the ball rolling
physics including sliding, rolling, spin decay, and energy conservation.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
    BallState,
    RollMode,
)
from src.engines.physics_engines.putting_green.python.green_surface import GreenSurface
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)


class TestRollMode:
    """Tests for RollMode enumeration."""

    def test_roll_modes_exist(self) -> None:
        """Verify all roll modes are defined."""
        assert hasattr(RollMode, "SLIDING")
        assert hasattr(RollMode, "ROLLING")
        assert hasattr(RollMode, "STOPPED")
