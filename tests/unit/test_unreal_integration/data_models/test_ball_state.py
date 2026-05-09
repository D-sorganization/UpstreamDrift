"""Unit tests for Unreal Engine data models.

Following TDD principles - tests written first to define expected behavior.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from src.unreal_integration.data_models import (
    BallState,
    ClubState,
    EnvironmentState,
    ForceVector,
    JointState,
    Quaternion,
    SwingMetrics,
    TrajectoryPoint,
    UnrealDataFrame,
    Vector3,
)


class TestBallState:
    """Tests for BallState data model."""

    def test_create_ball_state(self) -> None:
        """Test BallState creation."""
        bs = BallState(
            position=Vector3(x=0.0, y=0.0, z=0.05),
            velocity=Vector3(x=60.0, y=20.0, z=40.0),
            spin_rate=2500.0,
            spin_axis=Vector3(x=0.0, y=0.1, z=1.0),
        )
        assert bs.spin_rate == 2500.0
        assert bs.velocity.x == 60.0

    def test_ball_launch_angle(self) -> None:
        """Test BallState launch angle calculation."""
        bs = BallState(
            position=Vector3.zero(),
            velocity=Vector3(x=100.0, y=0.0, z=100.0),  # 45 degree launch
        )
        assert bs.launch_angle == pytest.approx(45.0, abs=0.1)
