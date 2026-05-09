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


class TestTrajectoryPoint:
    """Tests for TrajectoryPoint data model."""

    def test_create_trajectory_point(self) -> None:
        """Test TrajectoryPoint creation."""
        tp = TrajectoryPoint(
            time=0.5,
            position=Vector3(x=10.0, y=0.0, z=5.0),
            velocity=Vector3(x=50.0, y=0.0, z=25.0),
        )
        assert tp.time == 0.5
        assert tp.position.x == 10.0

    def test_trajectory_point_color(self) -> None:
        """Test TrajectoryPoint with color."""
        tp = TrajectoryPoint(
            time=0.5,
            position=Vector3(x=10.0, y=0.0, z=5.0),
            color=(1.0, 0.0, 0.0, 1.0),  # Red with full alpha
        )
        assert tp.color == (1.0, 0.0, 0.0, 1.0)
