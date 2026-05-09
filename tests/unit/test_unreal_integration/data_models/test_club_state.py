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


class TestClubState:
    """Tests for ClubState data model."""

    def test_create_club_state(self) -> None:
        """Test ClubState creation."""
        cs = ClubState(
            head_position=Vector3(x=0.5, y=0.8, z=0.1),
            head_velocity=Vector3(x=25.0, y=10.0, z=5.0),
            head_acceleration=Vector3(x=100.0, y=50.0, z=20.0),
            shaft_flex=[0.01, 0.02, 0.015, 0.01, 0.005],
            face_angle=2.5,
            loft_angle=10.0,
        )
        assert cs.head_position.x == 0.5
        assert cs.shaft_flex is not None
        assert len(cs.shaft_flex) == 5
        assert cs.face_angle == 2.5

    def test_club_head_speed(self) -> None:
        """Test ClubState head speed calculation."""
        cs = ClubState(
            head_position=Vector3.zero(),
            head_velocity=Vector3(x=30.0, y=40.0, z=0.0),
        )
        assert cs.head_speed == pytest.approx(50.0)  # 3-4-5 triangle

    def test_club_state_to_dict(self) -> None:
        """Test ClubState serialization."""
        cs = ClubState(
            head_position=Vector3(x=0.5, y=0.8, z=0.1),
            head_velocity=Vector3(x=25.0, y=10.0, z=5.0),
        )
        d = cs.to_dict()
        assert "head_position" in d
        assert "head_velocity" in d
        assert "head_speed" in d
