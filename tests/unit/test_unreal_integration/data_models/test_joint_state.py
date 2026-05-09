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


class TestJointState:
    """Tests for JointState data model."""

    def test_create_joint_state(self) -> None:
        """Test JointState creation."""
        js = JointState(
            name="shoulder_L",
            position=Vector3(x=0.1, y=1.4, z=0.2),
            rotation=Quaternion.identity(),
            velocity=Vector3.zero(),
            angular_velocity=Vector3.zero(),
        )
        assert js.name == "shoulder_L"
        assert js.position.x == 0.1

    def test_joint_state_with_angle(self) -> None:
        """Test JointState with joint angle."""
        js = JointState(
            name="elbow_L",
            position=Vector3.zero(),
            rotation=Quaternion.identity(),
            joint_angle=1.57,  # 90 degrees in radians
        )
        assert js.joint_angle == pytest.approx(1.57)

    def test_joint_state_to_dict(self) -> None:
        """Test JointState serialization."""
        js = JointState(
            name="shoulder_L",
            position=Vector3(x=0.1, y=1.4, z=0.2),
            rotation=Quaternion.identity(),
        )
        d = js.to_dict()
        assert d["name"] == "shoulder_L"
        assert "position" in d
        assert "rotation" in d

    def test_joint_state_from_dict(self) -> None:
        """Test JointState deserialization."""
        d = {
            "name": "shoulder_L",
            "position": {"x": 0.1, "y": 1.4, "z": 0.2},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        }
        js = JointState.from_dict(d)
        assert js.name == "shoulder_L"
        assert js.position.x == 0.1
