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


class TestDataModelPerformance:
    """Performance tests for data models."""

    def test_large_joint_set_serialization(self) -> None:
        """Test serialization performance with many joints."""
        joints = {}
        for i in range(50):  # Standard humanoid skeleton has ~50 bones
            joints[f"joint_{i}"] = JointState(
                name=f"joint_{i}",
                position=Vector3(x=float(i), y=0.0, z=0.0),
                rotation=Quaternion.identity(),
            )

        frame = UnrealDataFrame(
            timestamp=0.0167,
            frame_number=1,
            joints=joints,
        )

        # Serialization should complete quickly
        json_str = frame.to_json()
        assert len(json_str) > 0

        # Deserialization should also be fast
        frame2 = UnrealDataFrame.from_json(json_str)
        assert len(frame2.joints) == 50

    def test_trajectory_with_many_points(self) -> None:
        """Test trajectory with many points."""
        points = []
        for i in range(1000):
            points.append(
                TrajectoryPoint(
                    time=i * 0.001,
                    position=Vector3(x=float(i), y=0.0, z=float(i) ** 2 * 0.0001),
                )
            )

        frame = UnrealDataFrame(
            timestamp=1.0,
            frame_number=60,
            joints={},
            trajectory=points,
        )

        json_str = frame.to_json()
        assert len(json_str) > 0
