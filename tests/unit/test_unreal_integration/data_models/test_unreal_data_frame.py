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


class TestUnrealDataFrame:
    """Tests for UnrealDataFrame data model."""

    def test_create_data_frame(self) -> None:
        """Test UnrealDataFrame creation."""
        frame = UnrealDataFrame(
            timestamp=0.0167,
            frame_number=1,
            joints={
                "shoulder_L": JointState(
                    name="shoulder_L",
                    position=Vector3(x=0.1, y=1.4, z=0.2),
                    rotation=Quaternion.identity(),
                ),
            },
            forces=[
                ForceVector(
                    origin=Vector3.zero(),
                    direction=Vector3(x=0.0, y=-1.0, z=0.0),
                    magnitude=9.81,
                    force_type="gravity",
                ),
            ],
        )
        assert frame.timestamp == 0.0167
        assert frame.frame_number == 1
        assert "shoulder_L" in frame.joints

    def test_data_frame_with_club(self) -> None:
        """Test UnrealDataFrame with club state."""
        frame = UnrealDataFrame(
            timestamp=0.5,
            frame_number=30,
            joints={},
            club=ClubState(
                head_position=Vector3(x=0.5, y=0.8, z=0.1),
                head_velocity=Vector3(x=25.0, y=10.0, z=5.0),
            ),
        )
        assert frame.club is not None
        assert frame.club.head_position.x == 0.5

    def test_data_frame_with_metrics(self) -> None:
        """Test UnrealDataFrame with swing metrics."""
        frame = UnrealDataFrame(
            timestamp=1.0,
            frame_number=60,
            joints={},
            metrics=SwingMetrics(
                club_head_speed=45.2,
                x_factor=52.3,
            ),
        )
        assert frame.metrics is not None
        assert frame.metrics.club_head_speed == 45.2

    def test_data_frame_to_json(self) -> None:
        """Test UnrealDataFrame JSON serialization."""
        frame = UnrealDataFrame(
            timestamp=0.0167,
            frame_number=1,
            joints={
                "shoulder_L": JointState(
                    name="shoulder_L",
                    position=Vector3(x=0.1, y=1.4, z=0.2),
                    rotation=Quaternion.identity(),
                ),
            },
        )
        json_str = frame.to_json()
        data = json.loads(json_str)
        assert data["timestamp"] == 0.0167
        assert data["frame"] == 1
        assert "joints" in data

    def test_data_frame_from_json(self) -> None:
        """Test UnrealDataFrame JSON deserialization."""
        json_str = """{
            "timestamp": 0.0167,
            "frame": 1,
            "joints": {
                "shoulder_L": {
                    "name": "shoulder_L",
                    "position": {"x": 0.1, "y": 1.4, "z": 0.2},
                    "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
                }
            }
        }"""
        frame = UnrealDataFrame.from_json(json_str)
        assert frame.timestamp == 0.0167
        assert frame.frame_number == 1
        assert "shoulder_L" in frame.joints

    def test_data_frame_from_physics_state(self) -> None:
        """Test UnrealDataFrame creation from physics state."""
        # Simulated physics state
        q = np.array([0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0])  # 7 DOF
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # 6 DOF velocities
        t = 0.5

        frame = UnrealDataFrame.from_physics_state(
            q=q,
            v=v,
            timestamp=t,
            frame_number=30,
            joint_names=["pelvis", "shoulder_L"],
        )
        assert frame.timestamp == 0.5
        assert frame.frame_number == 30
        assert len(frame.joints) >= 1

    def test_data_frame_protocol_message(self) -> None:
        """Test UnrealDataFrame protocol message format."""
        frame = UnrealDataFrame(
            timestamp=0.0167,
            frame_number=1,
            joints={},
        )
        msg = frame.to_protocol_message()
        assert msg["type"] == "frame"
        assert "data" in msg
        assert msg["data"]["timestamp"] == 0.0167
