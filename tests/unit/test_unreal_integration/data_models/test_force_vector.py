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


class TestForceVector:
    """Tests for ForceVector data model."""

    def test_create_force_vector(self) -> None:
        """Test ForceVector creation."""
        fv = ForceVector(
            origin=Vector3(x=0.0, y=1.0, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
            magnitude=9.81,
            force_type="gravity",
        )
        assert fv.magnitude == 9.81
        assert fv.force_type == "gravity"

    def test_force_vector_endpoint(self) -> None:
        """Test ForceVector endpoint calculation."""
        fv = ForceVector(
            origin=Vector3(x=0.0, y=0.0, z=0.0),
            direction=Vector3(x=1.0, y=0.0, z=0.0),
            magnitude=5.0,
        )
        endpoint = fv.endpoint()
        assert endpoint.x == pytest.approx(5.0)
        assert endpoint.y == pytest.approx(0.0)
        assert endpoint.z == pytest.approx(0.0)

    def test_force_vector_to_dict(self) -> None:
        """Test ForceVector serialization."""
        fv = ForceVector(
            origin=Vector3(x=0.0, y=1.0, z=0.0),
            direction=Vector3(x=0.0, y=-1.0, z=0.0),
            magnitude=9.81,
            force_type="gravity",
        )
        d = fv.to_dict()
        assert d["magnitude"] == 9.81
        assert d["force_type"] == "gravity"

    def test_torque_vector(self) -> None:
        """Test ForceVector for torque representation."""
        tv = ForceVector(
            origin=Vector3(x=0.0, y=0.0, z=0.0),
            direction=Vector3(x=0.0, y=0.0, z=1.0),
            magnitude=10.5,
            force_type="torque",
            joint_name="shoulder_L",
        )
        assert tv.force_type == "torque"
        assert tv.joint_name == "shoulder_L"
