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


class TestDataModelContracts:
    """Tests for Design by Contract compliance."""

    @pytest.mark.parametrize(
        "x, match",
        [
            (float("nan"), "NaN"),
            (float("inf"), "infinite"),
        ],
        ids=["nan", "infinite"],
    )
    def test_vector3_invalid_values(self, x, match) -> None:
        """Test Vector3 rejects NaN and infinite values."""
        with pytest.raises(ValueError, match=match):
            Vector3(x=x, y=0.0, z=0.0, validate=True)

    def test_quaternion_normalization_check(self) -> None:
        """Test Quaternion validates normalization."""
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0, validate=True)
        # Should auto-normalize when validate=True
        assert q.magnitude == pytest.approx(1.0)

    def test_force_vector_positive_magnitude(self) -> None:
        """Test ForceVector requires positive magnitude."""
        with pytest.raises(ValueError, match="positive"):
            ForceVector(
                origin=Vector3.zero(),
                direction=Vector3(x=1.0, y=0.0, z=0.0),
                magnitude=-5.0,
                validate=True,
            )

    def test_joint_state_requires_name(self) -> None:
        """Test JointState requires non-empty name."""
        with pytest.raises(ValueError, match="name"):
            JointState(
                name="",
                position=Vector3.zero(),
                rotation=Quaternion.identity(),
                validate=True,
            )

    @pytest.mark.parametrize(
        "timestamp, frame_number, match",
        [
            (-1.0, 1, "timestamp"),
            (0.0, -1, "frame"),
        ],
        ids=["negative-timestamp", "negative-frame"],
    )
    def test_data_frame_invalid_values(self, timestamp, frame_number, match) -> None:
        """Test UnrealDataFrame rejects invalid timestamp and frame number."""
        with pytest.raises(ValueError, match=match):
            UnrealDataFrame(
                timestamp=timestamp,
                frame_number=frame_number,
                joints={},
                validate=True,
            )
