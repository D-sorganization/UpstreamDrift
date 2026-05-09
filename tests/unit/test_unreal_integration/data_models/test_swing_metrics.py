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


class TestSwingMetrics:
    """Tests for SwingMetrics data model."""

    def test_create_swing_metrics(self) -> None:
        """Test SwingMetrics creation."""
        sm = SwingMetrics(
            club_head_speed=45.2,
            x_factor=52.3,
            kinetic_energy=1250.5,
            smash_factor=1.48,
            attack_angle=-2.5,
            swing_path=3.0,
            face_to_path=-1.5,
        )
        assert sm.club_head_speed == 45.2
        assert sm.x_factor == 52.3

    def test_swing_metrics_calculated_fields(self) -> None:
        """Test SwingMetrics with calculated ball speed."""
        sm = SwingMetrics(
            club_head_speed=100.0,
            smash_factor=1.5,
        )
        assert sm.estimated_ball_speed == pytest.approx(150.0)

    def test_swing_metrics_to_dict(self) -> None:
        """Test SwingMetrics serialization."""
        sm = SwingMetrics(
            club_head_speed=45.2,
            x_factor=52.3,
        )
        d = sm.to_dict()
        assert d["club_head_speed"] == 45.2
        assert d["x_factor"] == 52.3
