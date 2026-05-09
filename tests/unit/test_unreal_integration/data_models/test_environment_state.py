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


class TestEnvironmentState:
    """Tests for EnvironmentState data model."""

    def test_create_environment_state(self) -> None:
        """Test EnvironmentState creation."""
        env = EnvironmentState(
            wind_velocity=Vector3(x=5.0, y=0.0, z=0.0),
            temperature=20.0,
            humidity=0.6,
            altitude=100.0,
            air_density=1.225,
        )
        assert env.wind_velocity.x == 5.0
        assert env.temperature == 20.0

    def test_default_environment(self) -> None:
        """Test EnvironmentState.default() factory method."""
        env = EnvironmentState.default()
        assert env.temperature == 20.0
        assert env.air_density == pytest.approx(1.225)
