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


class TestQuaternion:
    """Tests for Quaternion data model."""

    @pytest.mark.parametrize(
        "factory, expected_w",
        [
            (lambda: Quaternion(w=1.0, x=0.0, y=0.0, z=0.0), 1.0),
            (lambda: Quaternion.identity(), 1.0),
        ],
        ids=["from-values", "identity"],
    )
    def test_creation(self, factory, expected_w) -> None:
        """Test Quaternion creation methods."""
        q = factory()
        assert q.w == expected_w
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_from_euler(self) -> None:
        """Test Quaternion creation from Euler angles."""
        # 90 degrees around Z axis
        q = Quaternion.from_euler(roll=0, pitch=0, yaw=math.pi / 2)
        assert q.w == pytest.approx(math.cos(math.pi / 4))
        assert q.z == pytest.approx(math.sin(math.pi / 4))

    @pytest.mark.parametrize(
        "roll, pitch, yaw",
        [(0.1, 0.2, 0.3), (0.0, 0.0, 0.0), (0.5, -0.3, 0.8)],
        ids=["small-angles", "zero", "mixed"],
    )
    def test_euler_roundtrip(self, roll, pitch, yaw) -> None:
        """Test Quaternion Euler conversion roundtrip."""
        q = Quaternion.from_euler(roll=roll, pitch=pitch, yaw=yaw)
        r, p, y = q.to_euler()
        assert r == pytest.approx(roll, abs=1e-6)
        assert p == pytest.approx(pitch, abs=1e-6)
        assert y == pytest.approx(yaw, abs=1e-6)

    def test_data_models_magnitude(self) -> None:
        """Test Quaternion magnitude calculation."""
        q = Quaternion.identity()
        assert q.magnitude == pytest.approx(1.0)

    def test_data_models_normalized(self) -> None:
        """Test Quaternion normalization."""
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0)
        n = q.normalized()
        assert n.magnitude == pytest.approx(1.0)
        assert n.w == pytest.approx(1.0)

    def test_conjugate(self) -> None:
        """Test Quaternion conjugate."""
        q = Quaternion(w=1.0, x=2.0, y=3.0, z=4.0)
        c = q.conjugate()
        assert c.w == 1.0
        assert c.x == -2.0
        assert c.y == -3.0
        assert c.z == -4.0

    def test_data_models_to_dict(self) -> None:
        """Test Quaternion serialization to dict."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        d = q.to_dict()
        assert d == {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
