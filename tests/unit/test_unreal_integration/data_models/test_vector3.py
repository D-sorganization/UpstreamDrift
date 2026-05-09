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


class TestVector3:
    """Tests for Vector3 data model."""

    def test_create_from_values(self) -> None:
        """Test Vector3 creation from individual values."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_create_from_numpy(self) -> None:
        """Test Vector3 creation from numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        v = Vector3.from_numpy(arr)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    def test_to_numpy(self) -> None:
        """Test Vector3 conversion to numpy array."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        arr = v.to_numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_data_models_magnitude(self) -> None:
        """Test Vector3 magnitude calculation."""
        v = Vector3(x=3.0, y=4.0, z=0.0)
        assert v.magnitude == pytest.approx(5.0)

    def test_data_models_normalized(self) -> None:
        """Test Vector3 normalization."""
        v = Vector3(x=3.0, y=4.0, z=0.0)
        n = v.normalized()
        assert n.magnitude == pytest.approx(1.0)
        assert n.x == pytest.approx(0.6)
        assert n.y == pytest.approx(0.8)

    def test_data_models_to_dict(self) -> None:
        """Test Vector3 serialization to dict."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        d = v.to_dict()
        assert d == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_data_models_from_dict(self) -> None:
        """Test Vector3 deserialization from dict."""
        d = {"x": 1.0, "y": 2.0, "z": 3.0}
        v = Vector3.from_dict(d)
        assert v.x == 1.0
        assert v.y == 2.0
        assert v.z == 3.0

    @pytest.mark.parametrize(
        "op, v1_args, v2_args, expected",
        [
            ("add", (1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (5.0, 7.0, 9.0)),
            ("sub", (4.0, 5.0, 6.0), (1.0, 2.0, 3.0), (3.0, 3.0, 3.0)),
        ],
        ids=["addition", "subtraction"],
    )
    def test_vector_arithmetic(self, op, v1_args, v2_args, expected) -> None:
        """Test Vector3 addition and subtraction."""
        v1 = Vector3(x=v1_args[0], y=v1_args[1], z=v1_args[2])
        v2 = Vector3(x=v2_args[0], y=v2_args[1], z=v2_args[2])
        result = (v1 + v2) if op == "add" else (v1 - v2)
        assert result.x == expected[0]
        assert result.y == expected[1]
        assert result.z == expected[2]

    def test_scalar_multiplication(self) -> None:
        """Test Vector3 scalar multiplication."""
        v = Vector3(x=1.0, y=2.0, z=3.0)
        result = v * 2.0
        assert result.x == 2.0
        assert result.y == 4.0
        assert result.z == 6.0

    def test_data_models_dot_product(self) -> None:
        """Test Vector3 dot product."""
        v1 = Vector3(x=1.0, y=2.0, z=3.0)
        v2 = Vector3(x=4.0, y=5.0, z=6.0)
        assert v1.dot(v2) == pytest.approx(32.0)

    def test_data_models_cross_product(self) -> None:
        """Test Vector3 cross product."""
        v1 = Vector3(x=1.0, y=0.0, z=0.0)
        v2 = Vector3(x=0.0, y=1.0, z=0.0)
        result = v1.cross(v2)
        assert result.x == pytest.approx(0.0)
        assert result.y == pytest.approx(0.0)
        assert result.z == pytest.approx(1.0)

    def test_zero_vector(self) -> None:
        """Test Vector3.zero() factory method."""
        v = Vector3.zero()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.z == 0.0
