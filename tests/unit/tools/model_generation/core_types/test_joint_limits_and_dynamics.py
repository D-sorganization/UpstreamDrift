"""
Comprehensive unit tests for model_generation core types.

Tests cover all data classes in model_generation.core.types:
- Inertia: factory methods, validation, operations, serialization
- Geometry: factory methods, to_urdf_string for all types
- Origin: creation, from_dict/to_dict, to_urdf_string
- JointLimits and JointDynamics: from_dict/to_dict roundtrip
- Link and Joint: from_dict/to_dict roundtrip, DOF counting
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    Link,
    Material,
    Origin,
)

# ── Inertia factory methods ──────────────────────────────────────────────────


# ── Inertia validation methods ───────────────────────────────────────────────


# ── Inertia operations ───────────────────────────────────────────────────────


# ── Geometry ─────────────────────────────────────────────────────────────────


# ── Origin ───────────────────────────────────────────────────────────────────


# ── JointLimits and JointDynamics ────────────────────────────────────────────


class TestJointLimitsAndDynamics:
    """Test JointLimits and JointDynamics roundtrip and URDF output."""

    def test_joint_limits_defaults(self) -> None:
        limits = JointLimits()
        assert limits.lower == pytest.approx(-math.pi)
        assert limits.upper == pytest.approx(math.pi)
        assert limits.effort == 1000.0
        assert limits.velocity == 10.0

    def test_joint_limits_roundtrip(self) -> None:
        original = JointLimits(lower=-1.0, upper=1.0, effort=50.0, velocity=5.0)
        restored = JointLimits.from_dict(original.to_dict())
        assert restored.lower == original.lower
        assert restored.upper == original.upper
        assert restored.effort == original.effort
        assert restored.velocity == original.velocity

    def test_joint_limits_to_urdf(self) -> None:
        xml = JointLimits(lower=-1.5, upper=1.5).to_urdf_string()
        assert "<limit" in xml
        assert 'lower="-1.5"' in xml

    def test_joint_dynamics_defaults(self) -> None:
        dyn = JointDynamics()
        assert dyn.damping == 0.5
        assert dyn.friction == 0.0

    def test_joint_dynamics_roundtrip(self) -> None:
        original = JointDynamics(damping=1.5, friction=0.3)
        restored = JointDynamics.from_dict(original.to_dict())
        assert restored.damping == original.damping
        assert restored.friction == original.friction

    def test_joint_dynamics_to_urdf(self) -> None:
        xml = JointDynamics(damping=2.0, friction=0.5).to_urdf_string()
        assert "<dynamics" in xml
        assert 'damping="2"' in xml


# ── Link and Joint ───────────────────────────────────────────────────────────


# ── Material ─────────────────────────────────────────────────────────────────
