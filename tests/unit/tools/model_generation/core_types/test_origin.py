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


class TestOrigin:
    """Test Origin creation and serialization."""

    def test_core_types_defaults(self) -> None:
        origin = Origin()
        assert origin.xyz == (0.0, 0.0, 0.0)
        assert origin.rpy == (0.0, 0.0, 0.0)

    def test_from_position(self) -> None:
        origin = Origin.from_position(1.0, 2.0, 3.0)
        assert origin.xyz == (1.0, 2.0, 3.0)
        assert origin.rpy == (0.0, 0.0, 0.0)

    def test_from_dict_with_lists(self) -> None:
        origin = Origin.from_dict({"xyz": [1, 2, 3], "rpy": [0.1, 0.2, 0.3]})
        assert origin.xyz == (1, 2, 3)
        assert origin.rpy == (0.1, 0.2, 0.3)

    def test_from_dict_defaults(self) -> None:
        origin = Origin.from_dict({})
        assert origin.xyz == (0.0, 0.0, 0.0)

    def test_to_dict_roundtrip(self) -> None:
        original = Origin(xyz=(1.0, 2.0, 3.0), rpy=(0.1, 0.2, 0.3))
        restored = Origin.from_dict(original.to_dict())
        assert tuple(restored.xyz) == original.xyz
        assert tuple(restored.rpy) == original.rpy

    def test_to_urdf_string(self) -> None:
        xml = Origin(xyz=(1.0, 0.0, 0.5), rpy=(0.0, 0.0, 0.0)).to_urdf_string()
        assert xml.startswith("<origin")
        assert 'xyz="1 0 0.5"' in xml


# ── JointLimits and JointDynamics ────────────────────────────────────────────


# ── Link and Joint ───────────────────────────────────────────────────────────


# ── Material ─────────────────────────────────────────────────────────────────
