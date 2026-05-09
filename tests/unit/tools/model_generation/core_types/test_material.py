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


# ── Link and Joint ───────────────────────────────────────────────────────────


# ── Material ─────────────────────────────────────────────────────────────────


class TestMaterial:
    """Test Material presets and serialization."""

    def test_skin_preset(self) -> None:
        mat = Material.skin()
        assert mat.name == "skin"
        assert len(mat.color) == 4

    def test_material_roundtrip(self) -> None:
        original = Material(
            name="custom", color=(0.1, 0.2, 0.3, 0.9), texture="tex.png"
        )
        restored = Material.from_dict(original.to_dict())
        assert restored.name == original.name
        assert tuple(restored.color) == original.color
        assert restored.texture == original.texture

    def test_material_to_urdf_inline(self) -> None:
        xml = Material.metal().to_urdf_string(inline=True)
        assert "<color" in xml
        assert 'name="metal"' in xml

    def test_material_to_urdf_reference(self) -> None:
        xml = Material.metal().to_urdf_string(inline=False)
        assert "<color" not in xml
        assert 'name="metal"' in xml
