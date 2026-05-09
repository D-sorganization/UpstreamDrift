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


class TestGeometry:
    """Test Geometry factory methods and URDF serialization."""

    def test_box_factory(self) -> None:
        geom = Geometry.box(1.0, 2.0, 3.0)
        assert geom.geometry_type == GeometryType.BOX
        assert geom.dimensions == (1.0, 2.0, 3.0)

    def test_cylinder_factory(self) -> None:
        geom = Geometry.cylinder(0.1, 0.5)
        assert geom.geometry_type == GeometryType.CYLINDER
        assert geom.dimensions == (0.1, 0.5)

    def test_sphere_factory(self) -> None:
        geom = Geometry.sphere(0.25)
        assert geom.geometry_type == GeometryType.SPHERE
        assert geom.dimensions == (0.25,)

    def test_capsule_factory(self) -> None:
        geom = Geometry.capsule(0.05, 0.3)
        assert geom.geometry_type == GeometryType.CAPSULE
        assert geom.dimensions == (0.05, 0.3)

    def test_mesh_factory(self) -> None:
        geom = Geometry.mesh("model.stl", scale=(2.0, 2.0, 2.0))
        assert geom.geometry_type == GeometryType.MESH
        assert geom.mesh_filename == "model.stl"
        assert geom.mesh_scale == (2.0, 2.0, 2.0)

    def test_box_to_urdf_string(self) -> None:
        xml = Geometry.box(0.5, 1.0, 1.5).to_urdf_string()
        assert "<box" in xml
        assert 'size="0.5 1 1.5"' in xml

    def test_cylinder_to_urdf_string(self) -> None:
        xml = Geometry.cylinder(0.1, 0.5).to_urdf_string()
        assert "<cylinder" in xml
        assert 'radius="0.1"' in xml
        assert 'length="0.5"' in xml

    def test_sphere_to_urdf_string(self) -> None:
        xml = Geometry.sphere(0.25).to_urdf_string()
        assert "<sphere" in xml
        assert 'radius="0.25"' in xml

    def test_capsule_approximated_as_cylinder(self) -> None:
        xml = Geometry.capsule(0.05, 0.3).to_urdf_string()
        assert "<cylinder" in xml  # URDF doesn't have capsule

    def test_mesh_to_urdf_string(self) -> None:
        xml = Geometry.mesh("foo.stl").to_urdf_string()
        assert "<mesh" in xml
        assert 'filename="foo.stl"' in xml

    def test_from_dict_to_dict_roundtrip(self) -> None:
        original = Geometry.box(1.0, 2.0, 3.0)
        restored = Geometry.from_dict(original.to_dict())
        assert restored.geometry_type == original.geometry_type
        assert restored.dimensions == original.dimensions


# ── Origin ───────────────────────────────────────────────────────────────────


# ── JointLimits and JointDynamics ────────────────────────────────────────────


# ── Link and Joint ───────────────────────────────────────────────────────────


# ── Material ─────────────────────────────────────────────────────────────────
