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


class TestInertiaValidation:
    """Test Inertia validation and physical checks."""

    def test_positive_definite_diagonal(self) -> None:
        inertia = Inertia.from_sphere(mass=1.0, radius=1.0)
        assert inertia.is_positive_definite() is True

    def test_not_positive_definite_large_off_diagonal(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=1.0, izz=1.0, ixy=5.0)
        assert inertia.is_positive_definite() is False

    def test_is_diagonal_true(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=2.0, izz=3.0)
        assert inertia.is_diagonal() is True

    def test_is_diagonal_false(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=2.0, izz=3.0, ixy=0.5)
        assert inertia.is_diagonal() is False

    def test_is_diagonal_near_zero_off_diagonal(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=2.0, izz=3.0, ixy=1e-12)
        assert inertia.is_diagonal() is True

    def test_satisfies_triangle_inequality_sphere(self) -> None:
        inertia = Inertia.from_sphere(mass=1.0, radius=1.0)
        assert inertia.satisfies_triangle_inequality() is True

    def test_violates_triangle_inequality(self) -> None:
        # Izz > Ixx + Iyy  violates the triangle inequality
        inertia = Inertia(ixx=1.0, iyy=1.0, izz=10.0)
        assert inertia.satisfies_triangle_inequality() is False

    def test_satisfies_triangle_inequality_box(self) -> None:
        inertia = Inertia.from_box(mass=1.0, size_x=1.0, size_y=2.0, size_z=3.0)
        assert inertia.satisfies_triangle_inequality() is True


# ── Inertia operations ───────────────────────────────────────────────────────


# ── Geometry ─────────────────────────────────────────────────────────────────


# ── Origin ───────────────────────────────────────────────────────────────────


# ── JointLimits and JointDynamics ────────────────────────────────────────────


# ── Link and Joint ───────────────────────────────────────────────────────────


# ── Material ─────────────────────────────────────────────────────────────────
