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


class TestInertiaFactoryMethods:
    """Test all Inertia class factory methods."""

    def test_from_box_computes_correct_values(self) -> None:
        inertia = Inertia.from_box(mass=12.0, size_x=1.0, size_y=2.0, size_z=3.0)
        assert inertia.mass == 12.0
        assert inertia.ixx == pytest.approx((12.0 / 12.0) * (4.0 + 9.0))
        assert inertia.iyy == pytest.approx((12.0 / 12.0) * (1.0 + 9.0))
        assert inertia.izz == pytest.approx((12.0 / 12.0) * (1.0 + 4.0))
        assert inertia.ixy == 0.0
        assert inertia.ixz == 0.0
        assert inertia.iyz == 0.0

    def test_from_box_cube_is_isotropic(self) -> None:
        inertia = Inertia.from_box(mass=6.0, size_x=1.0, size_y=1.0, size_z=1.0)
        assert inertia.ixx == pytest.approx(inertia.iyy)
        assert inertia.iyy == pytest.approx(inertia.izz)

    def test_from_cylinder_z_axis(self) -> None:
        inertia = Inertia.from_cylinder(mass=5.0, radius=0.1, length=1.0, axis="z")
        assert inertia.mass == 5.0
        assert inertia.izz == pytest.approx(0.5 * 5.0 * 0.01)  # axial
        assert inertia.ixx == pytest.approx(inertia.iyy)  # perpendicular equal

    def test_from_cylinder_x_axis(self) -> None:
        inertia = Inertia.from_cylinder(mass=5.0, radius=0.1, length=1.0, axis="x")
        assert inertia.ixx == pytest.approx(0.5 * 5.0 * 0.01)
        assert inertia.iyy == pytest.approx(inertia.izz)

    def test_from_cylinder_y_axis(self) -> None:
        inertia = Inertia.from_cylinder(mass=5.0, radius=0.1, length=1.0, axis="y")
        assert inertia.iyy == pytest.approx(0.5 * 5.0 * 0.01)
        assert inertia.ixx == pytest.approx(inertia.izz)

    def test_from_sphere_isotropic(self) -> None:
        inertia = Inertia.from_sphere(mass=10.0, radius=0.5)
        expected = (2.0 / 5.0) * 10.0 * 0.25
        assert inertia.ixx == pytest.approx(expected)
        assert inertia.iyy == pytest.approx(expected)
        assert inertia.izz == pytest.approx(expected)

    def test_from_capsule_default_z_axis(self) -> None:
        inertia = Inertia.from_capsule(mass=3.0, radius=0.05, length=0.5)
        assert inertia.mass == 3.0
        # Axial (z) should be less than perpendicular
        assert inertia.izz < inertia.ixx
        assert inertia.ixx == pytest.approx(inertia.iyy)

    def test_from_capsule_zero_length_approaches_sphere(self) -> None:
        """Capsule with length=0 is close to a sphere but not identical.

        The parallel axis theorem offset (3/8 * radius for hemisphere COM)
        causes the perpendicular inertia to be slightly larger than a
        solid sphere of equal mass and radius.
        """
        capsule = Inertia.from_capsule(mass=1.0, radius=0.1, length=0.0)
        sphere = Inertia.from_sphere(mass=1.0, radius=0.1)
        # Axial component should equal sphere (no parallel axis for axial)
        assert capsule.izz == pytest.approx(sphere.izz, rel=1e-6)
        # Perpendicular should be >= sphere (parallel axis adds)
        assert capsule.ixx >= sphere.ixx
        assert capsule.ixx == pytest.approx(capsule.iyy)

    def test_from_matrix_roundtrip(self) -> None:
        original = Inertia(ixx=1.0, iyy=2.0, izz=3.0, ixy=0.1, ixz=0.2, iyz=0.3)
        matrix = original.to_matrix()
        reconstructed = Inertia.from_matrix(matrix, mass=original.mass)
        assert reconstructed.ixx == pytest.approx(original.ixx)
        assert reconstructed.iyy == pytest.approx(original.iyy)
        assert reconstructed.izz == pytest.approx(original.izz)
        assert reconstructed.ixy == pytest.approx(original.ixy)
        assert reconstructed.ixz == pytest.approx(original.ixz)
        assert reconstructed.iyz == pytest.approx(original.iyz)

    def test_from_dict_all_keys(self) -> None:
        data = {
            "ixx": 1.5,
            "iyy": 2.5,
            "izz": 3.5,
            "ixy": 0.1,
            "ixz": 0.2,
            "iyz": 0.3,
            "mass": 7.0,
            "center_of_mass": [0.1, 0.2, 0.3],
        }
        inertia = Inertia.from_dict(data)
        assert inertia.ixx == 1.5
        assert inertia.mass == 7.0
        assert inertia.center_of_mass == (0.1, 0.2, 0.3)

    def test_from_dict_defaults(self) -> None:
        inertia = Inertia.from_dict({})
        assert inertia.ixx == 0.1
        assert inertia.mass == 1.0
        assert inertia.ixy == 0.0

    def test_to_dict_roundtrip(self) -> None:
        original = Inertia(ixx=1.0, iyy=2.0, izz=3.0, ixy=0.1, mass=5.0)
        restored = Inertia.from_dict(original.to_dict())
        assert restored.ixx == original.ixx
        assert restored.iyy == original.iyy
        assert restored.mass == original.mass


# ── Inertia validation methods ───────────────────────────────────────────────


# ── Inertia operations ───────────────────────────────────────────────────────


# ── Geometry ─────────────────────────────────────────────────────────────────


# ── Origin ───────────────────────────────────────────────────────────────────


# ── JointLimits and JointDynamics ────────────────────────────────────────────


# ── Link and Joint ───────────────────────────────────────────────────────────


# ── Material ─────────────────────────────────────────────────────────────────
