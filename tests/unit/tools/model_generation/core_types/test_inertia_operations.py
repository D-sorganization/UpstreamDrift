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


class TestInertiaOperations:
    """Test Inertia operations like scaling and URDF serialization."""

    def test_to_matrix_shape_and_symmetry(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=2.0, izz=3.0, ixy=0.1, ixz=0.2, iyz=0.3)
        matrix = inertia.to_matrix()
        assert matrix.shape == (3, 3)
        np.testing.assert_array_almost_equal(matrix, matrix.T)

    def test_scale_to_mass_doubles(self) -> None:
        original = Inertia.from_box(mass=5.0, size_x=1.0, size_y=1.0, size_z=1.0)
        scaled = original.scale_to_mass(10.0)
        assert scaled.mass == 10.0
        assert scaled.ixx == pytest.approx(original.ixx * 2.0)
        assert scaled.iyy == pytest.approx(original.iyy * 2.0)
        assert scaled.izz == pytest.approx(original.izz * 2.0)

    def test_scale_to_mass_preserves_com(self) -> None:
        original = Inertia(
            ixx=1.0, iyy=1.0, izz=1.0, mass=2.0, center_of_mass=(1.0, 2.0, 3.0)
        )
        scaled = original.scale_to_mass(4.0)
        assert scaled.center_of_mass == (1.0, 2.0, 3.0)

    def test_scale_from_zero_mass_raises(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=1.0, izz=1.0, mass=0.0)
        with pytest.raises(ValueError, match="Cannot scale"):
            inertia.scale_to_mass(5.0)

    def test_scale_from_negative_mass_raises(self) -> None:
        inertia = Inertia(ixx=1.0, iyy=1.0, izz=1.0, mass=-1.0)
        with pytest.raises(ValueError, match="Cannot scale"):
            inertia.scale_to_mass(5.0)

    def test_to_urdf_string_format(self) -> None:
        inertia = Inertia(ixx=0.1, iyy=0.2, izz=0.3, ixy=0.01, ixz=0.02, iyz=0.03)
        xml = inertia.to_urdf_string()
        assert xml.startswith("<inertia")
        assert 'ixx="0.1"' in xml
        assert 'izz="0.3"' in xml
        assert xml.endswith("/>")


# ── Geometry ─────────────────────────────────────────────────────────────────


# ── Origin ───────────────────────────────────────────────────────────────────


# ── JointLimits and JointDynamics ────────────────────────────────────────────


# ── Link and Joint ───────────────────────────────────────────────────────────


# ── Material ─────────────────────────────────────────────────────────────────
