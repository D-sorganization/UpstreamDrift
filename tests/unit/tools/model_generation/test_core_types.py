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
        original = Inertia(ixx=1.0, iyy=1.0, izz=1.0, mass=2.0, center_of_mass=(1.0, 2.0, 3.0))
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


class TestOrigin:
    """Test Origin creation and serialization."""

    def test_defaults(self) -> None:
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


class TestLinkAndJoint:
    """Test Link and Joint from_dict/to_dict roundtrip and helpers."""

    def test_link_from_dict_minimal(self) -> None:
        link = Link.from_dict({"name": "base_link"})
        assert link.name == "base_link"
        assert link.inertia.mass == 1.0

    def test_link_roundtrip(self) -> None:
        original = Link(
            name="arm",
            inertia=Inertia.from_box(2.0, 0.1, 0.1, 0.5),
            visual_geometry=Geometry.box(0.1, 0.1, 0.5),
            visual_origin=Origin.from_position(0.0, 0.0, 0.25),
            visual_material=Material.skin(),
        )
        restored = Link.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.inertia.mass == pytest.approx(original.inertia.mass)
        assert restored.visual_geometry is not None
        assert restored.visual_geometry.geometry_type == GeometryType.BOX

    def test_joint_from_dict_defaults(self) -> None:
        joint = Joint.from_dict(
            {
                "name": "j1",
                "parent": "base",
                "child": "arm",
            }
        )
        assert joint.name == "j1"
        assert joint.joint_type == JointType.REVOLUTE
        assert joint.axis == (0.0, 0.0, 1.0)

    def test_joint_roundtrip(self) -> None:
        original = Joint(
            name="shoulder",
            joint_type=JointType.REVOLUTE,
            parent="torso",
            child="upper_arm",
            origin=Origin.from_position(0.0, 0.3, 0.0),
            axis=(1.0, 0.0, 0.0),
            limits=JointLimits(lower=-1.57, upper=1.57),
        )
        restored = Joint.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.joint_type == original.joint_type
        assert restored.parent == original.parent
        assert restored.child == original.child
        assert tuple(restored.axis) == original.axis

    def test_joint_auto_limits_for_revolute(self) -> None:
        joint = Joint(name="j", joint_type=JointType.REVOLUTE, parent="a", child="b")
        assert joint.limits is not None

    def test_joint_auto_limits_for_prismatic(self) -> None:
        joint = Joint(name="j", joint_type=JointType.PRISMATIC, parent="a", child="b")
        assert joint.limits is not None
        assert joint.limits.lower == -1.0

    def test_joint_no_auto_limits_for_continuous(self) -> None:
        joint = Joint(name="j", joint_type=JointType.CONTINUOUS, parent="a", child="b")
        assert joint.limits is None

    def test_is_composite(self) -> None:
        assert Joint(name="j", joint_type=JointType.GIMBAL, parent="a", child="b").is_composite()
        assert Joint(name="j", joint_type=JointType.UNIVERSAL, parent="a", child="b").is_composite()
        assert not Joint(
            name="j", joint_type=JointType.REVOLUTE, parent="a", child="b"
        ).is_composite()

    def test_dof_count(self) -> None:
        assert (
            Joint(name="j", joint_type=JointType.FIXED, parent="a", child="b").get_dof_count() == 0
        )
        assert (
            Joint(name="j", joint_type=JointType.REVOLUTE, parent="a", child="b").get_dof_count()
            == 1
        )
        assert (
            Joint(name="j", joint_type=JointType.FLOATING, parent="a", child="b").get_dof_count()
            == 6
        )
        assert (
            Joint(name="j", joint_type=JointType.GIMBAL, parent="a", child="b").get_dof_count() == 3
        )
        assert (
            Joint(name="j", joint_type=JointType.UNIVERSAL, parent="a", child="b").get_dof_count()
            == 2
        )
        assert (
            Joint(name="j", joint_type=JointType.PLANAR, parent="a", child="b").get_dof_count() == 3
        )


# ── Material ─────────────────────────────────────────────────────────────────


class TestMaterial:
    """Test Material presets and serialization."""

    def test_skin_preset(self) -> None:
        mat = Material.skin()
        assert mat.name == "skin"
        assert len(mat.color) == 4

    def test_material_roundtrip(self) -> None:
        original = Material(name="custom", color=(0.1, 0.2, 0.3, 0.9), texture="tex.png")
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
