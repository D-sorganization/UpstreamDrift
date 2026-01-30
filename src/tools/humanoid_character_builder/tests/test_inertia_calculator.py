"""
Unit tests for inertia calculation modules.
"""

import math

import numpy as np
from humanoid_character_builder.mesh.inertia_calculator import (
    InertiaMode,
    InertiaResult,
    MeshInertiaCalculator,
    validate_inertia_tensor,
)
from humanoid_character_builder.mesh.primitive_inertia import (
    PrimitiveInertiaCalculator,
    PrimitiveShape,
    estimate_segment_primitive,
)


class TestInertiaResult:
    """Tests for InertiaResult class."""

    def test_default_creation(self):
        result = InertiaResult.create_default(mass=1.0)
        assert result.mass == 1.0
        assert result.ixx > 0
        assert result.iyy > 0
        assert result.izz > 0

    def test_as_matrix(self):
        result = InertiaResult(
            ixx=1.0, iyy=2.0, izz=3.0, ixy=0.1, ixz=0.2, iyz=0.3, mass=1.0
        )
        matrix = result.as_matrix()

        assert matrix.shape == (3, 3)
        assert matrix[0, 0] == 1.0
        assert matrix[1, 1] == 2.0
        assert matrix[2, 2] == 3.0
        assert matrix[0, 1] == 0.1
        assert matrix[0, 2] == 0.2
        assert matrix[1, 2] == 0.3

    def test_as_urdf_dict(self):
        result = InertiaResult(
            ixx=1.0, iyy=2.0, izz=3.0, ixy=0.1, ixz=0.2, iyz=0.3, mass=1.0
        )
        urdf_dict = result.as_urdf_dict()

        assert "ixx" in urdf_dict
        assert "iyy" in urdf_dict
        assert "izz" in urdf_dict
        assert "ixy" in urdf_dict
        assert urdf_dict["ixx"] == 1.0

    def test_is_valid_positive(self):
        # Valid inertia for a symmetric shape
        result = InertiaResult(ixx=1.0, iyy=1.0, izz=1.0, mass=1.0)
        assert result.is_valid()

    def test_is_valid_negative_diagonal(self):
        result = InertiaResult(ixx=-1.0, iyy=1.0, izz=1.0, mass=1.0)
        assert not result.is_valid()

    def test_validate_positive_definite(self):
        # Sphere-like inertia is positive definite
        result = InertiaResult(ixx=1.0, iyy=1.0, izz=1.0, mass=1.0)
        assert result.validate_positive_definite()

    def test_as_dict(self):
        result = InertiaResult(
            ixx=1.0,
            iyy=2.0,
            izz=3.0,
            mass=5.0,
            volume=0.001,
            mode=InertiaMode.PRIMITIVE_APPROXIMATION,
        )
        data = result.as_dict()

        assert data["ixx"] == 1.0
        assert data["mass"] == 5.0
        assert data["volume"] == 0.001
        assert data["mode"] == "primitive"


class TestPrimitiveInertiaCalculator:
    """Tests for PrimitiveInertiaCalculator."""

    def test_box_inertia(self):
        calc = PrimitiveInertiaCalculator()
        result = calc.compute_box(mass=1.0, size_x=1.0, size_y=1.0, size_z=1.0)

        # For a unit cube with mass 1: I = (1/12) * 1 * (1^2 + 1^2) = 1/6
        expected_i = 1.0 / 6.0
        assert abs(result.ixx - expected_i) < 1e-10
        assert abs(result.iyy - expected_i) < 1e-10
        assert abs(result.izz - expected_i) < 1e-10
        assert result.volume == 1.0

    def test_cylinder_inertia(self):
        calc = PrimitiveInertiaCalculator()
        mass = 1.0
        radius = 0.5
        length = 1.0

        result = calc.compute_cylinder(mass, radius, length, axis="z")

        # I_zz = (1/2) * m * r^2
        expected_izz = 0.5 * mass * radius**2
        assert abs(result.izz - expected_izz) < 1e-10

        # I_xx = I_yy = (1/12) * m * (3*r^2 + h^2)
        expected_ixx = (1.0 / 12.0) * mass * (3 * radius**2 + length**2)
        assert abs(result.ixx - expected_ixx) < 1e-10
        assert abs(result.iyy - expected_ixx) < 1e-10

    def test_sphere_inertia(self):
        calc = PrimitiveInertiaCalculator()
        mass = 1.0
        radius = 1.0

        result = calc.compute_sphere(mass, radius)

        # I = (2/5) * m * r^2
        expected_i = 0.4 * mass * radius**2
        assert abs(result.ixx - expected_i) < 1e-10
        assert abs(result.iyy - expected_i) < 1e-10
        assert abs(result.izz - expected_i) < 1e-10

        # Volume = (4/3) * pi * r^3
        expected_volume = (4.0 / 3.0) * math.pi * radius**3
        assert abs(result.volume - expected_volume) < 1e-10

    def test_capsule_inertia(self):
        calc = PrimitiveInertiaCalculator()
        mass = 1.0
        radius = 0.1
        length = 0.5

        result = calc.compute_capsule(mass, radius, length, axis="z")

        # Should be valid inertia
        assert result.ixx > 0
        assert result.iyy > 0
        assert result.izz > 0
        assert result.validate_positive_definite()

    def test_ellipsoid_inertia(self):
        calc = PrimitiveInertiaCalculator()
        mass = 1.0
        a, b, c = 1.0, 0.5, 0.25

        result = calc.compute_ellipsoid(mass, a, b, c)

        # I_xx = (1/5) * m * (b^2 + c^2)
        expected_ixx = 0.2 * mass * (b**2 + c**2)
        assert abs(result.ixx - expected_ixx) < 1e-10

    def test_compute_generic(self):
        calc = PrimitiveInertiaCalculator()

        # Test with string shape
        result = calc.compute("box", 1.0, {"x": 0.1, "y": 0.1, "z": 0.1})
        assert result.ixx > 0

        # Test with tuple dimensions
        result = calc.compute(PrimitiveShape.SPHERE, 1.0, (0.1,))
        assert result.ixx > 0


class TestEstimateSegmentPrimitive:
    """Tests for segment primitive estimation."""

    def test_head_is_sphere(self):
        shape, dims = estimate_segment_primitive("head", 0.2)
        assert shape == PrimitiveShape.SPHERE
        assert "radius" in dims

    def test_thigh_is_capsule(self):
        shape, dims = estimate_segment_primitive("thigh", 0.4, 0.1)
        assert shape == PrimitiveShape.CAPSULE
        assert "radius" in dims
        assert "length" in dims

    def test_torso_is_box(self):
        shape, dims = estimate_segment_primitive("thorax", 0.3, 0.25, 0.15)
        assert shape == PrimitiveShape.BOX
        assert "x" in dims
        assert "y" in dims
        assert "z" in dims


class TestValidateInertiaTensor:
    """Tests for inertia tensor validation."""

    def test_valid_tensor(self):
        # Valid diagonal tensor
        I = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        errors = validate_inertia_tensor(I)
        assert len(errors) == 0

    def test_non_symmetric(self):
        I = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        errors = validate_inertia_tensor(I)
        assert any("symmetric" in e for e in errors)

    def test_negative_diagonal(self):
        I = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        errors = validate_inertia_tensor(I)
        assert any("positive" in e.lower() for e in errors)

    def test_wrong_shape(self):
        I = np.array([[1.0, 0.0], [0.0, 1.0]])
        errors = validate_inertia_tensor(I)
        assert any("3x3" in e for e in errors)


class TestMeshInertiaCalculator:
    """Tests for MeshInertiaCalculator."""

    def test_create_manual_inertia(self):
        result = MeshInertiaCalculator.create_manual_inertia(
            ixx=1.0, iyy=2.0, izz=3.0, mass=5.0, com=(0.1, 0.2, 0.3)
        )

        assert result.ixx == 1.0
        assert result.iyy == 2.0
        assert result.izz == 3.0
        assert result.mass == 5.0
        assert result.center_of_mass == (0.1, 0.2, 0.3)
        assert result.mode == InertiaMode.MANUAL

    def test_transform_inertia_rotation(self):
        calc = MeshInertiaCalculator()

        # Create inertia
        original = InertiaResult(ixx=1.0, iyy=2.0, izz=3.0, mass=1.0)

        # 180 degree rotation about Z should swap Ixx and Iyy signs of products
        # but diagonal should remain same (symmetric)
        R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)
        transformed = calc.transform_inertia(original, rotation=R)

        # Diagonal elements should be preserved
        assert abs(transformed.ixx - original.ixx) < 1e-10
        assert abs(transformed.iyy - original.iyy) < 1e-10
        assert abs(transformed.izz - original.izz) < 1e-10

    def test_check_trimesh_availability(self):
        calc = MeshInertiaCalculator()
        # Just verify it doesn't crash
        _ = calc._trimesh_available
