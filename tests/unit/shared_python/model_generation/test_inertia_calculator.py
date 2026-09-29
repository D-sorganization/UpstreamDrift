"""Value-asserting tests for the unified inertia calculator (issue #6995).

Covers ``InertiaCalculator.compute_from_geometry`` against closed-form analytic
tensors for box/cylinder/sphere, ``InertiaResult`` validity/positive-definite
checks, ``scale_to_mass`` linearity, serialization round-trips
(``as_matrix``/``as_urdf_dict``/``to_dict``/``as_dict``), ``_detect_mode``
dispatch, ``create_default``, and rejection of non-positive mass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from model_generation.core.types import Geometry
from model_generation.inertia.calculator import (
    InertiaCalculator,
    InertiaMode,
    InertiaResult,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def calc() -> InertiaCalculator:
    return InertiaCalculator()


# --------------------------------------------------------------------------- #
# compute_from_geometry vs analytic tensors
# --------------------------------------------------------------------------- #


class TestComputeFromGeometryAnalytic:
    def test_box_matches_analytic(self, calc: InertiaCalculator) -> None:
        # Solid box: I = (m/12) * (sum of squares of the *other* two sides).
        # m=12, sides 2x4x6 => ixx=(1)*(16+36)=52, iyy=(4+36)=40, izz=(4+16)=20.
        result = calc.compute_from_geometry(Geometry.box(2.0, 4.0, 6.0), mass=12.0)
        assert result.ixx == pytest.approx(52.0)
        assert result.iyy == pytest.approx(40.0)
        assert result.izz == pytest.approx(20.0)
        assert result.ixy == 0.0 and result.ixz == 0.0 and result.iyz == 0.0
        assert result.mode == InertiaMode.PRIMITIVE

    def test_sphere_matches_analytic(self, calc: InertiaCalculator) -> None:
        # Solid sphere: I = (2/5) m r^2. m=10, r=0.5 => 0.4*10*0.25 = 1.0.
        result = calc.compute_from_geometry(Geometry.sphere(0.5), mass=10.0)
        assert result.ixx == pytest.approx(1.0)
        assert result.iyy == pytest.approx(1.0)
        assert result.izz == pytest.approx(1.0)

    def test_cylinder_matches_analytic(self, calc: InertiaCalculator) -> None:
        # Solid cylinder (axis z): izz = 0.5 m r^2;
        # ixx = iyy = (m/12)(3 r^2 + L^2). m=12, r=2, L=6.
        result = calc.compute_from_geometry(Geometry.cylinder(2.0, 6.0), mass=12.0)
        assert result.izz == pytest.approx(0.5 * 12.0 * 4.0)  # 24
        expected_perp = (12.0 / 12.0) * (3.0 * 4.0 + 36.0)  # 48
        assert result.ixx == pytest.approx(expected_perp)
        assert result.iyy == pytest.approx(expected_perp)

    def test_inertia_scales_linearly_with_mass(self, calc: InertiaCalculator) -> None:
        small = calc.compute_from_geometry(Geometry.sphere(0.3), mass=1.0)
        big = calc.compute_from_geometry(Geometry.sphere(0.3), mass=3.0)
        assert big.ixx == pytest.approx(3.0 * small.ixx)


# --------------------------------------------------------------------------- #
# Non-positive mass rejected (DbC) -- regression for the #6995 fix
# --------------------------------------------------------------------------- #


class TestMassValidation:
    def test_negative_mass_raises(self, calc: InertiaCalculator) -> None:
        with pytest.raises(ValueError, match="mass must be positive"):
            calc.compute_from_geometry(Geometry.box(1.0, 1.0, 1.0), mass=-2.0)

    def test_zero_mass_raises(self, calc: InertiaCalculator) -> None:
        with pytest.raises(ValueError, match="mass must be positive"):
            calc.compute_from_geometry(Geometry.sphere(0.5), mass=0.0)

    def test_scale_to_zero_or_negative_mass_raises(self) -> None:
        result = InertiaResult(ixx=1.0, iyy=1.0, izz=1.0, mass=2.0)
        with pytest.raises(ValueError):
            result.scale_to_mass(0.0)
        with pytest.raises(ValueError):
            result.scale_to_mass(-1.0)


# --------------------------------------------------------------------------- #
# scale_to_mass linearity
# --------------------------------------------------------------------------- #


class TestScaleToMass:
    def test_scales_components_and_mass(self) -> None:
        result = InertiaResult(
            ixx=2.0, iyy=4.0, izz=6.0, ixy=0.1, ixz=0.2, iyz=0.3, mass=2.0
        )
        scaled = result.scale_to_mass(6.0)  # 3x
        assert scaled.mass == pytest.approx(6.0)
        assert scaled.ixx == pytest.approx(6.0)
        assert scaled.iyy == pytest.approx(12.0)
        assert scaled.izz == pytest.approx(18.0)
        assert scaled.ixy == pytest.approx(0.3)
        assert scaled.ixz == pytest.approx(0.6)
        assert scaled.iyz == pytest.approx(0.9)
        # Original is untouched (returns a new instance).
        assert result.mass == pytest.approx(2.0)

    def test_scale_from_nonpositive_mass_raises(self) -> None:
        result = InertiaResult(ixx=1.0, iyy=1.0, izz=1.0, mass=0.0)
        with pytest.raises(ValueError, match="Cannot scale from zero or negative"):
            result.scale_to_mass(5.0)


# --------------------------------------------------------------------------- #
# Validity / positive-definite checks
# --------------------------------------------------------------------------- #


class TestValidity:
    def test_valid_diagonal_inertia(self) -> None:
        # Equal principal moments satisfy the triangle inequalities.
        result = InertiaResult(ixx=2.0, iyy=2.0, izz=2.0, mass=1.0)
        assert result.is_valid() is True
        assert result.validate_positive_definite() is True

    def test_nonpositive_diagonal_is_invalid(self) -> None:
        assert InertiaResult(ixx=0.0, iyy=1.0, izz=1.0, mass=1.0).is_valid() is False
        assert InertiaResult(ixx=-1.0, iyy=1.0, izz=1.0, mass=1.0).is_valid() is False

    def test_triangle_inequality_violation_is_invalid(self) -> None:
        # izz must be >= |ixx - iyy| and <= ixx + iyy. izz=100 breaks the upper.
        result = InertiaResult(ixx=1.0, iyy=1.0, izz=100.0, mass=1.0)
        assert result.is_valid() is False

    def test_nan_breaks_positive_definite(self) -> None:
        result = InertiaResult(ixx=float("nan"), iyy=1.0, izz=1.0, mass=1.0)
        assert result.validate_positive_definite() is False

    def test_indefinite_matrix_not_positive_definite(self) -> None:
        # Large off-diagonal term makes the matrix indefinite.
        result = InertiaResult(ixx=1.0, iyy=1.0, izz=1.0, ixy=5.0, mass=1.0)
        assert result.validate_positive_definite() is False


# --------------------------------------------------------------------------- #
# Serialization round-trips
# --------------------------------------------------------------------------- #


class TestSerialization:
    def test_as_matrix_symmetric(self) -> None:
        result = InertiaResult(
            ixx=1.0, iyy=2.0, izz=3.0, ixy=0.1, ixz=0.2, iyz=0.3, mass=1.0
        )
        mat = result.as_matrix()
        assert mat.shape == (3, 3)
        assert np.allclose(mat, mat.T)
        assert mat[0, 0] == 1.0 and mat[1, 1] == 2.0 and mat[2, 2] == 3.0
        assert mat[0, 1] == 0.1 and mat[0, 2] == 0.2 and mat[1, 2] == 0.3

    def test_as_urdf_dict_keys_and_values(self) -> None:
        result = InertiaResult(
            ixx=1.0, iyy=2.0, izz=3.0, ixy=0.1, ixz=0.2, iyz=0.3, mass=1.0
        )
        urdf = result.as_urdf_dict()
        assert set(urdf) == {"ixx", "ixy", "ixz", "iyy", "iyz", "izz"}
        assert urdf["ixx"] == 1.0 and urdf["iyz"] == 0.3

    def test_to_dict_preserves_all_fields(self) -> None:
        result = InertiaResult(
            ixx=1.0,
            iyy=2.0,
            izz=3.0,
            mass=4.0,
            center_of_mass=(0.1, 0.2, 0.3),
            volume=0.5,
            mode=InertiaMode.MANUAL,
            is_watertight=True,
            source="unit-test",
        )
        d = result.to_dict()
        assert d["ixx"] == 1.0
        assert d["mass"] == 4.0
        assert d["center_of_mass"] == [0.1, 0.2, 0.3]
        assert d["volume"] == 0.5
        assert d["mode"] == "manual"
        assert d["is_watertight"] is True
        assert d["source"] == "unit-test"

    def test_as_dict_humanoid_format(self) -> None:
        result = InertiaResult(ixx=1.0, iyy=1.0, izz=1.0, mass=2.0, volume=0.3)
        d = result.as_dict()
        assert d["was_watertight"] is None
        assert d["center_of_mass"] == [0.0, 0.0, 0.0]
        assert d["mode"] == "primitive"


# --------------------------------------------------------------------------- #
# _detect_mode dispatch
# --------------------------------------------------------------------------- #


class TestDetectMode:
    def test_dict_with_ixx_is_manual(self, calc: InertiaCalculator) -> None:
        assert calc._detect_mode({"ixx": 1.0}) == InertiaMode.MANUAL

    def test_dict_without_inertia_is_primitive(self, calc: InertiaCalculator) -> None:
        assert calc._detect_mode({"radius": 0.1}) == InertiaMode.PRIMITIVE

    def test_mesh_path_is_mesh_uniform(self, calc: InertiaCalculator) -> None:
        assert calc._detect_mode("arm.STL") == InertiaMode.MESH_UNIFORM_DENSITY
        assert calc._detect_mode(Path("part.obj")) == InertiaMode.MESH_UNIFORM_DENSITY

    def test_non_mesh_path_is_primitive(self, calc: InertiaCalculator) -> None:
        assert calc._detect_mode("config.yaml") == InertiaMode.PRIMITIVE

    def test_geometry_primitive_is_primitive(self, calc: InertiaCalculator) -> None:
        assert calc._detect_mode(Geometry.box(1, 1, 1)) == InertiaMode.PRIMITIVE


# --------------------------------------------------------------------------- #
# create_default & manual
# --------------------------------------------------------------------------- #


class TestDefaults:
    def test_create_default_uses_one_tenth_mass(self) -> None:
        result = InertiaResult.create_default(mass=5.0)
        assert result.mass == pytest.approx(5.0)
        assert result.ixx == pytest.approx(0.5)
        assert result.iyy == pytest.approx(0.5)
        assert result.izz == pytest.approx(0.5)
        assert result.is_valid() is True

    def test_compute_manual_round_trip(self, calc: InertiaCalculator) -> None:
        result = calc.compute_from_manual(
            ixx=0.1, iyy=0.2, izz=0.3, mass=2.0, center_of_mass=(1.0, 0.0, 0.0)
        )
        assert result.mode == InertiaMode.MANUAL
        assert result.ixx == 0.1 and result.izz == 0.3
        assert result.center_of_mass == (1.0, 0.0, 0.0)
