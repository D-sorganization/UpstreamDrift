"""Tests for analytical inertia formulas for primitive shapes."""

from __future__ import annotations


import pytest
from model_generation.inertia.primitives import (
    box_inertia,
    capsule_inertia,
    combine_inertias,
    cone_inertia,
    cylinder_inertia,
    ellipsoid_inertia,
    hollow_cylinder_inertia,
    parallel_axis,
    sphere_inertia,
)

ZERO_OFFS = {"ixy": 0.0, "ixz": 0.0, "iyz": 0.0}


def test_box_unit_cube() -> None:
    result = box_inertia(12.0, 1.0, 1.0, 1.0)
    # I = (m/12)(s^2 + s^2) = (12/12)*2 = 2 for unit cube of mass 12
    assert result["ixx"] == pytest.approx(2.0)
    assert result["iyy"] == pytest.approx(2.0)
    assert result["izz"] == pytest.approx(2.0)
    for k in ("ixy", "ixz", "iyz"):
        assert result[k] == 0.0


def test_box_asymmetric() -> None:
    r = box_inertia(12.0, 2.0, 1.0, 1.0)
    assert r["ixx"] == pytest.approx((12 / 12) * (1 + 1))  # 2
    assert r["iyy"] == pytest.approx((12 / 12) * (4 + 1))  # 5
    assert r["izz"] == pytest.approx((12 / 12) * (4 + 1))


def test_cylinder_z_axis() -> None:
    r = cylinder_inertia(2.0, 1.0, 4.0, axis="z")
    # i_axial = 0.5*m*r^2 = 1.0
    # i_perp  = (m/12)(3 r^2 + L^2) = (2/12)(3 + 16) = 19/6
    assert r["izz"] == pytest.approx(1.0)
    assert r["ixx"] == pytest.approx(19.0 / 6.0)
    assert r["iyy"] == pytest.approx(19.0 / 6.0)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_cylinder_axis_orientation(axis: str) -> None:
    r = cylinder_inertia(3.0, 0.5, 2.0, axis=axis)
    diag = (r["ixx"], r["iyy"], r["izz"])
    # two perp values are equal; axial is different
    if axis == "x":
        assert r["iyy"] == pytest.approx(r["izz"])
        assert r["ixx"] != pytest.approx(r["iyy"])
    elif axis == "y":
        assert r["ixx"] == pytest.approx(r["izz"])
        assert r["iyy"] != pytest.approx(r["ixx"])
    else:
        assert r["ixx"] == pytest.approx(r["iyy"])
        assert r["izz"] != pytest.approx(r["ixx"])
    # all positive
    assert all(d > 0 for d in diag)


def test_sphere_inertia() -> None:
    r = sphere_inertia(5.0, 1.0)
    expected = (2.0 / 5.0) * 5.0 * 1.0
    assert r["ixx"] == pytest.approx(expected)
    assert r["iyy"] == pytest.approx(expected)
    assert r["izz"] == pytest.approx(expected)


def test_capsule_basic_z() -> None:
    r = capsule_inertia(1.0, 0.5, 1.0, axis="z")
    # axial perpendicular pair, izz axial
    assert r["ixx"] == pytest.approx(r["iyy"])
    assert r["ixx"] > r["izz"]


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_capsule_all_axes_positive(axis: str) -> None:
    r = capsule_inertia(2.0, 0.3, 1.5, axis=axis)
    assert r["ixx"] > 0 and r["iyy"] > 0 and r["izz"] > 0


def test_capsule_zero_radius_degenerate() -> None:
    # v_total == 0 -> sphere_inertia fallback
    r = capsule_inertia(1.0, 0.0, 0.0)
    # sphere_inertia(1, 0) returns zeros
    assert r["ixx"] == 0.0
    assert r["izz"] == 0.0


def test_ellipsoid_sphere_equivalence() -> None:
    # ellipsoid with a=b=c=r reduces to sphere
    r_sphere = sphere_inertia(3.0, 0.5)
    r_ell = ellipsoid_inertia(3.0, 0.5, 0.5, 0.5)
    assert r_ell["ixx"] == pytest.approx(r_sphere["ixx"])
    assert r_ell["izz"] == pytest.approx(r_sphere["izz"])


def test_hollow_cylinder_reduces_to_solid() -> None:
    # inner radius 0 -> equivalent to solid cylinder
    solid = cylinder_inertia(2.0, 1.0, 3.0, axis="z")
    hollow = hollow_cylinder_inertia(2.0, 0.0, 1.0, 3.0, axis="z")
    assert hollow["izz"] == pytest.approx(solid["izz"])
    assert hollow["ixx"] == pytest.approx(solid["ixx"])


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_hollow_cylinder_axes(axis: str) -> None:
    r = hollow_cylinder_inertia(2.0, 0.3, 0.5, 2.0, axis=axis)
    assert all(r[k] > 0 for k in ("ixx", "iyy", "izz"))


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_cone_axes(axis: str) -> None:
    r = cone_inertia(4.0, 0.5, 2.0, axis=axis)
    assert all(r[k] > 0 for k in ("ixx", "iyy", "izz"))


def test_cone_z_axial() -> None:
    r = cone_inertia(10.0, 1.0, 2.0, axis="z")
    assert r["izz"] == pytest.approx((3.0 / 10.0) * 10.0 * 1.0)


def test_parallel_axis_zero_offset_identity() -> None:
    original = {"ixx": 1.0, "iyy": 2.0, "izz": 3.0, "ixy": 0.1, "ixz": 0.2, "iyz": 0.3}
    shifted = parallel_axis(original, 5.0, (0.0, 0.0, 0.0))
    for k, v in original.items():
        assert shifted[k] == pytest.approx(v)


def test_parallel_axis_known_shift() -> None:
    base = {"ixx": 0.0, "iyy": 0.0, "izz": 0.0, "ixy": 0.0, "ixz": 0.0, "iyz": 0.0}
    shifted = parallel_axis(base, 2.0, (1.0, 0.0, 0.0))
    # Only iyy and izz get m*dx^2; ixx unaffected
    assert shifted["ixx"] == 0.0
    assert shifted["iyy"] == pytest.approx(2.0)
    assert shifted["izz"] == pytest.approx(2.0)


def test_parallel_axis_handles_missing_offdiag() -> None:
    # caller passes dict without ixy/ixz/iyz keys
    base = {"ixx": 1.0, "iyy": 1.0, "izz": 1.0}
    shifted = parallel_axis(base, 1.0, (1.0, 1.0, 0.0))
    assert shifted["ixy"] == pytest.approx(-1.0)


def test_combine_inertias_empty() -> None:
    r = combine_inertias([])
    assert r["ixx"] == 0.0
    assert all(v == 0.0 for v in r.values())


def test_combine_inertias_zero_total_mass() -> None:
    base = {"ixx": 1.0, "iyy": 1.0, "izz": 1.0, "ixy": 0, "ixz": 0, "iyz": 0}
    r = combine_inertias([(base, 0.0, (0, 0, 0))])
    assert all(v == 0.0 for v in r.values())


def test_combine_inertias_two_point_masses() -> None:
    # Two point masses (no rotational inertia) at +/- 1 along x, mass=1 each
    zero = {"ixx": 0, "iyy": 0, "izz": 0, "ixy": 0, "ixz": 0, "iyz": 0}
    r = combine_inertias([(zero, 1.0, (1.0, 0.0, 0.0)), (zero, 1.0, (-1.0, 0.0, 0.0))])
    # COM at origin. Each contributes m*d^2 = 1 to iyy, izz; ixx stays 0.
    assert r["ixx"] == pytest.approx(0.0)
    assert r["iyy"] == pytest.approx(2.0)
    assert r["izz"] == pytest.approx(2.0)


def test_box_invalid_mass_raises() -> None:
    # mass=None triggers ValueError; we keep the contract surface tested
    with pytest.raises(ValueError):
        box_inertia(None, 1.0, 1.0, 1.0)  # type: ignore[arg-type]
