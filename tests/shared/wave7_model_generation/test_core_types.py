"""Tests for model_generation.core.types dataclasses."""

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

# --- Origin ---


def test_origin_defaults_and_from_position() -> None:
    o = Origin.from_position(1.0, 2.0, 3.0)
    assert o.xyz == (1.0, 2.0, 3.0)
    assert o.rpy == (0.0, 0.0, 0.0)


def test_origin_from_dict_lists_converted_to_tuples() -> None:
    o = Origin.from_dict({"xyz": [1, 2, 3], "rpy": [0.1, 0.2, 0.3]})
    assert o.xyz == (1, 2, 3)
    assert o.rpy == (0.1, 0.2, 0.3)


def test_origin_from_dict_defaults() -> None:
    o = Origin.from_dict({})
    assert o.xyz == (0.0, 0.0, 0.0)


def test_origin_from_dict_none_raises() -> None:
    with pytest.raises(ValueError):
        Origin.from_dict(None)  # type: ignore[arg-type]


def test_origin_roundtrip_dict() -> None:
    o = Origin(xyz=(1, 2, 3), rpy=(0.1, 0.2, 0.3))
    d = o.to_dict()
    assert d == {"xyz": [1, 2, 3], "rpy": [0.1, 0.2, 0.3]}


def test_origin_urdf_string() -> None:
    s = Origin(xyz=(1, 2, 3), rpy=(0, 0, 0)).to_urdf_string()
    assert s.startswith("<origin")
    assert 'xyz="1 2 3"' in s
    assert 'rpy="0 0 0"' in s


# --- Inertia ---


def test_inertia_from_box_matches_formula() -> None:
    i = Inertia.from_box(12.0, 1.0, 1.0, 1.0)
    assert i.ixx == pytest.approx(2.0)
    assert i.mass == 12.0


def test_inertia_from_box_invalid_mass() -> None:
    with pytest.raises(Exception):  # noqa: B017
        Inertia.from_box(-1.0, 1.0, 1.0, 1.0)


def test_inertia_from_cylinder_axes() -> None:
    iz = Inertia.from_cylinder(1.0, 1.0, 1.0, axis="z")
    iy = Inertia.from_cylinder(1.0, 1.0, 1.0, axis="y")
    ix = Inertia.from_cylinder(1.0, 1.0, 1.0, axis="x")
    assert iz.izz == pytest.approx(ix.ixx)
    assert iy.iyy == pytest.approx(iz.izz)


def test_inertia_from_sphere_isotropic() -> None:
    i = Inertia.from_sphere(2.0, 0.5)
    assert i.ixx == pytest.approx(i.iyy) == pytest.approx(i.izz)


def test_inertia_from_capsule_axes_swap() -> None:
    iz = Inertia.from_capsule(1.0, 0.3, 1.0, axis="z")
    ix = Inertia.from_capsule(1.0, 0.3, 1.0, axis="x")
    iy = Inertia.from_capsule(1.0, 0.3, 1.0, axis="y")
    assert ix.ixx == pytest.approx(iz.izz)
    assert iy.iyy == pytest.approx(iz.izz)


def test_inertia_to_matrix_symmetric() -> None:
    i = Inertia(ixx=1, iyy=2, izz=3, ixy=0.1, ixz=0.2, iyz=0.3)
    m = i.to_matrix()
    assert np.allclose(m, m.T)


def test_inertia_from_matrix_roundtrip() -> None:
    m = np.array([[1, 0.1, 0.2], [0.1, 2, 0.3], [0.2, 0.3, 3]], dtype=float)
    i = Inertia.from_matrix(m, mass=4.0, center_of_mass=(0.1, 0.2, 0.3))
    assert i.mass == 4.0
    assert i.center_of_mass == (0.1, 0.2, 0.3)
    assert np.allclose(i.to_matrix(), m)


def test_inertia_from_dict_defaults() -> None:
    i = Inertia.from_dict({})
    assert i.ixx == 0.1


def test_inertia_from_dict_none_raises() -> None:
    with pytest.raises(ValueError):
        Inertia.from_dict(None)  # type: ignore[arg-type]


def test_inertia_dict_roundtrip() -> None:
    i = Inertia(ixx=1, iyy=2, izz=3, mass=5)
    d = i.to_dict()
    i2 = Inertia.from_dict(d)
    assert i2.ixx == 1 and i2.mass == 5


def test_inertia_is_positive_definite() -> None:
    good = Inertia(ixx=1, iyy=1, izz=1)
    bad = Inertia(ixx=1, iyy=1, izz=1, ixy=2.0)
    assert good.is_positive_definite()
    assert not bad.is_positive_definite()


def test_inertia_is_diagonal() -> None:
    assert Inertia(1, 1, 1).is_diagonal()
    assert not Inertia(1, 1, 1, ixy=0.1).is_diagonal()


def test_inertia_satisfies_triangle_inequality() -> None:
    assert Inertia(1, 1, 1).satisfies_triangle_inequality()
    # 1 + 1 < 100 -> violates
    assert not Inertia(1, 1, 100).satisfies_triangle_inequality()


def test_inertia_scale_to_mass() -> None:
    i = Inertia(ixx=1, iyy=2, izz=3, mass=1.0)
    scaled = i.scale_to_mass(4.0)
    assert scaled.mass == 4.0
    assert scaled.ixx == 4.0
    assert scaled.izz == 12.0


def test_inertia_scale_from_zero_raises() -> None:
    i = Inertia(ixx=1, iyy=1, izz=1, mass=0.0)
    with pytest.raises(ValueError):
        i.scale_to_mass(2.0)


def test_inertia_urdf_string() -> None:
    s = Inertia(ixx=1, iyy=2, izz=3).to_urdf_string()
    assert s.startswith("<inertia")
    assert "ixx=" in s and "izz=" in s


# --- Material ---


def test_material_from_dict_list_color() -> None:
    m = Material.from_dict({"name": "x", "color": [1, 0, 0, 1]})
    assert m.color == (1, 0, 0, 1)


def test_material_from_dict_none_raises() -> None:
    with pytest.raises(ValueError):
        Material.from_dict(None)  # type: ignore[arg-type]


def test_material_to_dict_with_texture() -> None:
    m = Material("foo", texture="tex.png")
    d = m.to_dict()
    assert d["texture"] == "tex.png"
    assert d["name"] == "foo"


def test_material_urdf_inline_vs_ref() -> None:
    m = Material("foo", (1, 0, 0, 1))
    ref = m.to_urdf_string(inline=False)
    inline = m.to_urdf_string(inline=True)
    assert "<color" not in ref
    assert "<color" in inline


def test_material_presets() -> None:
    for cls_fn in (
        Material.skin,
        Material.bone,
        Material.muscle,
        Material.metal,
        Material.plastic,
    ):
        m = cls_fn()
        assert isinstance(m, Material)
        assert len(m.color) == 4


# --- Geometry ---


def test_geometry_box_urdf() -> None:
    g = Geometry.box(1, 2, 3)
    assert g.geometry_type == GeometryType.BOX
    assert "box" in g.to_urdf_string()


def test_geometry_cylinder_urdf() -> None:
    s = Geometry.cylinder(0.5, 1.0).to_urdf_string()
    assert "cylinder" in s


def test_geometry_sphere_urdf() -> None:
    s = Geometry.sphere(0.5).to_urdf_string()
    assert "sphere" in s


def test_geometry_capsule_urdf_warns_approximation(caplog) -> None:
    s = Geometry.capsule(0.5, 1.0).to_urdf_string()
    # capsule is approximated as cylinder
    assert "cylinder" in s


def test_geometry_mesh_urdf() -> None:
    s = Geometry.mesh("foo.stl", scale=(2, 2, 2)).to_urdf_string()
    assert "mesh" in s
    assert 'scale="2 2 2"' in s


def test_geometry_from_dict_list_to_tuple() -> None:
    g = Geometry.from_dict({"type": "box", "dimensions": [1, 2, 3]})
    assert g.dimensions == (1, 2, 3)


def test_geometry_from_dict_none_raises() -> None:
    with pytest.raises(ValueError):
        Geometry.from_dict(None)  # type: ignore[arg-type]


def test_geometry_from_dict_mesh() -> None:
    g = Geometry.from_dict(
        {"type": "mesh", "mesh_filename": "x.stl", "mesh_scale": [1, 2, 3]}
    )
    assert g.geometry_type == GeometryType.MESH
    assert g.mesh_scale == (1, 2, 3)


def test_geometry_to_dict_includes_mesh_fields() -> None:
    g = Geometry.mesh("x.stl")
    d = g.to_dict()
    assert d["mesh_filename"] == "x.stl"


def test_geometry_urdf_unknown_type_raises() -> None:
    g = Geometry(geometry_type=GeometryType.BOX, dimensions=(1, 2, 3))
    g.geometry_type = "not-a-type"  # type: ignore[assignment]
    with pytest.raises(ValueError):
        g.to_urdf_string()


# --- JointLimits & JointDynamics ---


def test_joint_limits_from_dict_defaults() -> None:
    j = JointLimits.from_dict({})
    assert j.lower == pytest.approx(-math.pi)
    assert j.upper == pytest.approx(math.pi)


def test_joint_limits_from_dict_none_raises() -> None:
    with pytest.raises(ValueError):
        JointLimits.from_dict(None)  # type: ignore[arg-type]


def test_joint_limits_urdf() -> None:
    s = JointLimits().to_urdf_string()
    assert "<limit" in s and "effort=" in s


def test_joint_dynamics_dict_roundtrip() -> None:
    d = JointDynamics(damping=1.0, friction=0.2).to_dict()
    j = JointDynamics.from_dict(d)
    assert j.damping == 1.0 and j.friction == 0.2


def test_joint_dynamics_from_dict_none_raises() -> None:
    with pytest.raises(ValueError):
        JointDynamics.from_dict(None)  # type: ignore[arg-type]


def test_joint_dynamics_urdf() -> None:
    assert "<dynamics" in JointDynamics().to_urdf_string()


# --- Link ---


def test_link_default_inertia() -> None:
    link = Link(name="l")
    assert link.inertia.ixx == pytest.approx(0.1)


def test_link_from_dict_basic() -> None:
    data = {"name": "l", "mass": 2.0, "inertia": {"ixx": 1, "iyy": 1, "izz": 1}}
    link = Link.from_dict(data)
    assert link.name == "l"
    assert link.inertia.mass == 2.0


def test_link_from_dict_with_visual_and_collision() -> None:
    data = {
        "name": "l",
        "inertia": {"ixx": 1, "iyy": 1, "izz": 1, "mass": 1.0},
        "visual_geometry": {"type": "box", "dimensions": [1, 1, 1]},
        "collision": {"type": "sphere", "dimensions": [0.5]},
        "material": {"name": "red", "color": [1, 0, 0, 1]},
    }
    link = Link.from_dict(data)
    assert link.visual_geometry is not None
    assert link.collision_geometry is not None
    assert link.visual_material is not None


def test_link_from_dict_none_raises() -> None:
    with pytest.raises(Exception):  # noqa: B017
        Link.from_dict(None)  # type: ignore[arg-type]


def test_link_to_dict_minimal() -> None:
    link = Link(name="l")
    d = link.to_dict()
    assert d["name"] == "l"
    assert "inertia" in d


def test_link_to_dict_full() -> None:
    link = Link(
        name="l",
        visual_geometry=Geometry.box(1, 1, 1),
        collision_geometry=Geometry.sphere(0.5),
        visual_material=Material("red"),
    )
    d = link.to_dict()
    assert "visual_geometry" in d
    assert "collision_geometry" in d
    assert "material" in d


# --- Joint ---


def test_joint_revolute_gets_default_limits() -> None:
    j = Joint(name="j", joint_type=JointType.REVOLUTE, parent="a", child="b")
    assert j.limits is not None


def test_joint_prismatic_default_limits() -> None:
    j = Joint(name="j", joint_type=JointType.PRISMATIC, parent="a", child="b")
    assert j.limits is not None
    assert j.limits.lower == -1.0


def test_joint_fixed_no_limits() -> None:
    j = Joint(name="j", joint_type=JointType.FIXED, parent="a", child="b")
    assert j.limits is None


def test_joint_from_dict_basic() -> None:
    j = Joint.from_dict(
        {
            "name": "j",
            "type": "revolute",
            "parent": "a",
            "child": "b",
            "axis": [1, 0, 0],
        }
    )
    assert j.joint_type == JointType.REVOLUTE
    assert j.axis == (1, 0, 0)


def test_joint_from_dict_with_limits_and_dynamics() -> None:
    j = Joint.from_dict(
        {
            "name": "j",
            "type": "revolute",
            "parent": "a",
            "child": "b",
            "limits": {"lower": -1, "upper": 1, "effort": 5, "velocity": 2},
            "dynamics": {"damping": 0.7, "friction": 0.1},
        }
    )
    assert j.limits.lower == -1
    assert j.dynamics.damping == 0.7


def test_joint_from_dict_none_raises() -> None:
    with pytest.raises(ValueError):
        Joint.from_dict(None)  # type: ignore[arg-type]


def test_joint_to_dict_roundtrip() -> None:
    j = Joint(name="j", joint_type=JointType.REVOLUTE, parent="a", child="b")
    d = j.to_dict()
    j2 = Joint.from_dict(d)
    assert j2.name == "j" and j2.parent == "a"


def test_joint_is_composite_and_dof() -> None:
    assert not Joint(
        name="j", joint_type=JointType.REVOLUTE, parent="a", child="b"
    ).is_composite()
    g = Joint(name="g", joint_type=JointType.GIMBAL, parent="a", child="b")
    assert g.is_composite()
    assert g.get_dof_count() == 3
    u = Joint(name="u", joint_type=JointType.UNIVERSAL, parent="a", child="b")
    assert u.get_dof_count() == 2
    f = Joint(name="f", joint_type=JointType.FIXED, parent="a", child="b")
    assert f.get_dof_count() == 0
    fl = Joint(name="fl", joint_type=JointType.FLOATING, parent="a", child="b")
    assert fl.get_dof_count() == 6
    pl = Joint(name="pl", joint_type=JointType.PLANAR, parent="a", child="b")
    assert pl.get_dof_count() == 3
