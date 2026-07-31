"""Unit tests for the URDF ``<inertial>`` reader / writer pair.

These two functions are inverses, so the bulk of the suite is a
round-trip table: build a representative :class:`SegmentProperties`
for each shape kind, write it to XML, re-read it, and assert the
result is bitwise (within ``rtol=1e-9``) identical for every
numeric field.

The remaining cases cover the documented error modes:

* missing ``<inertia>`` child  → ``ValueError``
* missing ``<mass>`` child     → ``ValueError``
* missing required attribute   → ``ValueError``
* non-numeric attribute        → ``ValueError``
* non-Element argument         → ``TypeError``
* SegmentProperties rejects negative mass at construction time, so
  we verify the writer's defensive check by bypassing dataclass
  construction with a stand-in stub.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pytest

from anthropometrics import SegmentProperties
from anthropometrics.readers import read_urdf_inertial
from anthropometrics.writers import write_urdf_inertial


# --------------------------------------------------------------------------- #
# Fixture builders.                                                           #
# --------------------------------------------------------------------------- #
def _segment(
    *,
    com: tuple[float, float, float],
    inertia: np.ndarray,
    mass_kg: float = 2.0,
    length_m: float = 0.30,
    name: str = "test_segment",
) -> SegmentProperties:
    """Build a valid :class:`SegmentProperties` for round-trip tests."""
    return SegmentProperties(
        name=name,
        body_part_id="test",
        length_m=length_m,
        proximal_marker=None,
        distal_marker=None,
        mass_kg=mass_kg,
        com_xyz_m=np.asarray(com, dtype=float),
        inertia_tensor=inertia,
        source_method="unit_test",
        source_subject_height_m=1.80,
        source_subject_mass_kg=75.0,
    )


def _solid_cylinder_inertia(mass: float, radius: float, length: float) -> np.ndarray:
    """Solid cylinder, axis = local +z."""
    perp = (1.0 / 12.0) * mass * (3.0 * radius**2 + length**2)
    axial = 0.5 * mass * radius**2
    return np.diag([perp, perp, axial]).astype(float)


def _solid_ellipsoid_inertia(mass: float, a: float, b: float, c: float) -> np.ndarray:
    """Solid ellipsoid with semi-axes (a, b, c) along (x, y, z)."""
    return np.diag(
        [
            (1.0 / 5.0) * mass * (b**2 + c**2),
            (1.0 / 5.0) * mass * (a**2 + c**2),
            (1.0 / 5.0) * mass * (a**2 + b**2),
        ]
    ).astype(float)


def _solid_sphere_inertia(mass: float, radius: float) -> np.ndarray:
    val = (2.0 / 5.0) * mass * radius**2
    return np.diag([val, val, val]).astype(float)


def _solid_box_inertia(mass: float, w: float, h: float, d: float) -> np.ndarray:
    """Box with full dimensions w (x), h (y), d (z)."""
    return np.diag(
        [
            (1.0 / 12.0) * mass * (h**2 + d**2),
            (1.0 / 12.0) * mass * (w**2 + d**2),
            (1.0 / 12.0) * mass * (w**2 + h**2),
        ]
    ).astype(float)


def _full_off_diagonal_tensor(scale: float = 1.0) -> np.ndarray:
    """A symmetric positive-definite tensor with non-zero off-diagonals."""
    return (
        np.array(
            [
                [0.05, 0.005, -0.003],
                [0.005, 0.04, 0.002],
                [-0.003, 0.002, 0.06],
            ],
            dtype=float,
        )
        * scale
    )


# --------------------------------------------------------------------------- #
# Round-trip table — one parametrised test per shape kind.                    #
# --------------------------------------------------------------------------- #
_ROUND_TRIP_CASES: list[tuple[str, SegmentProperties]] = [
    (
        "cylinder_at_origin",
        _segment(
            com=(0.0, 0.0, 0.15),
            inertia=_solid_cylinder_inertia(mass=2.0, radius=0.04, length=0.30),
        ),
    ),
    (
        "cylinder_offset_com",
        _segment(
            com=(0.01, -0.02, 0.12),
            inertia=_solid_cylinder_inertia(mass=3.5, radius=0.05, length=0.40),
            mass_kg=3.5,
            length_m=0.40,
        ),
    ),
    (
        "ellipsoid",
        _segment(
            com=(0.05, 0.0, 0.0),
            inertia=_solid_ellipsoid_inertia(mass=1.5, a=0.06, b=0.04, c=0.05),
            mass_kg=1.5,
        ),
    ),
    (
        "sphere",
        _segment(
            com=(0.0, 0.0, 0.0),
            inertia=_solid_sphere_inertia(mass=4.0, radius=0.08),
            mass_kg=4.0,
        ),
    ),
    (
        "box",
        _segment(
            com=(0.0, 0.0, 0.0),
            inertia=_solid_box_inertia(mass=2.5, w=0.10, h=0.05, d=0.20),
            mass_kg=2.5,
        ),
    ),
    (
        "off_diagonal",
        _segment(
            com=(-0.01, 0.02, 0.03),
            inertia=_full_off_diagonal_tensor(),
        ),
    ),
    (
        "tiny_values",
        _segment(
            com=(1e-7, 0.0, 0.0),
            inertia=np.diag([1e-6, 2e-6, 1.5e-6]).astype(float),
            mass_kg=1e-3,
        ),
    ),
    (
        "large_values",
        _segment(
            com=(0.5, -0.4, 0.3),
            inertia=np.diag([12.0, 15.0, 10.0]).astype(float),
            mass_kg=120.0,
            length_m=1.5,
        ),
    ),
]


@pytest.mark.parametrize(
    "case_id,props",
    _ROUND_TRIP_CASES,
    ids=[case[0] for case in _ROUND_TRIP_CASES],
)
def test_round_trip_preserves_record(case_id: str, props: SegmentProperties) -> None:
    """Write -> read -> identical numeric fields, ``rtol=1e-9 atol=1e-12``."""
    elem = write_urdf_inertial(props)

    restored = read_urdf_inertial(
        elem,
        name=props.name,
        body_part_id=props.body_part_id,
        length_m=props.length_m,
        proximal_marker=props.proximal_marker,
        distal_marker=props.distal_marker,
        source_method=props.source_method,
        source_subject_height_m=props.source_subject_height_m,
        source_subject_mass_kg=props.source_subject_mass_kg,
    )

    assert restored.mass_kg == pytest.approx(
        props.mass_kg, rel=1e-9, abs=1e-12
    ), case_id
    np.testing.assert_allclose(
        restored.com_xyz_m, props.com_xyz_m, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        restored.inertia_tensor,
        props.inertia_tensor,
        rtol=1e-9,
        atol=1e-12,
    )
    # Metadata fields are passed through untouched.
    assert restored.name == props.name
    assert restored.body_part_id == props.body_part_id
    assert restored.length_m == props.length_m
    assert restored.source_method == props.source_method


# --------------------------------------------------------------------------- #
# Writer output structure.                                                    #
# --------------------------------------------------------------------------- #
def test_writer_produces_inertial_with_three_children() -> None:
    props = _segment(
        com=(0.1, 0.2, 0.3),
        inertia=np.diag([0.01, 0.01, 0.01]).astype(float),
    )
    elem = write_urdf_inertial(props)
    assert elem.tag == "inertial"
    tags = [child.tag for child in elem]
    assert tags == ["origin", "mass", "inertia"]


def test_writer_origin_xyz_has_three_components() -> None:
    props = _segment(
        com=(0.11, -0.22, 0.33),
        inertia=np.diag([0.02, 0.02, 0.02]).astype(float),
    )
    elem = write_urdf_inertial(props)
    origin = elem.find("origin")
    assert origin is not None
    parts = origin.get("xyz", "").split()
    assert len(parts) == 3
    assert [float(p) for p in parts] == pytest.approx([0.11, -0.22, 0.33])


def test_writer_inertia_has_six_components() -> None:
    props = _segment(
        com=(0.0, 0.0, 0.0),
        inertia=_full_off_diagonal_tensor(),
    )
    elem = write_urdf_inertial(props)
    inertia = elem.find("inertia")
    assert inertia is not None
    for attr in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        assert attr in inertia.attrib, attr


def test_writer_mass_attribute_is_value() -> None:
    props = _segment(
        com=(0.0, 0.0, 0.0),
        inertia=np.diag([0.01, 0.01, 0.01]).astype(float),
        mass_kg=7.25,
    )
    elem = write_urdf_inertial(props)
    mass = elem.find("mass")
    assert mass is not None
    assert float(mass.get("value", "nan")) == pytest.approx(7.25)


# --------------------------------------------------------------------------- #
# Reader: accepting parent or inertial element.                               #
# --------------------------------------------------------------------------- #
def test_reader_accepts_link_parent() -> None:
    """Reader must accept a ``<link>`` containing a single ``<inertial>``."""
    props = _segment(
        com=(0.0, 0.05, 0.0),
        inertia=np.diag([0.01, 0.02, 0.015]).astype(float),
    )
    inertial = write_urdf_inertial(props)
    link = ET.Element("link", {"name": "wrapped"})
    link.append(inertial)

    restored = read_urdf_inertial(link)
    np.testing.assert_allclose(
        restored.inertia_tensor, props.inertia_tensor, rtol=1e-9, atol=1e-12
    )


def test_reader_origin_defaults_to_zero_when_missing() -> None:
    """URDF allows ``<origin>`` to be omitted entirely (defaults to zeros)."""
    xml = (
        "<inertial>"
        '  <mass value="1.5"/>'
        '  <inertia ixx="0.01" ixy="0.0" ixz="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    restored = read_urdf_inertial(elem)
    np.testing.assert_array_equal(restored.com_xyz_m, np.zeros(3))


def test_reader_origin_without_xyz_defaults_to_zero() -> None:
    """An ``<origin>`` without ``xyz`` defaults to ``"0 0 0"``."""
    xml = (
        "<inertial>"
        '  <origin rpy="0 0 0"/>'
        '  <mass value="1.5"/>'
        '  <inertia ixx="0.01" ixy="0.0" ixz="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    restored = read_urdf_inertial(elem)
    np.testing.assert_array_equal(restored.com_xyz_m, np.zeros(3))


# --------------------------------------------------------------------------- #
# Error paths.                                                                #
# --------------------------------------------------------------------------- #
def test_missing_inertia_child_raises() -> None:
    xml = '<inertial>  <origin xyz="0 0 0"/>  <mass value="1.0"/></inertial>'
    elem = ET.fromstring(xml)
    with pytest.raises(ValueError, match="<inertia>"):
        read_urdf_inertial(elem)


def test_missing_mass_child_raises() -> None:
    xml = (
        "<inertial>"
        '  <origin xyz="0 0 0"/>'
        '  <inertia ixx="0.01" ixy="0.0" ixz="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    with pytest.raises(ValueError, match="<mass>"):
        read_urdf_inertial(elem)


def test_missing_inertia_attribute_raises() -> None:
    xml = (
        "<inertial>"
        '  <mass value="1.0"/>'
        '  <inertia ixx="0.01" ixy="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'  # ixz missing
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    with pytest.raises(ValueError, match="ixz"):
        read_urdf_inertial(elem)


def test_non_numeric_inertia_attribute_raises() -> None:
    xml = (
        "<inertial>"
        '  <mass value="1.0"/>'
        '  <inertia ixx="not-a-number" ixy="0.0" ixz="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    with pytest.raises(ValueError, match="not a valid float"):
        read_urdf_inertial(elem)


def test_non_numeric_mass_raises() -> None:
    xml = (
        "<inertial>"
        '  <mass value="oops"/>'
        '  <inertia ixx="0.01" ixy="0.0" ixz="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    with pytest.raises(ValueError, match="not a valid float"):
        read_urdf_inertial(elem)


def test_origin_xyz_wrong_arity_raises() -> None:
    xml = (
        "<inertial>"
        '  <origin xyz="0 0"/>'
        '  <mass value="1.0"/>'
        '  <inertia ixx="0.01" ixy="0.0" ixz="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    with pytest.raises(ValueError, match="3 space-separated"):
        read_urdf_inertial(elem)


def test_origin_xyz_non_numeric_raises() -> None:
    xml = (
        "<inertial>"
        '  <origin xyz="0 nope 0"/>'
        '  <mass value="1.0"/>'
        '  <inertia ixx="0.01" ixy="0.0" ixz="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    with pytest.raises(ValueError, match="non-numeric"):
        read_urdf_inertial(elem)


def test_negative_mass_raises_value_error() -> None:
    """URDF input with a negative ``<mass value=...>`` rejected by reader."""
    xml = (
        "<inertial>"
        '  <mass value="-1.5"/>'
        '  <inertia ixx="0.01" ixy="0.0" ixz="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    with pytest.raises(ValueError, match="must be positive"):
        read_urdf_inertial(elem)


def test_zero_mass_raises_value_error() -> None:
    xml = (
        "<inertial>"
        '  <mass value="0.0"/>'
        '  <inertia ixx="0.01" ixy="0.0" ixz="0.0"'
        '           iyy="0.02" iyz="0.0" izz="0.015"/>'
        "</inertial>"
    )
    elem = ET.fromstring(xml)
    with pytest.raises(ValueError, match="must be positive"):
        read_urdf_inertial(elem)


def test_non_element_input_raises_type_error() -> None:
    with pytest.raises(TypeError, match="Element"):
        read_urdf_inertial("<inertial/>")  # type: ignore[arg-type]


def test_wrapper_without_inertial_child_raises() -> None:
    link = ET.Element("link", {"name": "no_inertia"})
    with pytest.raises(ValueError, match="<inertial>"):
        read_urdf_inertial(link)


# --------------------------------------------------------------------------- #
# Writer error paths.                                                         #
# --------------------------------------------------------------------------- #
def test_writer_rejects_non_segment_input() -> None:
    with pytest.raises(TypeError, match="SegmentProperties"):
        write_urdf_inertial("not a segment")  # type: ignore[arg-type]


def test_writer_rejects_non_positive_mass_via_stub() -> None:
    """The dataclass blocks bad mass at construction; the writer also defends.

    To reach the writer's defensive ``mass_kg <= 0`` check we hand it a
    duck-typed stand-in that passes ``isinstance(SegmentProperties)``
    via a subclass that bypasses ``__post_init__``.
    """

    class _StubProps(SegmentProperties):
        # Bypass the parent dataclass invariants so we can inject a
        # non-positive mass solely to exercise the writer's guard.
        def __post_init__(self) -> None:  # noqa: D401 - intentional override
            object.__setattr__(self, "com_xyz_m", np.zeros(3))
            object.__setattr__(self, "inertia_tensor", np.eye(3) * 0.01)

    bad: Any = _StubProps(
        name="x",
        body_part_id="x",
        length_m=1.0,
        proximal_marker=None,
        distal_marker=None,
        mass_kg=-1.0,
        com_xyz_m=np.zeros(3),
        inertia_tensor=np.eye(3) * 0.01,
        source_method="x",
        source_subject_height_m=1.0,
        source_subject_mass_kg=1.0,
    )
    with pytest.raises(ValueError, match="must be positive"):
        write_urdf_inertial(bad)
