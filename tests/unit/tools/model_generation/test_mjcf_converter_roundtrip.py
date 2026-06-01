"""Round-trip and element-parsing tests for the MJCF/URDF converter.

Covers issue #7000:
- urdf_to_mjcf -> mjcf_to_urdf preserves body count, joint types, masses, inertia
- _parse_body_inertial (diagonal + full inertia)
- _parse_mjcf_geom for box / sphere / cylinder / mesh
- malformed XML -> error
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import defusedxml.ElementTree as DefusedET
import pytest
from model_generation.converters.mjcf_converter import MJCFConverter
from model_generation.core.types import GeometryType, JointType

# A two-link arm: revolute joint, box + cylinder visuals, diagonal inertia.
ROUNDTRIP_URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="base">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" iyy="0.2" izz="0.3" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual><geometry><box size="0.2 0.3 0.4"/></geometry></visual>
  </link>
  <link name="link1">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual><geometry><cylinder radius="0.05" length="0.4"/></geometry></visual>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0.5"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="1"/>
  </joint>
</robot>
"""


def _element(xml_str: str) -> ET.Element:
    """Parse an MJCF snippet into an Element via defusedxml (matches module)."""
    return DefusedET.fromstring(xml_str)


@pytest.fixture
def converter() -> MJCFConverter:
    return MJCFConverter()


class TestRoundTrip:
    """parse -> emit -> parse preserves model structure."""

    def test_roundtrip_preserves_body_count(self, converter: MJCFConverter) -> None:
        original = converter._urdf_parser.parse(ROUNDTRIP_URDF)
        mjcf = converter.urdf_to_mjcf(ROUNDTRIP_URDF)
        reparsed = converter._parse_mjcf(_element(mjcf))

        assert len(reparsed.links) == len(original.links) == 2
        assert {link.name for link in reparsed.links} == {"base", "link1"}

    def test_roundtrip_preserves_joint_type_and_name(
        self, converter: MJCFConverter
    ) -> None:
        mjcf = converter.urdf_to_mjcf(ROUNDTRIP_URDF)
        reparsed = converter._parse_mjcf(_element(mjcf))

        assert len(reparsed.joints) == 1
        joint = reparsed.joints[0]
        assert joint.name == "j1"
        assert joint.joint_type == JointType.REVOLUTE
        assert joint.parent == "base"
        assert joint.child == "link1"

    def test_roundtrip_preserves_masses(self, converter: MJCFConverter) -> None:
        mjcf = converter.urdf_to_mjcf(ROUNDTRIP_URDF)
        reparsed = converter._parse_mjcf(_element(mjcf))
        masses = {link.name: link.inertia.mass for link in reparsed.links}

        assert masses["base"] == pytest.approx(2.0)
        assert masses["link1"] == pytest.approx(1.0)

    def test_roundtrip_preserves_diagonal_inertia(
        self, converter: MJCFConverter
    ) -> None:
        mjcf = converter.urdf_to_mjcf(ROUNDTRIP_URDF)
        reparsed = converter._parse_mjcf(_element(mjcf))
        base = next(link for link in reparsed.links if link.name == "base")

        assert base.inertia.ixx == pytest.approx(0.1)
        assert base.inertia.iyy == pytest.approx(0.2)
        assert base.inertia.izz == pytest.approx(0.3)

    def test_mjcf_to_urdf_returns_valid_xml(self, converter: MJCFConverter) -> None:
        mjcf = converter.urdf_to_mjcf(ROUNDTRIP_URDF)
        urdf = converter.mjcf_to_urdf(mjcf)
        # Result must itself be parseable URDF naming both links.
        root = DefusedET.fromstring(urdf)
        link_names = {link.get("name") for link in root.findall("link")}
        assert {"base", "link1"} <= link_names


class TestParseBodyInertial:
    """_parse_body_inertial handles diagonal, full, and missing inertia."""

    def test_diagonal_inertia(self) -> None:
        body = _element(
            '<body name="b">'
            '<inertial mass="3.0" pos="0.1 0.2 0.3" diaginertia="0.4 0.5 0.6"/>'
            "</body>"
        )
        inertia = MJCFConverter._parse_body_inertial(body)
        assert inertia.mass == pytest.approx(3.0)
        assert inertia.ixx == pytest.approx(0.4)
        assert inertia.iyy == pytest.approx(0.5)
        assert inertia.izz == pytest.approx(0.6)
        assert inertia.center_of_mass == pytest.approx((0.1, 0.2, 0.3))

    def test_full_inertia(self) -> None:
        body = _element(
            '<body name="b">'
            '<inertial mass="1.5" fullinertia="0.1 0.2 0.3 0.01 0.02 0.03"/>'
            "</body>"
        )
        inertia = MJCFConverter._parse_body_inertial(body)
        assert inertia.mass == pytest.approx(1.5)
        assert inertia.ixx == pytest.approx(0.1)
        assert inertia.ixy == pytest.approx(0.01)
        assert inertia.ixz == pytest.approx(0.02)
        assert inertia.iyz == pytest.approx(0.03)

    def test_missing_inertial_returns_default(self) -> None:
        body = _element('<body name="b"/>')
        inertia = MJCFConverter._parse_body_inertial(body)
        assert inertia.mass == pytest.approx(1.0)
        assert inertia.ixx == pytest.approx(0.1)


class TestParseMjcfGeom:
    """_parse_mjcf_geom decodes each primitive (MuJoCo uses half-sizes)."""

    def test_box(self, converter: MJCFConverter) -> None:
        geom, _ = converter._parse_mjcf_geom(
            _element('<geom type="box" size="0.1 0.15 0.2"/>')
        )
        assert geom is not None
        assert geom.geometry_type == GeometryType.BOX
        # Half-sizes are doubled back to full dimensions.
        assert geom.dimensions == pytest.approx((0.2, 0.3, 0.4))

    def test_sphere(self, converter: MJCFConverter) -> None:
        geom, _ = converter._parse_mjcf_geom(
            _element('<geom type="sphere" size="0.25"/>')
        )
        assert geom is not None
        assert geom.geometry_type == GeometryType.SPHERE
        assert geom.dimensions[0] == pytest.approx(0.25)

    def test_cylinder(self, converter: MJCFConverter) -> None:
        geom, _ = converter._parse_mjcf_geom(
            _element('<geom type="cylinder" size="0.05 0.2"/>')
        )
        assert geom is not None
        assert geom.geometry_type == GeometryType.CYLINDER
        radius, length = geom.dimensions
        assert radius == pytest.approx(0.05)
        # Half-length doubled back to full length.
        assert length == pytest.approx(0.4)

    def test_geom_position_origin(self, converter: MJCFConverter) -> None:
        _, origin = converter._parse_mjcf_geom(
            _element('<geom type="sphere" size="0.1" pos="1 2 3"/>')
        )
        assert origin.xyz == pytest.approx((1.0, 2.0, 3.0))

    def test_unknown_geom_returns_none(self, converter: MJCFConverter) -> None:
        geom, _ = converter._parse_mjcf_geom(
            _element('<geom type="ellipsoid" size="1 2 3"/>')
        )
        assert geom is None


class TestMalformedInput:
    """Malformed / empty input raises rather than silently succeeding."""

    def test_malformed_xml_raises(self, converter: MJCFConverter) -> None:
        with pytest.raises(ET.ParseError):
            converter.mjcf_to_urdf("<mujoco><body></mujoco>")

    def test_none_source_urdf_to_mjcf_raises(self, converter: MJCFConverter) -> None:
        with pytest.raises(ValueError, match="source must be provided"):
            converter.urdf_to_mjcf(None)  # type: ignore[arg-type]

    def test_none_source_mjcf_to_urdf_raises(self, converter: MJCFConverter) -> None:
        with pytest.raises(ValueError, match="source must be provided"):
            converter.mjcf_to_urdf(None)  # type: ignore[arg-type]
