"""Round-trip URDF generation and semantic equality tests (#1064).

Tests that a URDF parsed -> to_urdf() -> re-parsed produces
semantically identical models.

Design by Contract
------------------
- parse(to_urdf(model)) ≡ model (semantic equality)
- Link names, joint names, geometry types preserved through round-trip
- Inertia values, joint limits, origins preserved with bounded precision
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from model_generation.converters.urdf_parser import ParsedModel

# Minimal URDF for round-trip testing
MINIMAL_URDF = """\
<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.2" izz="0.3" ixy="0.01" ixz="0.02" iyz="0.03"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.2 0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0.0 0.0 1.0 1.0"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <box size="0.1 0.2 0.3"/>
      </geometry>
    </collision>
  </link>

  <link name="child_link">
    <inertial>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.05" iyy="0.06" izz="0.07" ixy="0.0" ixz="0.0" iyz="0.0"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.2"/>
      </geometry>
    </visual>
  </link>

  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="child_link"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5708" upper="1.5708" effort="100" velocity="3.14"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>
</robot>
"""

MULTI_JOINT_URDF = """\
<?xml version="1.0"?>
<robot name="three_link_robot">
  <link name="base"/>
  <link name="link_a"/>
  <link name="link_b"/>

  <joint name="joint_ab" type="revolute">
    <parent link="base"/>
    <child link="link_a"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="5"/>
  </joint>

  <joint name="joint_bc" type="continuous">
    <parent link="link_a"/>
    <child link="link_b"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"""


class TestURDFRoundTrip:
    """Parse -> to_urdf -> re-parse must preserve semantics."""

    def _round_trip(self, urdf_str: str) -> tuple[ParsedModel, ParsedModel]:
        """Parse, write, re-parse, return both models."""
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser(resolve_meshes=False)
        model_a = parser.parse_string(urdf_str)
        xml_b = model_a.to_urdf()
        model_b = parser.parse_string(xml_b)
        return model_a, model_b

    def test_round_trip_preserves_robot_name(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        assert b.name == a.name == "test_robot"

    def test_round_trip_preserves_link_count(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        assert len(b.links) == len(a.links)

    def test_round_trip_preserves_link_names(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        names_a = {link.name for link in a.links}
        names_b = {link.name for link in b.links}
        assert names_b == names_a

    def test_round_trip_preserves_joint_count(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        assert len(b.joints) == len(a.joints)

    def test_round_trip_preserves_joint_names(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        names_a = {j.name for j in a.joints}
        names_b = {j.name for j in b.joints}
        assert names_b == names_a

    def test_round_trip_preserves_joint_type(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        for ja, jb in zip(
            sorted(a.joints, key=lambda j: j.name),
            sorted(b.joints, key=lambda j: j.name),
            strict=True,
        ):
            assert jb.joint_type == ja.joint_type

    def test_round_trip_preserves_parent_child(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        for ja, jb in zip(
            sorted(a.joints, key=lambda j: j.name),
            sorted(b.joints, key=lambda j: j.name),
            strict=True,
        ):
            assert jb.parent == ja.parent
            assert jb.child == ja.child

    def test_round_trip_preserves_inertia_mass(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        for la, lb in zip(
            sorted(a.links, key=lambda lnk: lnk.name),
            sorted(b.links, key=lambda lnk: lnk.name),
            strict=True,
        ):
            assert lb.inertia.mass == pytest.approx(la.inertia.mass, abs=1e-6)

    def test_round_trip_preserves_joint_limits(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        ja = a.get_joint("joint_1")
        jb = b.get_joint("joint_1")
        assert ja is not None and jb is not None
        assert ja.limits is not None and jb.limits is not None
        assert jb.limits.lower == pytest.approx(ja.limits.lower, abs=1e-4)
        assert jb.limits.upper == pytest.approx(ja.limits.upper, abs=1e-4)
        assert jb.limits.effort == pytest.approx(ja.limits.effort, abs=1e-4)

    def test_round_trip_preserves_dynamics(self) -> None:
        a, b = self._round_trip(MINIMAL_URDF)
        ja = a.get_joint("joint_1")
        jb = b.get_joint("joint_1")
        assert ja is not None and jb is not None
        assert jb.dynamics.damping == pytest.approx(ja.dynamics.damping, abs=1e-6)
        assert jb.dynamics.friction == pytest.approx(ja.dynamics.friction, abs=1e-6)


class TestURDFMultiJointRoundTrip:
    """Multi-joint models round-trip correctly."""

    def test_three_link_preserves_topology(self) -> None:
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser(resolve_meshes=False)
        model_a = parser.parse_string(MULTI_JOINT_URDF)
        xml_b = model_a.to_urdf()
        model_b = parser.parse_string(xml_b)

        assert len(model_b.links) == 3
        assert len(model_b.joints) == 2

        names_a = {j.name for j in model_a.joints}
        names_b = {j.name for j in model_b.joints}
        assert names_b == names_a

    def test_root_link_preserved(self) -> None:
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser(resolve_meshes=False)
        model_a = parser.parse_string(MULTI_JOINT_URDF)
        xml_b = model_a.to_urdf()
        model_b = parser.parse_string(xml_b)

        root_a = model_a.get_root_link()
        root_b = model_b.get_root_link()
        assert root_a is not None and root_b is not None
        assert root_b.name == root_a.name == "base"


class TestURDFSemanticEquality:
    """Semantic equality checks for parsed models."""

    def test_parsed_output_is_valid_xml(self) -> None:
        """to_urdf() must produce valid XML."""
        import defusedxml.ElementTree as ET
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser(resolve_meshes=False)
        model = parser.parse_string(MINIMAL_URDF)
        xml_str = model.to_urdf()
        root = ET.fromstring(xml_str)
        assert root.tag == "robot"

    def test_to_urdf_contains_all_links(self) -> None:
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser(resolve_meshes=False)
        model = parser.parse_string(MINIMAL_URDF)
        xml_str = model.to_urdf()
        assert "base_link" in xml_str
        assert "child_link" in xml_str

    def test_to_urdf_contains_all_joints(self) -> None:
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser(resolve_meshes=False)
        model = parser.parse_string(MINIMAL_URDF)
        xml_str = model.to_urdf()
        assert "joint_1" in xml_str
        assert "revolute" in xml_str

    def test_copy_produces_independent_model(self) -> None:
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser(resolve_meshes=False)
        model = parser.parse_string(MINIMAL_URDF)
        copy = model.copy()

        # Modify copy, original should be unchanged
        copy.name = "modified"
        assert model.name == "test_robot"
        assert copy.name == "modified"


class TestURDFMeshPathHandling:
    """Validation around mesh filename handling in the URDF parser."""

    def _write_urdf(self, directory: Path, filename: str) -> Path:
        urdf_path = directory / "robot.urdf"
        urdf_text = f"""
        <robot name="mesh-path-test">
          <link name="base">
            <visual>
              <geometry>
                <mesh filename="{filename}" scale="1 1 1"/>
              </geometry>
            </visual>
          </link>
        </robot>
        """
        urdf_path.write_text(urdf_text)
        return urdf_path

    def test_traversal_mesh_path_is_not_resolved(self, tmp_path: Path) -> None:
        from model_generation.converters.urdf_parser import URDFParser

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside_mesh = tmp_path / "outside.stl"
        outside_mesh.write_text("placeholder")

        urdf_path = self._write_urdf(workspace, "../outside.stl")
        model = URDFParser().parse(str(urdf_path))
        mesh_filename = model.links[0].visual_geometry.mesh_filename

        assert mesh_filename == "../outside.stl"

    def test_unsupported_uri_mesh_path_is_preserved_as_text(
        self, tmp_path: Path
    ) -> None:
        from model_generation.converters.urdf_parser import URDFParser

        urdf_path = self._write_urdf(tmp_path, "http://example.com/mesh.stl")
        model = URDFParser().parse(str(urdf_path))
        mesh_filename = model.links[0].visual_geometry.mesh_filename

        assert mesh_filename == "http://example.com/mesh.stl"


# URDF fixture used for the MJCF<->URDF bidirectional round-trip. Kept
# deliberately small so the *preserved* invariants are unambiguous; the lossy
# fields are asserted explicitly below.
MJCF_ROUNDTRIP_URDF = """\
<?xml version="1.0"?>
<robot name="rt_robot">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" iyy="0.2" izz="0.3" ixy="0.0" ixz="0.0" iyz="0.0"/>
    </inertial>
    <visual>
      <geometry><box size="0.1 0.2 0.3"/></geometry>
    </visual>
  </link>
  <link name="link_a">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.05" iyy="0.06" izz="0.07" ixy="0.0" ixz="0.0" iyz="0.0"/>
    </inertial>
    <visual>
      <geometry><cylinder radius="0.03" length="0.2"/></geometry>
    </visual>
  </link>
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="link_a"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5708" upper="1.5708" effort="100" velocity="3.14"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>
</robot>
"""


class TestMJCFURDFRoundTrip:
    """URDF -> MJCF -> URDF bidirectional round-trip (#3174).

    MJCF (MuJoCo) and URDF do not have a 1:1 feature mapping. This suite pins
    down both what *is* preserved (topology, names, joint type, joint limit
    range, mass, geometry types) and what is *lossy*, asserting the lossy
    fields rather than skipping them so the loss is documented and
    regression-guarded.

    Documented lossy joint mappings (URDF -> MJCF -> URDF):

    - ``effort`` and ``velocity`` joint limits have no MJCF equivalent and are
      reset to the URDF parser's defaults (1000.0 / 10.0) on the return trip.
    - Joint ``friction`` has no MJCF equivalent and is lost (returns 0.0);
      ``damping`` is representable and is preserved.
    - ``revolute`` and ``continuous`` both map to MJCF ``hinge`` and therefore
      both return as ``revolute`` (continuous -> revolute is lossy).
    """

    def _round_trip(self) -> tuple[ParsedModel, ParsedModel, str]:
        from model_generation.converters.mjcf_converter import MJCFConverter
        from model_generation.converters.urdf_parser import URDFParser

        parser = URDFParser(resolve_meshes=False)
        converter = MJCFConverter()

        model_a = parser.parse_string(MJCF_ROUNDTRIP_URDF)
        mjcf_xml = converter.urdf_to_mjcf(model_a)
        urdf_back = converter.mjcf_to_urdf(mjcf_xml)
        model_b = parser.parse_string(urdf_back)
        return model_a, model_b, mjcf_xml

    # --- Preserved invariants ------------------------------------------------

    def test_intermediate_mjcf_is_well_formed(self) -> None:
        import defusedxml.ElementTree as ET

        _, _, mjcf_xml = self._round_trip()
        root = ET.fromstring(mjcf_xml)
        assert root.tag == "mujoco"
        assert root.get("model") == "rt_robot"

    def test_round_trip_preserves_link_count_and_names(self) -> None:
        a, b, _ = self._round_trip()
        assert len(b.links) == len(a.links) == 2
        assert {link.name for link in b.links} == {link.name for link in a.links}

    def test_round_trip_preserves_joint_count_and_names(self) -> None:
        a, b, _ = self._round_trip()
        assert len(b.joints) == len(a.joints) == 1
        assert {j.name for j in b.joints} == {j.name for j in a.joints}

    def test_round_trip_preserves_parent_child_topology(self) -> None:
        a, b, _ = self._round_trip()
        ja = a.get_joint("joint_1")
        jb = b.get_joint("joint_1")
        assert ja is not None and jb is not None
        assert jb.parent == ja.parent == "base_link"
        assert jb.child == ja.child == "link_a"

    def test_round_trip_preserves_revolute_joint_type(self) -> None:
        from model_generation.core.types import JointType

        _, b, _ = self._round_trip()
        jb = b.get_joint("joint_1")
        assert jb is not None
        assert jb.joint_type == JointType.REVOLUTE

    def test_round_trip_preserves_joint_limit_range(self) -> None:
        a, b, _ = self._round_trip()
        ja = a.get_joint("joint_1")
        jb = b.get_joint("joint_1")
        assert ja.limits is not None and jb.limits is not None
        assert jb.limits.lower == pytest.approx(ja.limits.lower, abs=1e-4)
        assert jb.limits.upper == pytest.approx(ja.limits.upper, abs=1e-4)

    def test_round_trip_preserves_link_mass(self) -> None:
        a, b, _ = self._round_trip()
        for la, lb in zip(
            sorted(a.links, key=lambda lnk: lnk.name),
            sorted(b.links, key=lambda lnk: lnk.name),
            strict=True,
        ):
            assert lb.inertia.mass == pytest.approx(la.inertia.mass, abs=1e-6)

    def test_round_trip_preserves_visual_geometry_types(self) -> None:
        from model_generation.core.types import GeometryType

        _, b, _ = self._round_trip()
        geoms = {
            link.name: link.visual_geometry.geometry_type
            for link in b.links
            if link.visual_geometry is not None
        }
        assert geoms["base_link"] == GeometryType.BOX
        assert geoms["link_a"] == GeometryType.CYLINDER

    # --- Documented lossy mappings ------------------------------------------

    def test_lossy_effort_and_velocity_are_reset_to_defaults(self) -> None:
        """effort/velocity have no MJCF equivalent and revert to URDF defaults."""
        a, b, _ = self._round_trip()
        ja = a.get_joint("joint_1")
        jb = b.get_joint("joint_1")
        # The originals are the fixture values, distinct from the defaults.
        assert ja.limits.effort == pytest.approx(100.0)
        assert ja.limits.velocity == pytest.approx(3.14)
        # After the round-trip they are the parser defaults (the lossy result).
        assert jb.limits.effort == pytest.approx(1000.0)
        assert jb.limits.velocity == pytest.approx(10.0)

    def test_lossy_joint_friction_is_dropped(self) -> None:
        """Joint friction has no MJCF equivalent and is lost (returns 0.0)."""
        a, b, _ = self._round_trip()
        ja = a.get_joint("joint_1")
        jb = b.get_joint("joint_1")
        assert ja.dynamics.friction == pytest.approx(0.1)
        assert jb.dynamics.friction == pytest.approx(0.0)
        # Damping, by contrast, *is* representable in MJCF and is preserved.
        assert jb.dynamics.damping == pytest.approx(ja.dynamics.damping, abs=1e-6)

    def test_lossy_continuous_joint_maps_to_revolute(self) -> None:
        """continuous -> MJCF hinge -> revolute (the continuous type is lost)."""
        from model_generation.converters.mjcf_converter import MJCFConverter
        from model_generation.converters.urdf_parser import URDFParser
        from model_generation.core.types import JointType

        parser = URDFParser(resolve_meshes=False)
        converter = MJCFConverter()

        model_a = parser.parse_string(MULTI_JOINT_URDF)
        urdf_back = converter.mjcf_to_urdf(converter.urdf_to_mjcf(model_a))
        model_b = parser.parse_string(urdf_back)

        jb = model_b.get_joint("joint_bc")
        assert jb is not None
        # Originally continuous, now revolute after the MJCF hinge round-trip.
        assert model_a.get_joint("joint_bc").joint_type == JointType.CONTINUOUS
        assert jb.joint_type == JointType.REVOLUTE
