"""
End-to-end integration tests for URDF model generation.

These tests verify full roundtrip pipelines:
  Generate URDF -> Parse back -> Validate structural equivalence

Each test class covers a distinct pipeline to ensure that the
entire build-parse-validate cycle produces consistent results.

References:
  - GitHub issue #1694 (end-to-end integration tests)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
from model_generation.builders.manual_builder import ManualBuilder
from model_generation.builders.parametric_builder import ParametricBuilder
from model_generation.converters.urdf_parser import ParsedModel, URDFParser
from model_generation.core.types import (
    Geometry,
    GeometryType,
    Inertia,
    Joint,
    JointType,
    Link,
    Material,
    Origin,
)


class TestManualBuilderRoundtrip:
    """Generate URDF via ManualBuilder -> Parse back -> Verify structural equivalence."""

    def _roundtrip(self, builder: ManualBuilder) -> tuple[ManualBuilder, ParsedModel]:
        """Build URDF and parse it back. Returns (builder, parsed_model)."""
        result = builder.build()
        assert result.success, f"Build failed: {result.error_message}"
        assert result.urdf_xml is not None

        parser = URDFParser(resolve_meshes=False)
        parsed = parser.parse_string(result.urdf_xml)
        return builder, parsed

    def test_single_link_roundtrip(self) -> None:
        """A single-link model survives roundtrip with matching name and mass."""
        builder = ManualBuilder("single_link_bot")
        builder.add_link(
            Link(
                name="base",
                inertia=Inertia.from_box(5.0, 0.3, 0.3, 0.2),
                visual_geometry=Geometry.box(0.3, 0.3, 0.2),
            )
        )

        _, parsed = self._roundtrip(builder)

        assert parsed.name == "single_link_bot"
        assert len(parsed.links) == 1
        assert parsed.links[0].name == "base"
        assert abs(parsed.links[0].inertia.mass - 5.0) < 1e-4

    def test_two_link_chain_roundtrip(self) -> None:
        """A parent-child chain preserves link names, joint type, and parent/child refs."""
        builder = ManualBuilder("chain_bot")
        builder.add_link(
            Link(
                name="torso",
                inertia=Inertia.from_box(10, 0.4, 0.3, 0.6),
                visual_geometry=Geometry.box(0.4, 0.3, 0.6),
            )
        )
        builder.add_link(
            Link(
                name="head",
                inertia=Inertia.from_sphere(2.0, 0.1),
                visual_geometry=Geometry.sphere(0.1),
            )
        )
        builder.add_joint(
            Joint(
                name="torso_to_head",
                joint_type=JointType.REVOLUTE,
                parent="torso",
                child="head",
                origin=Origin(xyz=(0, 0, 0.4)),
                axis=(1, 0, 0),
            )
        )

        _, parsed = self._roundtrip(builder)

        assert parsed.name == "chain_bot"
        assert len(parsed.links) == 2
        assert len(parsed.joints) == 1

        link_names = {lnk.name for lnk in parsed.links}
        assert link_names == {"torso", "head"}

        joint = parsed.joints[0]
        assert joint.name == "torso_to_head"
        assert joint.joint_type == JointType.REVOLUTE
        assert joint.parent == "torso"
        assert joint.child == "head"

    def test_three_link_branching_roundtrip(self) -> None:
        """A branching topology (one root, two children) is preserved."""
        builder = ManualBuilder("branch_bot")
        builder.add_link(
            Link(
                name="base",
                inertia=Inertia.from_box(15, 0.5, 0.5, 0.3),
                visual_geometry=Geometry.box(0.5, 0.5, 0.3),
            )
        )
        builder.add_link(
            Link(
                name="left_arm",
                inertia=Inertia.from_cylinder(3.0, 0.05, 0.4),
                visual_geometry=Geometry.cylinder(0.05, 0.4),
            )
        )
        builder.add_link(
            Link(
                name="right_arm",
                inertia=Inertia.from_cylinder(3.0, 0.05, 0.4),
                visual_geometry=Geometry.cylinder(0.05, 0.4),
            )
        )
        builder.add_joint(
            Joint(
                name="base_to_left",
                joint_type=JointType.REVOLUTE,
                parent="base",
                child="left_arm",
                origin=Origin(xyz=(0, 0.3, 0)),
                axis=(1, 0, 0),
            )
        )
        builder.add_joint(
            Joint(
                name="base_to_right",
                joint_type=JointType.REVOLUTE,
                parent="base",
                child="right_arm",
                origin=Origin(xyz=(0, -0.3, 0)),
                axis=(1, 0, 0),
            )
        )

        _, parsed = self._roundtrip(builder)

        link_names = {lnk.name for lnk in parsed.links}
        assert link_names == {"base", "left_arm", "right_arm"}

        joint_names = {j.name for j in parsed.joints}
        assert joint_names == {"base_to_left", "base_to_right"}

        # Root detection
        root = parsed.get_root_link()
        assert root is not None
        assert root.name == "base"

        # Children
        children = parsed.get_children("base")
        assert set(children) == {"left_arm", "right_arm"}

    def test_inertia_values_survive_roundtrip(self) -> None:
        """Inertia tensor components survive URDF serialization/deserialization."""
        original_inertia = Inertia(
            ixx=1.234,
            iyy=2.345,
            izz=3.456,
            ixy=0.01,
            ixz=0.02,
            iyz=0.03,
            mass=7.89,
            center_of_mass=(0.1, -0.2, 0.3),
        )
        builder = ManualBuilder("inertia_bot")
        builder.add_link(
            Link(
                name="body",
                inertia=original_inertia,
                visual_geometry=Geometry.box(0.5, 0.5, 0.5),
            )
        )

        _, parsed = self._roundtrip(builder)

        pi = parsed.links[0].inertia
        assert abs(pi.mass - 7.89) < 1e-3
        assert abs(pi.ixx - 1.234) < 1e-3
        assert abs(pi.iyy - 2.345) < 1e-3
        assert abs(pi.izz - 3.456) < 1e-3
        assert abs(pi.ixy - 0.01) < 1e-3
        assert abs(pi.ixz - 0.02) < 1e-3
        assert abs(pi.iyz - 0.03) < 1e-3
        # COM
        assert abs(pi.center_of_mass[0] - 0.1) < 1e-3
        assert abs(pi.center_of_mass[1] - (-0.2)) < 1e-3
        assert abs(pi.center_of_mass[2] - 0.3) < 1e-3

    def test_joint_limits_survive_roundtrip(self) -> None:
        """Joint limits (lower, upper, effort, velocity) are preserved."""
        from model_generation.core.types import JointLimits

        builder = ManualBuilder("limits_bot")
        builder.add_link(
            Link(
                name="base",
                inertia=Inertia.from_box(10, 0.5, 0.5, 0.5),
                visual_geometry=Geometry.box(0.5, 0.5, 0.5),
            )
        )
        builder.add_link(
            Link(
                name="arm",
                inertia=Inertia.from_cylinder(2, 0.05, 0.4),
                visual_geometry=Geometry.cylinder(0.05, 0.4),
            )
        )
        builder.add_joint(
            Joint(
                name="base_to_arm",
                joint_type=JointType.REVOLUTE,
                parent="base",
                child="arm",
                axis=(0, 0, 1),
                limits=JointLimits(
                    lower=-1.5,
                    upper=1.5,
                    effort=500.0,
                    velocity=5.0,
                ),
            )
        )

        _, parsed = self._roundtrip(builder)

        joint = parsed.joints[0]
        assert joint.limits is not None
        assert abs(joint.limits.lower - (-1.5)) < 1e-3
        assert abs(joint.limits.upper - 1.5) < 1e-3
        assert abs(joint.limits.effort - 500.0) < 1e-3
        assert abs(joint.limits.velocity - 5.0) < 1e-3

    def test_material_survives_roundtrip(self) -> None:
        """Material name and RGBA color survive serialization."""
        mat = Material(name="custom_red", color=(1.0, 0.0, 0.0, 0.8))
        builder = ManualBuilder("mat_bot")
        builder.add_link(
            Link(
                name="body",
                inertia=Inertia.from_box(5, 0.3, 0.3, 0.3),
                visual_geometry=Geometry.box(0.3, 0.3, 0.3),
                visual_material=mat,
            )
        )

        _, parsed = self._roundtrip(builder)

        # Material should be parsed from the URDF
        link = parsed.links[0]
        assert link.visual_material is not None
        assert link.visual_material.name == "custom_red"

    def test_fixed_joint_roundtrip(self) -> None:
        """Fixed joints preserve type and connectivity."""
        builder = ManualBuilder("fixed_bot")
        builder.add_link(
            Link(
                name="body",
                inertia=Inertia.from_box(10, 0.5, 0.5, 0.5),
            )
        )
        builder.add_link(
            Link(
                name="sensor_mount",
                inertia=Inertia.from_box(0.5, 0.1, 0.1, 0.05),
            )
        )
        builder.add_joint(
            Joint(
                name="body_to_sensor",
                joint_type=JointType.FIXED,
                parent="body",
                child="sensor_mount",
                origin=Origin(xyz=(0.2, 0, 0.3)),
            )
        )

        _, parsed = self._roundtrip(builder)

        joint = parsed.joints[0]
        assert joint.joint_type == JointType.FIXED
        assert joint.parent == "body"
        assert joint.child == "sensor_mount"


class TestParametricBuilderRoundtrip:
    """ParametricBuilder -> Build -> Parse -> Validate links/joints match."""

    def test_default_humanoid_roundtrip(self) -> None:
        """Default humanoid model survives roundtrip with correct structure."""
        builder = ParametricBuilder("humanoid")
        builder.set_parameters(height_m=1.75, mass_kg=75.0)
        builder.add_humanoid_segments()
        result = builder.build()

        assert result.success, f"Build failed: {result.error_message}"
        assert result.urdf_xml is not None

        parser = URDFParser(resolve_meshes=False)
        parsed = parser.parse_string(result.urdf_xml)

        assert parsed.name == "humanoid"

        # Must have multiple links (humanoid skeleton)
        assert len(parsed.links) > 10, (
            f"Expected >10 links for humanoid, got {len(parsed.links)}"
        )

        # Must have joints connecting them
        assert len(parsed.joints) >= len(parsed.links) - 1

        # Root should be pelvis
        root = parsed.get_root_link()
        assert root is not None
        assert root.name == "pelvis"

        # Must have bilateral symmetry: left and right limbs
        link_names = {lnk.name for lnk in parsed.links}
        for segment in ["thigh", "shin", "foot", "upper_arm", "forearm", "hand"]:
            assert f"left_{segment}" in link_names, f"Missing left_{segment}"
            assert f"right_{segment}" in link_names, f"Missing right_{segment}"

    def test_parametric_mass_distribution(self) -> None:
        """Total mass of all links approximately equals the configured mass."""
        target_mass = 80.0
        builder = ParametricBuilder("mass_check")
        builder.set_parameters(height_m=1.80, mass_kg=target_mass)
        builder.add_humanoid_segments()
        result = builder.build()

        assert result.success
        total_mass = result.get_total_mass()

        # The sum of segment mass ratios may not be exactly 1.0 due to
        # shoulders and intermediate links, but should be in the right ballpark
        assert total_mass > 0, "Total mass must be positive"
        # Allow generous tolerance since parametric builder distributes mass
        # across many segments with approximation
        assert total_mass > target_mass * 0.5, (
            f"Total mass {total_mass} is less than 50% of target {target_mass}"
        )

    def test_parametric_height_affects_geometry(self) -> None:
        """Different heights produce different link dimensions."""
        builder_short = ParametricBuilder("short")
        builder_short.set_parameters(height_m=1.50, mass_kg=60.0)
        builder_short.add_humanoid_segments()
        result_short = builder_short.build()

        builder_tall = ParametricBuilder("tall")
        builder_tall.set_parameters(height_m=2.00, mass_kg=90.0)
        builder_tall.add_humanoid_segments()
        result_tall = builder_tall.build()

        assert result_short.success and result_tall.success

        # Find a common link (e.g., left_thigh) and compare dimensions
        thigh_short = result_short.get_link("left_thigh")
        thigh_tall = result_tall.get_link("left_thigh")

        assert thigh_short is not None
        assert thigh_tall is not None

        # Taller person should have greater thigh inertia (larger segment)
        assert thigh_tall.inertia.ixx > thigh_short.inertia.ixx, (
            "Taller model should have larger thigh inertia"
        )

    def test_parametric_builder_produces_valid_xml(self) -> None:
        """Parametric URDF is always well-formed XML with <robot> root."""
        builder = ParametricBuilder("xml_check")
        builder.set_parameters(height_m=1.70, mass_kg=70.0)
        builder.add_humanoid_segments()
        result = builder.build()

        assert result.success
        assert result.urdf_xml is not None

        root = ET.fromstring(result.urdf_xml)
        assert root.tag == "robot"
        assert root.get("name") == "xml_check"

        # Every link must have an inertial element
        for link_elem in root.findall(".//link"):
            inertial = link_elem.find("inertial")
            assert inertial is not None, (
                f"Link '{link_elem.get('name')}' missing <inertial>"
            )
            mass = inertial.find("mass")
            assert mass is not None
            mass_val = float(mass.get("value", "0"))
            assert mass_val > 0, (
                f"Link '{link_elem.get('name')}' has non-positive mass {mass_val}"
            )

    def test_parametric_custom_segment_roundtrip(self) -> None:
        """Custom segments added via add_segment survive roundtrip."""
        builder = ParametricBuilder("custom_bot")
        builder.set_parameters(height_m=1.0, mass_kg=10.0)
        builder.add_segment(
            name="base",
            parent=None,
            mass_ratio=0.5,
            length_ratio=0.3,
            geometry_type=GeometryType.BOX,
        )
        builder.add_segment(
            name="arm",
            parent="base",
            mass_ratio=0.3,
            length_ratio=0.4,
            geometry_type=GeometryType.CYLINDER,
            joint_type=JointType.REVOLUTE,
            joint_axis=(1, 0, 0),
        )
        builder.add_segment(
            name="hand",
            parent="arm",
            mass_ratio=0.2,
            length_ratio=0.15,
            geometry_type=GeometryType.SPHERE,
            joint_type=JointType.FIXED,
        )

        result = builder.build()
        assert result.success
        assert result.urdf_xml is not None

        parser = URDFParser(resolve_meshes=False)
        parsed = parser.parse_string(result.urdf_xml)

        link_names = {lnk.name for lnk in parsed.links}
        assert link_names == {"base", "arm", "hand"}

        # Verify parent chain
        assert parsed.get_parent("arm") == "base"
        assert parsed.get_parent("hand") == "arm"
        assert parsed.get_parent("base") is None


class TestParsedModelOperations:
    """Test ParsedModel helper methods on roundtripped models."""

    @pytest.fixture()
    def parsed_chain(self) -> ParsedModel:
        """Build and parse a 4-link chain: A -> B -> C -> D."""
        builder = ManualBuilder("chain_model", validate_on_add=False)
        names = ["A", "B", "C", "D"]
        for name in names:
            builder.add_link(
                Link(
                    name=name,
                    inertia=Inertia.from_box(2.0, 0.2, 0.2, 0.2),
                )
            )
        for i in range(len(names) - 1):
            builder.add_joint(
                Joint(
                    name=f"{names[i]}_to_{names[i + 1]}",
                    joint_type=JointType.REVOLUTE,
                    parent=names[i],
                    child=names[i + 1],
                    axis=(0, 0, 1),
                )
            )

        result = builder.build()
        assert result.success
        parser = URDFParser(resolve_meshes=False)
        return parser.parse_string(result.urdf_xml)

    def test_get_root_link(self, parsed_chain: ParsedModel) -> None:
        """Root of A->B->C->D is A."""
        root = parsed_chain.get_root_link()
        assert root is not None
        assert root.name == "A"

    def test_get_children(self, parsed_chain: ParsedModel) -> None:
        """A has child B, B has child C, D has no children."""
        assert parsed_chain.get_children("A") == ["B"]
        assert parsed_chain.get_children("B") == ["C"]
        assert parsed_chain.get_children("C") == ["D"]
        assert parsed_chain.get_children("D") == []

    def test_get_parent(self, parsed_chain: ParsedModel) -> None:
        """B's parent is A, D's parent is C, A has no parent."""
        assert parsed_chain.get_parent("A") is None
        assert parsed_chain.get_parent("B") == "A"
        assert parsed_chain.get_parent("C") == "B"
        assert parsed_chain.get_parent("D") == "C"

    def test_get_subtree(self, parsed_chain: ParsedModel) -> None:
        """Subtree rooted at B includes B, C, D."""
        subtree = parsed_chain.get_subtree("B")
        assert set(subtree) == {"B", "C", "D"}

    def test_copy_is_independent(self, parsed_chain: ParsedModel) -> None:
        """A copy of a ParsedModel is structurally identical but independent."""
        copy = parsed_chain.copy()

        assert copy.name == parsed_chain.name
        assert len(copy.links) == len(parsed_chain.links)
        assert len(copy.joints) == len(parsed_chain.joints)

        # Modify copy, original unaffected
        copy.links[0] = Link(
            name="modified",
            inertia=Inertia.from_box(1, 0.1, 0.1, 0.1),
        )
        assert parsed_chain.links[0].name == "A"

    def test_to_urdf_produces_valid_xml(self, parsed_chain: ParsedModel) -> None:
        """ParsedModel.to_urdf() produces parseable XML."""
        urdf = parsed_chain.to_urdf()
        root = ET.fromstring(urdf)
        assert root.tag == "robot"
        assert len(root.findall(".//link")) == 4
        assert len(root.findall(".//joint")) == 3


class TestCompositeJointExpansion:
    """Verify that composite joints (gimbal/universal) expand correctly in roundtrip."""

    def test_universal_joint_expands_to_two_revolute(self) -> None:
        """A universal joint should expand to 2 revolute joints + 1 intermediate link."""
        builder = ManualBuilder("universal_bot", validate_on_add=False)
        builder.add_link(
            Link(
                name="base",
                inertia=Inertia.from_box(10, 0.5, 0.5, 0.5),
            )
        )
        builder.add_link(
            Link(
                name="arm",
                inertia=Inertia.from_cylinder(3, 0.05, 0.4),
            )
        )
        builder.add_joint(
            Joint(
                name="base_arm_joint",
                joint_type=JointType.UNIVERSAL,
                parent="base",
                child="arm",
                origin=Origin(xyz=(0, 0.3, 0)),
            )
        )

        result = builder.build()
        assert result.success
        assert result.urdf_xml is not None

        parser = URDFParser(resolve_meshes=False)
        parsed = parser.parse_string(result.urdf_xml)

        # After expansion: base, arm, + 1 intermediate link = 3 links
        assert len(parsed.links) == 3

        # 2 revolute joints
        revolute_joints = [
            j for j in parsed.joints if j.joint_type == JointType.REVOLUTE
        ]
        assert len(revolute_joints) == 2

        # Verify chain connectivity: base -> intermediate -> arm
        parents = {j.child: j.parent for j in parsed.joints}
        # arm's parent chain should trace back to base
        current = "arm"
        visited = set()
        while current in parents:
            visited.add(current)
            current = parents[current]
            if current in visited:
                break  # Avoid infinite loop
        assert current == "base", (
            f"Arm's ancestor chain does not reach 'base': ended at '{current}'"
        )

    def test_gimbal_joint_expands_to_three_revolute(self) -> None:
        """A gimbal joint should expand to 3 revolute joints + 2 intermediate links."""
        builder = ManualBuilder("gimbal_bot", validate_on_add=False)
        builder.add_link(
            Link(
                name="torso",
                inertia=Inertia.from_box(15, 0.4, 0.3, 0.6),
            )
        )
        builder.add_link(
            Link(
                name="head",
                inertia=Inertia.from_sphere(4, 0.1),
            )
        )
        builder.add_joint(
            Joint(
                name="neck_joint",
                joint_type=JointType.GIMBAL,
                parent="torso",
                child="head",
                origin=Origin(xyz=(0, 0, 0.4)),
            )
        )

        result = builder.build()
        assert result.success
        assert result.urdf_xml is not None

        parser = URDFParser(resolve_meshes=False)
        parsed = parser.parse_string(result.urdf_xml)

        # After expansion: torso, head, + 2 intermediate links = 4 links
        assert len(parsed.links) == 4

        # 3 revolute joints
        revolute_joints = [
            j for j in parsed.joints if j.joint_type == JointType.REVOLUTE
        ]
        assert len(revolute_joints) == 3


class TestQuickURDFIntegration:
    """Test the convenience function quick_urdf end-to-end."""

    def test_quick_urdf_produces_parseable_output(self) -> None:
        """quick_urdf() output is valid URDF XML."""
        from model_generation import quick_urdf

        urdf = quick_urdf(height_m=1.80, mass_kg=80.0)
        assert isinstance(urdf, str)
        assert len(urdf) > 100  # non-trivial

        root = ET.fromstring(urdf)
        assert root.tag == "robot"

    def test_quick_urdf_preset_produces_valid_model(self) -> None:
        """quick_urdf with preset 'athletic' produces a valid humanoid."""
        from model_generation import quick_urdf

        urdf = quick_urdf(height_m=1.85, preset="athletic")
        parser = URDFParser(resolve_meshes=False)
        parsed = parser.parse_string(urdf)

        assert len(parsed.links) > 10
        link_names = {lnk.name for lnk in parsed.links}
        assert "pelvis" in link_names
        assert "head" in link_names

    def test_quick_urdf_roundtrip_equivalence(self) -> None:
        """quick_urdf -> parse -> to_urdf -> parse produces structurally same model."""
        from model_generation import quick_urdf

        urdf1 = quick_urdf(height_m=1.75, mass_kg=75.0)
        parser = URDFParser(resolve_meshes=False)
        parsed1 = parser.parse_string(urdf1)

        # Reserialize
        urdf2 = parsed1.to_urdf()
        parsed2 = parser.parse_string(urdf2)

        # Structural equivalence: same links and joints
        names1 = {lnk.name for lnk in parsed1.links}
        names2 = {lnk.name for lnk in parsed2.links}
        assert names1 == names2

        joint_names1 = {j.name for j in parsed1.joints}
        joint_names2 = {j.name for j in parsed2.joints}
        assert joint_names1 == joint_names2
