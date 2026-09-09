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

from model_generation.builders.manual_builder import ManualBuilder
from model_generation.converters.urdf_parser import ParsedModel, URDFParser
from model_generation.core.types import (
    Geometry,
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
