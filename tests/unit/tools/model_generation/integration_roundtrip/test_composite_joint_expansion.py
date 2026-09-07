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
from model_generation.converters.urdf_parser import URDFParser
from model_generation.core.types import (
    Inertia,
    Joint,
    JointType,
    Link,
    Origin,
)


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
