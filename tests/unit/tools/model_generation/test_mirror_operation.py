"""
Comprehensive unit tests for ManualBuilder.mirror().

Tests cover:
- Mirror about X, Y, Z axes
- Position negation on the correct axis
- Inertia product terms negation per axis
- Joint axis mirroring
- Handedness toggling
- Roundtrip: mirror(Y) then mirror(Y) returns to original
"""

from __future__ import annotations

import pytest
from model_generation.builders.manual_builder import Handedness, ManualBuilder
from model_generation.core.types import (
    Inertia,
    Joint,
    JointType,
    Link,
    Origin,
)


def _make_builder_with_two_links() -> ManualBuilder:
    """Create a builder with base + arm and a revolute joint."""
    builder = ManualBuilder("test_robot", validate_on_add=False)
    builder.add_link(
        Link(
            name="base",
            inertia=Inertia(
                ixx=1.0,
                iyy=2.0,
                izz=3.0,
                ixy=0.1,
                ixz=0.2,
                iyz=0.3,
                mass=5.0,
                center_of_mass=(0.1, 0.2, 0.3),
            ),
            visual_origin=Origin(xyz=(1.0, 2.0, 3.0), rpy=(0.0, 0.0, 0.0)),
            collision_origin=Origin(xyz=(0.5, 1.0, 1.5), rpy=(0.0, 0.0, 0.0)),
        )
    )
    builder.add_link(
        Link(
            name="arm",
            inertia=Inertia(
                ixx=0.5,
                iyy=0.6,
                izz=0.7,
                ixy=0.01,
                ixz=0.02,
                iyz=0.03,
                mass=2.0,
                center_of_mass=(0.0, 0.1, -0.2),
            ),
            visual_origin=Origin(xyz=(0.0, 0.5, 0.0), rpy=(0.0, 0.0, 0.0)),
            collision_origin=Origin(xyz=(0.0, 0.5, 0.0), rpy=(0.0, 0.0, 0.0)),
        )
    )
    builder.add_joint(
        Joint(
            name="base_to_arm",
            joint_type=JointType.REVOLUTE,
            parent="base",
            child="arm",
            origin=Origin(xyz=(0.0, 0.3, 0.0), rpy=(0.0, 0.0, 0.0)),
            axis=(1.0, 0.0, 0.0),
        )
    )
    return builder


def _snapshot(builder: ManualBuilder) -> dict:
    """Capture link/joint data for comparison."""
    return {
        "links": [link.to_dict() for link in builder.links],
        "joints": [joint.to_dict() for joint in builder.joints],
        "handedness": builder.handedness,
    }


# ── Mirror about Y axis (default) ───────────────────────────────────────────


class TestMirrorY:
    """Test mirror about Y axis."""

    def test_visual_origin_y_negated(self) -> None:
        builder = _make_builder_with_two_links()
        builder.mirror("y")
        base = builder.links[0]
        assert base.visual_origin.xyz[1] == pytest.approx(-2.0)
        # X and Z should be unchanged
        assert base.visual_origin.xyz[0] == pytest.approx(1.0)
        assert base.visual_origin.xyz[2] == pytest.approx(3.0)

    def test_collision_origin_y_negated(self) -> None:
        builder = _make_builder_with_two_links()
        builder.mirror("y")
        base = builder.links[0]
        assert base.collision_origin.xyz[1] == pytest.approx(-1.0)

    def test_com_y_negated(self) -> None:
        builder = _make_builder_with_two_links()
        builder.mirror("y")
        base = builder.links[0]
        assert base.inertia.center_of_mass[1] == pytest.approx(-0.2)
        assert base.inertia.center_of_mass[0] == pytest.approx(0.1)

    def test_inertia_products_y_mirror(self) -> None:
        """For Y axis (idx=1): ixz stays, ixy and iyz are negated."""
        builder = _make_builder_with_two_links()
        builder.mirror("y")
        base = builder.links[0]
        # axis_idx=1: ixy negated (idx!=2), ixz NOT negated (idx==1), iyz negated (idx!=0)
        assert base.inertia.ixy == pytest.approx(-0.1)
        assert base.inertia.ixz == pytest.approx(0.2)  # unchanged
        assert base.inertia.iyz == pytest.approx(-0.3)

    def test_joint_origin_y_negated(self) -> None:
        builder = _make_builder_with_two_links()
        builder.mirror("y")
        joint = builder.joints[0]
        assert joint.origin.xyz[1] == pytest.approx(-0.3)

    def test_joint_axis_y_negated(self) -> None:
        builder = _make_builder_with_two_links()
        builder.mirror("y")
        joint = builder.joints[0]
        # axis was (1,0,0), mirror Y negates index 1 -> (1,0,0)
        assert joint.axis == (1.0, 0.0, 0.0)  # unchanged because idx 1 was 0

    def test_handedness_toggles(self) -> None:
        builder = _make_builder_with_two_links()
        assert builder.handedness == Handedness.RIGHT
        builder.mirror("y")
        assert builder.handedness == Handedness.LEFT

    def test_diagonal_inertia_preserved(self) -> None:
        builder = _make_builder_with_two_links()
        original_ixx = builder.links[0].inertia.ixx
        builder.mirror("y")
        assert builder.links[0].inertia.ixx == pytest.approx(original_ixx)

    def test_mass_preserved(self) -> None:
        builder = _make_builder_with_two_links()
        builder.mirror("y")
        assert builder.links[0].inertia.mass == 5.0
        assert builder.links[1].inertia.mass == 2.0


# ── Mirror about X axis ─────────────────────────────────────────────────────


class TestMirrorX:
    """Test mirror about X axis."""

    def test_visual_origin_x_negated(self) -> None:
        builder = _make_builder_with_two_links()
        builder.mirror("x")
        base = builder.links[0]
        assert base.visual_origin.xyz[0] == pytest.approx(-1.0)
        assert base.visual_origin.xyz[1] == pytest.approx(2.0)

    def test_inertia_products_x_mirror(self) -> None:
        """For X axis (idx=0): ixy negated (idx!=2), ixz negated (idx!=1), iyz stays (idx==0)."""
        builder = _make_builder_with_two_links()
        builder.mirror("x")
        base = builder.links[0]
        assert base.inertia.ixy == pytest.approx(-0.1)
        assert base.inertia.ixz == pytest.approx(-0.2)
        assert base.inertia.iyz == pytest.approx(0.3)  # unchanged

    def test_joint_axis_x_component_negated(self) -> None:
        builder = _make_builder_with_two_links()
        builder.mirror("x")
        joint = builder.joints[0]
        # axis was (1,0,0) => (-1, 0, 0)
        assert joint.axis == (-1.0, 0.0, 0.0)


# ── Mirror about Z axis ─────────────────────────────────────────────────────


class TestMirrorZ:
    """Test mirror about Z axis."""

    def test_visual_origin_z_negated(self) -> None:
        builder = _make_builder_with_two_links()
        builder.mirror("z")
        base = builder.links[0]
        assert base.visual_origin.xyz[2] == pytest.approx(-3.0)
        assert base.visual_origin.xyz[0] == pytest.approx(1.0)

    def test_inertia_products_z_mirror(self) -> None:
        """For Z axis (idx=2): ixy stays (idx==2), ixz negated (idx!=1), iyz negated (idx!=0)."""
        builder = _make_builder_with_two_links()
        builder.mirror("z")
        base = builder.links[0]
        assert base.inertia.ixy == pytest.approx(0.1)  # unchanged
        assert base.inertia.ixz == pytest.approx(-0.2)
        assert base.inertia.iyz == pytest.approx(-0.3)


# ── Roundtrip ────────────────────────────────────────────────────────────────


class TestMirrorRoundtrip:
    """Test that mirror(axis) then mirror(axis) returns to original."""

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_double_mirror_restores_positions(self, axis: str) -> None:
        builder = _make_builder_with_two_links()
        before = _snapshot(builder)
        builder.mirror(axis)
        builder.mirror(axis)
        after = _snapshot(builder)

        for i, link_before in enumerate(before["links"]):
            link_after = after["links"][i]
            # Visual origin should be restored
            for j in range(3):
                assert link_after["visual_origin"]["xyz"][j] == pytest.approx(
                    link_before["visual_origin"]["xyz"][j]
                )
            # Inertia products restored
            for key in ("ixy", "ixz", "iyz"):
                assert link_after["inertia"][key] == pytest.approx(link_before["inertia"][key])
            # COM restored
            for j in range(3):
                assert link_after["inertia"]["center_of_mass"][j] == pytest.approx(
                    link_before["inertia"]["center_of_mass"][j]
                )

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_double_mirror_restores_handedness(self, axis: str) -> None:
        builder = _make_builder_with_two_links()
        assert builder.handedness == Handedness.RIGHT
        builder.mirror(axis)
        builder.mirror(axis)
        assert builder.handedness == Handedness.RIGHT

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_double_mirror_restores_joint_axis(self, axis: str) -> None:
        builder = _make_builder_with_two_links()
        original_axis = builder.joints[0].axis
        builder.mirror(axis)
        builder.mirror(axis)
        for i in range(3):
            assert builder.joints[0].axis[i] == pytest.approx(original_axis[i])


# ── get_mirrored (copy-based) ────────────────────────────────────────────────


class TestGetMirrored:
    """Test get_mirrored returns a new independent builder."""

    def test_original_unchanged(self) -> None:
        builder = _make_builder_with_two_links()
        original_y = builder.links[0].visual_origin.xyz[1]
        mirrored = builder.get_mirrored("y")
        # Original should be unchanged
        assert builder.links[0].visual_origin.xyz[1] == pytest.approx(original_y)
        # Mirrored should differ
        assert mirrored.links[0].visual_origin.xyz[1] == pytest.approx(-original_y)

    def test_mirrored_handedness(self) -> None:
        builder = _make_builder_with_two_links()
        mirrored = builder.get_mirrored("y")
        assert builder.handedness == Handedness.RIGHT
        assert mirrored.handedness == Handedness.LEFT


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestMirrorEdgeCases:
    """Edge cases for mirror operation."""

    def test_mirror_empty_builder(self) -> None:
        builder = ManualBuilder("empty", validate_on_add=False)
        builder.mirror("y")  # should not raise
        assert builder.handedness == Handedness.LEFT

    def test_mirror_single_link_no_joint(self) -> None:
        builder = ManualBuilder("single", validate_on_add=False)
        builder.add_link(
            Link(
                name="solo",
                inertia=Inertia(ixx=1.0, iyy=1.0, izz=1.0, mass=1.0),
                visual_origin=Origin(xyz=(0.0, 1.0, 0.0)),
            )
        )
        builder.mirror("y")
        assert builder.links[0].visual_origin.xyz[1] == pytest.approx(-1.0)

    def test_rpy_unchanged_by_mirror(self) -> None:
        builder = _make_builder_with_two_links()
        original_rpy = builder.links[0].visual_origin.rpy
        builder.mirror("y")
        assert builder.links[0].visual_origin.rpy == original_rpy
