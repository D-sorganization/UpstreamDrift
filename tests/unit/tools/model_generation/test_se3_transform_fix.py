"""
Tests for SE(3) transform composition in FrankensteinEditor.delete_link().

When reparenting children after deleting a link, the child joint's origin
must be properly composed using the parent joint's SE(3) transform (both
position and rotation), not just simple position addition.

Bug: The original code simply added positions:
    child_joint.origin.xyz = parent_xyz + child_xyz
    (and kept child_joint.origin.rpy unchanged)

Fix: Compose SE(3) transforms properly:
    new_position = parent_pos + R(parent_rpy) @ child_pos
    new_rotation = compose(parent_rpy, child_rpy)
"""

from __future__ import annotations

import math

import numpy as np
from model_generation.core.types import Joint
from model_generation.editor import FrankensteinEditor

# ---------------------------------------------------------------------------
# Helper: build a 3-link chain  A --[j_ab]--> B --[j_bc]--> C
# ---------------------------------------------------------------------------

THREE_LINK_CHAIN_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="three_link_chain">
    <link name="A">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <link name="B">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <link name="C">
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/>
        </inertial>
    </link>
    <joint name="j_ab" type="fixed">
        <parent link="A"/>
        <child link="B"/>
        <origin xyz="{ab_x} {ab_y} {ab_z}" rpy="{ab_r} {ab_p} {ab_yaw}"/>
    </joint>
    <joint name="j_bc" type="fixed">
        <parent link="B"/>
        <child link="C"/>
        <origin xyz="{bc_x} {bc_y} {bc_z}" rpy="{bc_r} {bc_p} {bc_yaw}"/>
    </joint>
</robot>
"""


def _build_chain_urdf(
    ab_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ab_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bc_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    bc_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    return THREE_LINK_CHAIN_TEMPLATE.format(
        ab_x=ab_xyz[0],
        ab_y=ab_xyz[1],
        ab_z=ab_xyz[2],
        ab_r=ab_rpy[0],
        ab_p=ab_rpy[1],
        ab_yaw=ab_rpy[2],
        bc_x=bc_xyz[0],
        bc_y=bc_xyz[1],
        bc_z=bc_xyz[2],
        bc_r=bc_rpy[0],
        bc_p=bc_rpy[1],
        bc_yaw=bc_rpy[2],
    )


def _delete_B_and_get_ac_joint(
    ab_xyz: tuple[float, float, float],
    ab_rpy: tuple[float, float, float],
    bc_xyz: tuple[float, float, float],
    bc_rpy: tuple[float, float, float],
) -> Joint:
    """Build chain, delete B with reparent, return the new A->C joint."""
    urdf = _build_chain_urdf(ab_xyz, ab_rpy, bc_xyz, bc_rpy)
    editor = FrankensteinEditor()
    editor.load_model("m", urdf)
    # load_model sets read_only by default; duplicate to get writable copy
    editor.duplicate_model("m", "editable")

    result = editor.delete_link("editable", "B", reparent_children=True)
    assert result is True, "delete_link should succeed"

    model = editor.get_model("editable")
    assert model is not None

    # After deleting B, link C should be connected to A via one joint
    link_names = [lnk.name for lnk in model.links]
    assert "B" not in link_names, "B should have been removed"
    assert "A" in link_names
    assert "C" in link_names

    # There should be exactly one joint left: A -> C
    assert len(model.joints) == 1, f"Expected 1 joint, got {len(model.joints)}"
    ac_joint = model.joints[0]
    assert ac_joint.parent == "A"
    assert ac_joint.child == "C"
    return ac_joint


# ---------------------------------------------------------------------------
# Reference implementation for expected values
# ---------------------------------------------------------------------------


def _expected_se3_compose(
    p_xyz: tuple[float, float, float],
    p_rpy: tuple[float, float, float],
    c_xyz: tuple[float, float, float],
    c_rpy: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute expected SE(3) composition using scipy (ground truth)."""
    from scipy.spatial.transform import Rotation

    p_rot = Rotation.from_euler("xyz", p_rpy)
    c_rot = Rotation.from_euler("xyz", c_rpy)

    new_pos = np.array(p_xyz) + p_rot.apply(np.array(c_xyz))
    new_rpy = (p_rot * c_rot).as_euler("xyz")

    return tuple(new_pos.tolist()), tuple(new_rpy.tolist())


# ===========================================================================
# Test Cases
# ===========================================================================


class TestSE3TransformComposition:
    """Tests for proper SE(3) transform composition during link deletion."""

    def test_identity_rpy_matches_simple_addition(self) -> None:
        """When parent RPY is zero, result should match simple position addition.

        This is the baseline case -- the old (buggy) code happens to be
        correct when there is no rotation.
        """
        ab_xyz = (1.0, 2.0, 3.0)
        ab_rpy = (0.0, 0.0, 0.0)
        bc_xyz = (0.5, 0.5, 0.5)
        bc_rpy = (0.0, 0.0, 0.0)

        ac_joint = _delete_B_and_get_ac_joint(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        # With zero RPY, simple addition is correct
        expected_xyz = (1.5, 2.5, 3.5)
        expected_rpy = (0.0, 0.0, 0.0)

        np.testing.assert_allclose(ac_joint.origin.xyz, expected_xyz, atol=1e-10)
        np.testing.assert_allclose(ac_joint.origin.rpy, expected_rpy, atol=1e-10)

    def test_90deg_yaw_rotation(self) -> None:
        """90-degree yaw rotation should swap x/y for child position.

        Parent joint: position (1, 0, 0), yaw = pi/2
        Child joint:  position (1, 0, 0), no rotation

        After composition:
        - Parent rotates child's x-axis to point along parent's y-axis
        - new_pos = (1, 0, 0) + R_z(pi/2) @ (1, 0, 0) = (1, 0, 0) + (0, 1, 0) = (1, 1, 0)
        - new_rpy = (0, 0, pi/2)

        The old (buggy) code would produce (2, 0, 0) and (0, 0, 0) -- WRONG.
        """
        ab_xyz = (1.0, 0.0, 0.0)
        ab_rpy = (0.0, 0.0, math.pi / 2)
        bc_xyz = (1.0, 0.0, 0.0)
        bc_rpy = (0.0, 0.0, 0.0)

        ac_joint = _delete_B_and_get_ac_joint(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        expected_pos, expected_rpy = _expected_se3_compose(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        np.testing.assert_allclose(ac_joint.origin.xyz, expected_pos, atol=1e-10)
        np.testing.assert_allclose(ac_joint.origin.rpy, expected_rpy, atol=1e-10)

        # Verify specific values for clarity
        np.testing.assert_allclose(
            ac_joint.origin.xyz,
            (1.0, 1.0, 0.0),
            atol=1e-10,
            err_msg="90-deg yaw should rotate child x-offset into y",
        )

    def test_90deg_pitch_rotation(self) -> None:
        """90-degree pitch rotation should swap x/z for child position.

        Parent joint: position (0, 0, 1), pitch = pi/2
        Child joint:  position (1, 0, 0), no rotation

        After composition:
        - Pitch (R_y) rotates child's x-axis to point along -z
        - new_pos = (0, 0, 1) + R_y(pi/2) @ (1, 0, 0) = (0, 0, 1) + (0, 0, -1) = (0, 0, 0)
        """
        ab_xyz = (0.0, 0.0, 1.0)
        ab_rpy = (0.0, math.pi / 2, 0.0)
        bc_xyz = (1.0, 0.0, 0.0)
        bc_rpy = (0.0, 0.0, 0.0)

        ac_joint = _delete_B_and_get_ac_joint(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        expected_pos, expected_rpy = _expected_se3_compose(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        np.testing.assert_allclose(ac_joint.origin.xyz, expected_pos, atol=1e-10)
        np.testing.assert_allclose(ac_joint.origin.rpy, expected_rpy, atol=1e-10)

    def test_combined_translation_and_rotation(self) -> None:
        """Test general case with both translation and rotation.

        Parent: xyz=(1, 2, 3), rpy=(0.3, 0.5, 0.7)
        Child:  xyz=(0.5, 0.5, 0.5), rpy=(0.1, 0.2, 0.3)
        """
        ab_xyz = (1.0, 2.0, 3.0)
        ab_rpy = (0.3, 0.5, 0.7)
        bc_xyz = (0.5, 0.5, 0.5)
        bc_rpy = (0.1, 0.2, 0.3)

        ac_joint = _delete_B_and_get_ac_joint(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        expected_pos, expected_rpy = _expected_se3_compose(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        np.testing.assert_allclose(ac_joint.origin.xyz, expected_pos, atol=1e-10)
        np.testing.assert_allclose(ac_joint.origin.rpy, expected_rpy, atol=1e-10)

    def test_rotation_composition_without_translation(self) -> None:
        """Test that rotations compose even when child position is at origin.

        Parent: xyz=(0, 0, 0), rpy=(0, 0, pi/4)
        Child:  xyz=(0, 0, 0), rpy=(0, 0, pi/4)
        Expected: xyz=(0, 0, 0), rpy=(0, 0, pi/2)
        """
        ab_xyz = (0.0, 0.0, 0.0)
        ab_rpy = (0.0, 0.0, math.pi / 4)
        bc_xyz = (0.0, 0.0, 0.0)
        bc_rpy = (0.0, 0.0, math.pi / 4)

        ac_joint = _delete_B_and_get_ac_joint(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        expected_pos, expected_rpy = _expected_se3_compose(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        np.testing.assert_allclose(ac_joint.origin.xyz, expected_pos, atol=1e-10)
        np.testing.assert_allclose(ac_joint.origin.rpy, expected_rpy, atol=1e-10)

        # Verify the composed yaw is pi/2
        np.testing.assert_allclose(
            ac_joint.origin.rpy[2],
            math.pi / 2,
            atol=1e-10,
            err_msg="Two pi/4 yaw rotations should compose to pi/2",
        )

    def test_90deg_roll_rotation(self) -> None:
        """90-degree roll rotation should swap y/z for child position.

        Parent joint: position (0, 0, 0), roll = pi/2
        Child joint:  position (0, 1, 0), no rotation

        After composition:
        - Roll (R_x) rotates child's y-axis to point along z
        - new_pos = (0, 0, 0) + R_x(pi/2) @ (0, 1, 0) = (0, 0, 1)
        """
        ab_xyz = (0.0, 0.0, 0.0)
        ab_rpy = (math.pi / 2, 0.0, 0.0)
        bc_xyz = (0.0, 1.0, 0.0)
        bc_rpy = (0.0, 0.0, 0.0)

        ac_joint = _delete_B_and_get_ac_joint(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        expected_pos, expected_rpy = _expected_se3_compose(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        np.testing.assert_allclose(ac_joint.origin.xyz, expected_pos, atol=1e-10)
        np.testing.assert_allclose(ac_joint.origin.rpy, expected_rpy, atol=1e-10)

        # Verify specific values
        np.testing.assert_allclose(
            ac_joint.origin.xyz,
            (0.0, 0.0, 1.0),
            atol=1e-10,
            err_msg="90-deg roll should rotate child y-offset into z",
        )

    def test_180deg_yaw_reverses_x(self) -> None:
        """180-degree yaw should negate both x and y of child position.

        Parent: xyz=(0, 0, 0), yaw = pi
        Child:  xyz=(1, 0, 0)
        Expected: xyz=(-1, 0, 0)
        """
        ab_xyz = (0.0, 0.0, 0.0)
        ab_rpy = (0.0, 0.0, math.pi)
        bc_xyz = (1.0, 0.0, 0.0)
        bc_rpy = (0.0, 0.0, 0.0)

        ac_joint = _delete_B_and_get_ac_joint(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        expected_pos, expected_rpy = _expected_se3_compose(ab_xyz, ab_rpy, bc_xyz, bc_rpy)

        np.testing.assert_allclose(ac_joint.origin.xyz, expected_pos, atol=1e-10)

        # The child at (1,0,0) rotated by pi around z becomes (-1, 0, 0)
        np.testing.assert_allclose(
            ac_joint.origin.xyz,
            (-1.0, 0.0, 0.0),
            atol=1e-10,
            err_msg="180-deg yaw should negate x for child",
        )
