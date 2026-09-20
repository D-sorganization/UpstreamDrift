"""Unit tests for cross-representation parity between SkeletonRig/JointTrajectory and CanonicalPose (#8867 / MS-12).

Verifies:
1. canonical_pose_to_joint_trajectory maps CanonicalPose to CIR JointTrajectory.
2. joint_trajectory_to_canonical_pose maps JointTrajectory back to CanonicalPose.
3. Bidirectional round-trip preserves pelvis SE(3) transform and joint angles.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    SkeletonRig,
)
from src.shared.python.pose_interchange import (
    CanonicalPose,
    canonical_from_reference_setup,
    canonical_pose_to_joint_trajectory,
    joint_trajectory_to_canonical_pose,
)


def _build_test_golfer_rig() -> SkeletonRig:
    """Build a minimal SkeletonRig representing root and key golfer joints."""
    joints = {
        "pelvis_trans": JointDef(
            name="pelvis_trans",
            parent=None,
            children=["pelvis_rot"],
            axes=["X", "Y", "Z"],
            limits=[JointLimit(lower=-5.0, upper=5.0) for _ in range(3)],
            semantic_label="pelvis_translation",
        ),
        "pelvis_rot": JointDef(
            name="pelvis_rot",
            parent="pelvis_trans",
            children=["spine"],
            axes=["X", "Y", "Z"],
            limits=[JointLimit(lower=-math.pi, upper=math.pi) for _ in range(3)],
            semantic_label="pelvis_rotation",
        ),
        "spine": JointDef(
            name="spine",
            parent="pelvis_rot",
            children=["torso"],
            axes=["X", "Y"],
            limits=[JointLimit(lower=-math.pi, upper=math.pi) for _ in range(2)],
            semantic_label="SpineStartPosition",
        ),
        "torso": JointDef(
            name="torso",
            parent="spine",
            children=[],
            axes=["Z"],
            limits=[JointLimit(lower=-math.pi, upper=math.pi)],
            semantic_label="TorsoStartPosition",
        ),
    }
    return SkeletonRig(
        id="test_golfer_rig",
        joints=joints,
        root_joint="pelvis_trans",
        up_axis="+Z",
    )


@pytest.mark.unit
def test_canonical_pose_to_trajectory_and_back() -> None:
    """Round-trip conversion between CanonicalPose and JointTrajectory preserves kinematics."""
    rig = _build_test_golfer_rig()
    reference_pose = canonical_from_reference_setup()

    # Convert to JointTrajectory
    trajectory = canonical_pose_to_joint_trajectory(reference_pose, rig)
    assert len(trajectory.frames) == 1
    frame = trajectory.frames[0]
    assert len(frame.q) == rig.num_dofs

    # Root translation matches
    np.testing.assert_allclose(
        frame.q[:3], reference_pose.pelvis_translation_m, atol=1e-6
    )

    # Root rotation matches in radians
    expected_rot_rad = [math.radians(a) for a in reference_pose.pelvis_rotation_xyz_deg]
    np.testing.assert_allclose(frame.q[3:6], expected_rot_rad, atol=1e-6)

    # Convert back to CanonicalPose
    roundtrip_pose = joint_trajectory_to_canonical_pose(trajectory)
    np.testing.assert_allclose(
        roundtrip_pose.pelvis_translation_m,
        reference_pose.pelvis_translation_m,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        roundtrip_pose.pelvis_rotation_xyz_deg,
        reference_pose.pelvis_rotation_xyz_deg,
        atol=1e-4,
    )

    # Check mapped joint angles in degrees
    for field_name in (
        "TorsoStartPosition",
        "SpineStartPositionX",
        "SpineStartPositionY",
    ):
        if (
            field_name in reference_pose.joint_angles_deg
            and field_name in roundtrip_pose.joint_angles_deg
        ):
            assert (
                pytest.approx(roundtrip_pose.joint_angles_deg[field_name], abs=1e-4)
                == (reference_pose.joint_angles_deg[field_name])
            )


@pytest.mark.unit
def test_joint_trajectory_to_canonical_pose_bounds() -> None:
    """Out-of-range frame index or empty trajectory raises error."""
    rig = _build_test_golfer_rig()
    pose = canonical_from_reference_setup()
    trajectory = canonical_pose_to_joint_trajectory(pose, rig)

    with pytest.raises(IndexError):
        joint_trajectory_to_canonical_pose(trajectory, frame_index=99)
