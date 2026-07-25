"""Regression tests for issue #7980.

The former ``_positions_to_angles`` used ``np.linalg.norm(target_pos)`` - the
distance from the *world origin* to the marker - in a law-of-cosines formula,
wrote the result to the second-to-last joint of the chain (so left/right limbs
overwrote each other), and left 12 of 22 joints at zero. A 0.5 m rigid
translation of the capture volume changed the reported joint angles by 78.6
degrees, and 8 of the 10 non-zero joints were pinned at exactly pi by the
``np.clip(cos_angle, -1, 1)`` saturation.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from src.learning.retargeting.retargeter import MotionRetargeter, SkeletonConfig

pytestmark = pytest.mark.unit

MARKER_NAMES = [
    "LSHO",
    "RSHO",
    "LELB",
    "RELB",
    "LWRI",
    "RWRI",
    "LHIP",
    "RHIP",
    "LKNE",
    "RKNE",
    "LANK",
    "RANK",
]

STANDING_POSE = np.array(
    [
        [0.18, 0.0, 1.40],
        [-0.18, 0.0, 1.40],
        [0.20, 0.0, 1.15],
        [-0.20, 0.0, 1.15],
        [0.22, 0.0, 0.92],
        [-0.22, 0.0, 0.92],
        [0.10, 0.0, 0.95],
        [-0.10, 0.0, 0.95],
        [0.11, 0.0, 0.52],
        [-0.11, 0.0, 0.52],
        [0.12, 0.0, 0.10],
        [-0.12, 0.0, 0.10],
    ]
)


def _retargeter() -> MotionRetargeter:
    return MotionRetargeter(
        SkeletonConfig.create_humanoid(), SkeletonConfig.create_humanoid()
    )


class TestTranslationInvariance:
    def test_rigid_translation_does_not_change_joint_angles(self) -> None:
        """Joint angles are a property of the pose, not of where it happens."""
        retargeter = _retargeter()
        frame_a = STANDING_POSE[None]
        frame_b = frame_a + np.array([0.5, 0.0, 0.0])

        angles_a = retargeter.retarget_from_mocap(frame_a, MARKER_NAMES)
        angles_b = retargeter.retarget_from_mocap(frame_b, MARKER_NAMES)

        max_diff = float(np.max(np.abs(angles_a - angles_b)))
        assert max_diff < 1e-6, f"{np.degrees(max_diff)} deg drift under translation"

    @pytest.mark.parametrize("shift", [0.1, -2.0, 10.0])
    def test_invariance_across_translation_magnitudes(self, shift: float) -> None:
        retargeter = _retargeter()
        base = retargeter.retarget_from_mocap(STANDING_POSE[None], MARKER_NAMES)
        moved = retargeter.retarget_from_mocap(
            STANDING_POSE[None] + np.array([shift, shift / 2, 0.0]), MARKER_NAMES
        )
        assert np.max(np.abs(base - moved)) < 1e-6


class TestSolveQuality:
    def test_no_joint_is_pinned_at_pi(self) -> None:
        """arccos saturation used to pin 8 of 10 non-zero joints at exactly pi."""
        angles = _retargeter().retarget_from_mocap(STANDING_POSE[None], MARKER_NAMES)
        assert not np.any(np.isclose(np.abs(angles), np.pi, atol=1e-9))

    def test_left_and_right_limbs_are_not_identical(self) -> None:
        """Both hips used to be written to angles[pelvis], so they collided."""
        target = SkeletonConfig.create_humanoid()
        retargeter = MotionRetargeter(SkeletonConfig.create_humanoid(), target)
        pose = STANDING_POSE.copy()
        pose[8] += np.array([0.0, 0.15, 0.0])  # bend the LEFT knee forward only
        angles = retargeter.retarget_from_mocap(pose[None], MARKER_NAMES)[0]

        left = angles[target.joint_names.index("left_hip")]
        right = angles[target.joint_names.index("right_hip")]
        assert left != right

    def test_unconstrained_joints_are_reported(self) -> None:
        """Joints no marker constrains are named, not silently returned as 0."""
        retargeter = _retargeter()
        retargeter.retarget_from_mocap(STANDING_POSE[None], MARKER_NAMES)
        assert set(retargeter.unconstrained_joints) == {
            "neck",
            "head",
            "left_foot",
            "right_foot",
            "left_hand",
            "right_hand",
        }

    def test_poor_fit_is_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        """A pose the skeleton cannot reach must not be reported silently."""
        retargeter = _retargeter()
        with caplog.at_level(logging.WARNING):
            retargeter.retarget_from_mocap(STANDING_POSE[None], MARKER_NAMES)
        assert any("RMS per marker" in rec.getMessage() for rec in caplog.records)

    def test_solver_reduces_the_residual(self) -> None:
        """The IK must beat the zero-angle initialisation it starts from."""
        retargeter = _retargeter()
        target = retargeter.target
        mapping = retargeter._infer_marker_mapping(MARKER_NAMES)
        joint_positions = {
            joint: STANDING_POSE[MARKER_NAMES.index(marker)]
            for marker, joint in mapping.items()
        }
        indices = [target.get_joint_index(j) for j in joint_positions]
        targets = np.array(list(joint_positions.values()))
        centred = targets - targets.mean(axis=0)

        def residual(angles: np.ndarray) -> float:
            fk = retargeter.forward_kinematics(angles, target)[indices]
            diff = (fk - fk.mean(axis=0)) - centred
            return float(np.vdot(diff, diff))

        zero_residual = residual(np.zeros(target.n_joints))
        solved = retargeter._positions_to_angles(joint_positions)
        assert residual(solved) < zero_residual


class TestForwardKinematicsUsesJointAxes:
    def test_joint_axes_change_the_result(self) -> None:
        """joint_axes was populated but never read (issue #7980)."""
        z_skel = SkeletonConfig.create_humanoid()
        y_skel = SkeletonConfig.create_humanoid()
        y_skel.joint_axes = np.tile(np.array([0.0, 1.0, 0.0]), (y_skel.n_joints, 1))

        r_z = MotionRetargeter(z_skel, z_skel)
        r_y = MotionRetargeter(y_skel, y_skel)
        angles = np.linspace(0.1, 0.5, z_skel.n_joints)

        assert not np.allclose(
            r_z.forward_kinematics(angles, z_skel),
            r_y.forward_kinematics(angles, y_skel),
        )

    def test_zero_angles_reproduce_the_rest_pose(self) -> None:
        """With all angles zero, FK must be the accumulated T-pose offsets."""
        skeleton = SkeletonConfig.create_humanoid()
        retargeter = MotionRetargeter(skeleton, skeleton)
        positions = retargeter.forward_kinematics(np.zeros(skeleton.n_joints), skeleton)
        for idx, parent in enumerate(skeleton.parent_indices):
            expected = (
                skeleton.joint_offsets[idx]
                if parent < 0
                else positions[parent] + skeleton.joint_offsets[idx]
            )
            np.testing.assert_allclose(positions[idx], expected)

    def test_a_joint_rotation_does_not_move_itself(self) -> None:
        """A joint's own angle moves its descendants, never its own origin."""
        skeleton = SkeletonConfig.create_humanoid()
        retargeter = MotionRetargeter(skeleton, skeleton)
        elbow = skeleton.joint_names.index("left_elbow")
        wrist = skeleton.joint_names.index("left_wrist")

        angles = np.zeros(skeleton.n_joints)
        base = retargeter.forward_kinematics(angles, skeleton)
        angles[elbow] = 0.7
        moved = retargeter.forward_kinematics(angles, skeleton)

        np.testing.assert_allclose(moved[elbow], base[elbow])
        assert not np.allclose(moved[wrist], base[wrist])

    def test_zero_axis_is_a_fixed_joint(self) -> None:
        """A zero rotation axis must behave as identity, not produce NaN."""
        skeleton = SkeletonConfig.create_humanoid()
        skeleton.joint_axes = np.zeros((skeleton.n_joints, 3))
        retargeter = MotionRetargeter(skeleton, skeleton)
        positions = retargeter.forward_kinematics(
            np.full(skeleton.n_joints, 0.9), skeleton
        )
        rest = retargeter.forward_kinematics(np.zeros(skeleton.n_joints), skeleton)
        np.testing.assert_allclose(positions, rest)
