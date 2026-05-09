"""Adversarial: boundary conditions across the pipeline."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointLimit,
    JointStateFrame,
    JointTrajectory,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    MarkerFrame,
    MarkerTrajectory,
    Marker,
    MotionMatchingRequest,
    SkeletonRig,
)


# ---------------------------------------------------------------------------
# Single-frame trajectories
# ---------------------------------------------------------------------------


def test_single_frame_keypoint_sequence_valid() -> None:
    """A 1-frame KeypointSequence must be a valid CIR object."""
    seq = KeypointSequence(
        id="single",
        frames=[
            KeypointFrame(
                timestamp=0.0,
                schema_name="MediaPipe_33",
                keypoints=[Keypoint(x=0.0, y=0.0, confidence=1.0)],
            )
        ],
        fps=30.0,
        schema_name="MediaPipe_33",
    )
    assert len(seq.frames) == 1


def test_single_joint_skeleton_valid() -> None:
    """A skeleton with exactly one (root) joint must validate."""
    rig = SkeletonRig(
        id="solo",
        joints={
            "root": JointDef(
                name="root",
                parent=None,
                offset=[0.0, 0.0, 0.0],
                axis=[0.0, 1.0, 0.0],
                limit=JointLimit(lower=-1.0, upper=1.0),
            )
        },
        root_joint="root",
        up_axis="+Y",
    )
    assert len(rig.joints) == 1


# ---------------------------------------------------------------------------
# Empty / pathological inputs
# ---------------------------------------------------------------------------


def test_empty_keypoint_sequence_rejected() -> None:
    """Zero-frame KeypointSequence must raise on construction (frames must
    be non-empty per CIR contract)."""
    with pytest.raises(ValidationError):
        KeypointSequence(
            id="empty",
            frames=[],
            fps=30.0,
            schema_name="MediaPipe_33",
        )


def test_keypoint_with_negative_confidence_rejected() -> None:
    """Confidence < 0 must be rejected by the Keypoint validator."""
    with pytest.raises(ValidationError):
        Keypoint(x=0.0, y=0.0, confidence=-0.1)


def test_keypoint_with_confidence_above_one_rejected() -> None:
    """Confidence > 1 must be rejected."""
    with pytest.raises(ValidationError):
        Keypoint(x=0.0, y=0.0, confidence=1.1)


def test_zero_confidence_keypoints_accepted() -> None:
    """All-zero confidence is a valid (fully unobservable) frame."""
    f = KeypointFrame(
        timestamp=0.0,
        schema_name="MediaPipe_33",
        keypoints=[Keypoint(x=0.0, y=0.0, confidence=0.0) for _ in range(3)],
    )
    assert all(k.confidence == 0.0 for k in f.keypoints)


# ---------------------------------------------------------------------------
# Joint limits
# ---------------------------------------------------------------------------


def test_joint_limit_lower_above_upper_rejected() -> None:
    """JointLimit(lower=1, upper=-1) must be rejected."""
    with pytest.raises(ValidationError):
        JointLimit(lower=1.0, upper=-1.0)


def test_joint_limit_equal_bounds_locked_joint() -> None:
    """JointLimit(lower=upper=0) is a locked joint and must be permitted."""
    lim = JointLimit(lower=0.0, upper=0.0)
    assert lim.lower == lim.upper == 0.0


# ---------------------------------------------------------------------------
# MotionMatchingRequest
# ---------------------------------------------------------------------------


def test_matching_request_requires_at_least_one_target() -> None:
    """A request with no trajectory, markers, or keypoints must be rejected."""
    rig = SkeletonRig(
        id="rig",
        joints={
            "root": JointDef(
                name="root",
                parent=None,
                offset=[0.0, 0.0, 0.0],
                axis=[0.0, 1.0, 0.0],
                limit=JointLimit(lower=-1.0, upper=1.0),
            )
        },
        root_joint="root",
        up_axis="+Y",
    )
    with pytest.raises(ValidationError):
        MotionMatchingRequest(
            id="empty",
            target_trajectory=None,
            target_markers=None,
            target_keypoints=None,
            skeleton=rig,
        )
