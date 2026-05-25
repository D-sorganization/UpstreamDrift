"""Adversarial: confirm every CIR invariant rejects bad input.

Parametrised matrix over (model_class, bad_kwargs, expected_substring).
If a row fails to raise, that means the invariant is missing in the
production code and an issue should be filed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.shared.python.motion_pipeline.contracts import (
    CameraIntrinsics,
    JointDef,
    JointLimit,
    Keypoint,
    KeypointFrame,
    KeypointSequence,
    SkeletonRig,
)

# ---------------------------------------------------------------------------
# Invariant violation matrix
# ---------------------------------------------------------------------------

# Each row: (id, model_class, bad_kwargs)
INVARIANT_CASES = [
    # CameraIntrinsics
    (
        "intrinsics-zero-fx",
        CameraIntrinsics,
        {"fx": 0.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    ),
    (
        "intrinsics-negative-fy",
        CameraIntrinsics,
        {"fx": 1.0, "fy": -1.0, "cx": 0.0, "cy": 0.0},
    ),
    # Keypoint
    (
        "keypoint-confidence-negative",
        Keypoint,
        {"x": 0.0, "y": 0.0, "confidence": -0.5},
    ),
    ("keypoint-confidence-too-high", Keypoint, {"x": 0.0, "y": 0.0, "confidence": 5.0}),
    # JointLimit
    ("limit-inverted", JointLimit, {"lower": 1.0, "upper": 0.0}),
    # JointDef
    (
        "joint-empty-name",
        JointDef,
        {
            "name": "",
            "parent": None,
            "tpose_offset": [0.0, 0.0, 0.0],
            "axes": ["Y"],
            "limits": [JointLimit(lower=-1.0, upper=1.0)],
        },
    ),
    (
        "joint-bad-offset-length",
        JointDef,
        {
            "name": "j",
            "parent": None,
            "tpose_offset": [0.0, 0.0],  # not 3D
            "axes": ["Y"],
            "limits": [JointLimit(lower=-1.0, upper=1.0)],
        },
    ),
]


# Cases that surface real bugs in production — see filed issues.
KNOWN_GAPS = {
    "joint-empty-name": "GH #4720 — JointDef accepts empty name",
    "joint-bad-offset-length": "GH #4720 — JointDef accepts non-3D offset",
}


@pytest.mark.parametrize(
    ("case_id", "model", "kwargs"),
    INVARIANT_CASES,
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_invariant_rejects_bad_input(
    case_id: str, model: type, kwargs: dict, request
) -> None:
    """Every (model, bad_kwargs) pair must raise ValidationError."""
    if case_id in KNOWN_GAPS:
        request.node.add_marker(
            pytest.mark.xfail(strict=True, reason=KNOWN_GAPS[case_id])
        )
    with pytest.raises((ValidationError, ValueError)):
        model(**kwargs)


# ---------------------------------------------------------------------------
# Sequence-level invariants
# ---------------------------------------------------------------------------


def test_keypoint_sequence_rejects_non_monotonic_timestamps() -> None:
    """timestamps must be non-decreasing."""
    with pytest.raises(ValidationError):
        KeypointSequence(
            id="bad",
            frames=[
                KeypointFrame(
                    timestamp=1.0,
                    schema_name="MediaPipe_33",
                    keypoints=[Keypoint(x=0.0, y=0.0)],
                ),
                KeypointFrame(
                    timestamp=0.5,  # back in time
                    schema_name="MediaPipe_33",
                    keypoints=[Keypoint(x=0.0, y=0.0)],
                ),
            ],
            fps=30.0,
            schema_name="MediaPipe_33",
        )


def test_skeleton_rejects_unknown_root() -> None:
    """A SkeletonRig whose root_joint is not in the joints dict must raise."""
    with pytest.raises((ValidationError, ValueError)):
        SkeletonRig(
            id="bad",
            joints={
                "root": JointDef(
                    name="root",
                    parent=None,
                    tpose_offset=[0.0, 0.0, 0.0],
                    axes=["Y"],
                    limits=[JointLimit(lower=-1.0, upper=1.0)],
                )
            },
            root_joint="nonexistent",
            up_axis="+Y",
        )
