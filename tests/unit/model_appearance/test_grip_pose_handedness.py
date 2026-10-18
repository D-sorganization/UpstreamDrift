"""Handedness-aware GripPose (HRG-GR0, #13201)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.model_appearance.grip_pose import GripPose, Hand

# Body frame: x anterior, y toward the golfer's anatomical left, z up.  The
# sagittal plane is x-z, so its reflection flips y (the lateral axis).
P = np.diag([1.0, -1.0, 1.0])


def test_default_is_right_with_lead_left() -> None:
    pose = GripPose()
    assert pose.handedness == "right"
    assert pose.side(Hand.LEAD) == "L"
    assert pose.side(Hand.TRAIL) == "R"


def test_left_handed_sides_swap() -> None:
    pose = GripPose(handedness="left")
    assert pose.side(Hand.LEAD) == "R"
    assert pose.side(Hand.TRAIL) == "L"


def test_invalid_handedness_raises() -> None:
    with pytest.raises(ValueError, match="handedness"):
        GripPose(handedness="both")  # type: ignore[arg-type]


@pytest.mark.parametrize("hand", [Hand.LEAD, Hand.TRAIL])
def test_sagittal_mirror_of_right_equals_left(hand: Hand) -> None:
    right = GripPose().landmarks(hand)
    left = GripPose(handedness="left").landmarks(hand)
    assert right.keys() == left.keys()
    for name, point in right.items():
        np.testing.assert_allclose(P @ point, left[name], atol=1e-9, err_msg=name)


def test_lateral_landmarks_differ_between_handedness() -> None:
    right = GripPose().landmarks(Hand.LEAD)["palm_normal"]
    left = GripPose(handedness="left").landmarks(Hand.LEAD)["palm_normal"]
    assert not np.allclose(right, left)
