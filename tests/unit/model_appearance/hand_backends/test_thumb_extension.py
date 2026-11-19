"""No thumb joint bends back past straight (#15278, epic #15150).

Owner review 2026-11-19: the first and second joints of the left thumb looked
hyperextended. The segment from the thumb's first joint (deep in the hand, near
the wrist) to its second cannot bend back past a straight line, and neither
can the joint after it, on either hand. The floor comes from
``scripts/config/hand_visual_quality.json``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
)
from src.shared.python.model_appearance.hand_backends.skinned_landmarks import (
    hand_landmarks,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
)
from tests.unit.model_appearance.hand_backends.test_skinned_human import (
    _hand,
    _pair,
)

pytestmark = pytest.mark.unit

_Q = load_hand_visual_quality()
#: Each thumb joint, as the segment before it and the segment that turns on it.
_THUMB_JOINTS = (("thumb_meta", "thumb_prox"), ("thumb_prox", "thumb_dist"))
_SLACK_DEG = 1e-6


def test_the_configured_thumb_limits_do_not_reach_past_straight() -> None:
    settings = default_settings()
    floor = float(_Q["min_thumb_joint_flexion_deg"])
    assert settings.thumb_mcp_limits_deg[0] >= floor
    assert settings.thumb_ip_limits_deg[0] >= floor


@pytest.mark.parametrize("side", ["lead", "trail"])
def test_no_thumb_joint_bends_back_past_straight(side: str) -> None:
    """Measured on the posed bones, not read from the joint angles: the bend
    of each segment from the one before it, about the joint's own hinge, with
    flexion positive. Checked on the hand alone and on the pair."""
    floor = float(_Q["min_thumb_joint_flexion_deg"])
    for name, hand in (("lone", _hand(side)), ("pair", _pair()[side])):
        marks = hand_landmarks(hand)
        for before, after in _THUMB_JOINTS:
            inner, outer = marks.joint(before), marks.joint(after)
            assert outer.hinge_axis is not None
            turn = np.cross(inner.bone_axis, outer.bone_axis) @ outer.hinge_axis
            bend = math.degrees(math.atan2(turn, inner.bone_axis @ outer.bone_axis))
            assert bend >= floor - _SLACK_DEG, (
                f"{name} {side} {after} bends back {-bend:.1f} degrees"
            )
