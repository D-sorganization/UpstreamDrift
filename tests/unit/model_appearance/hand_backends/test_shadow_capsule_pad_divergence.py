"""Shadow vs capsule pad retargeting across shaft roll (#15122)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.shared.python.grip_spec import GripSpec, HandGripSpec
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.shadow import (
    finger_pad_contact_gaps,
    solve_shadow_hand_qpos,
    visual_settings,
)
from src.shared.python.model_appearance.hand_backends.shadow_capsule_compare import (
    pad_divergence_settings,
    solve_and_compare,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]

mujoco = pytest.importorskip("mujoco")

_ROLLS_DEG = (0.0, 20.0, -20.0)
_GRIP_R_M = 0.0118


def _spec(roll_deg: float) -> GripSpec:
    if roll_deg == 0.0:
        return GripSpec.legacy()
    return replace(
        GripSpec.legacy(),
        lead=HandGripSpec(shaft_roll_deg=roll_deg),
        trail=HandGripSpec(shaft_roll_deg=-roll_deg),
    )


@pytest.mark.parametrize("roll_deg", _ROLLS_DEG)
@pytest.mark.parametrize("hand", (Hand.LEAD, Hand.TRAIL))
@pytest.mark.parametrize(
    "finger", ("index", "middle", "ring", "little"),
)
def test_digit_pads_contact_grip_surface(roll_deg: float, hand: Hand, finger: str) -> None:
    spec = _spec(roll_deg)
    hand_spec = spec.lead if hand == Hand.LEAD else spec.trail
    solved = solve_shadow_hand_qpos(
        hand, hand_spec, grip_spec=spec, grip_radius_m=_GRIP_R_M
    )
    gaps = finger_pad_contact_gaps(hand, solved, grip_radius_m=_GRIP_R_M)
    tol = visual_settings().finger_contact_tolerance_m
    assert gaps[finger] <= tol, (hand, finger, roll_deg, gaps[finger])


@pytest.mark.parametrize("roll_deg", _ROLLS_DEG)
@pytest.mark.parametrize("hand", (Hand.LEAD, Hand.TRAIL))
@pytest.mark.parametrize(
    "finger", ("index", "middle", "ring", "little"),
)
def test_shadow_capsule_pad_divergence_per_finger(
    roll_deg: float, hand: Hand, finger: str
) -> None:
    spec = _spec(roll_deg)
    pose = spec.to_grip_pose()
    _solved, gaps = solve_and_compare(
        hand, grip_spec=spec, grip_radius_m=_GRIP_R_M, grip_pose=pose
    )
    settings = pad_divergence_settings()
    assert gaps[finger] <= settings.per_finger_tolerance_m[finger], (
        hand,
        finger,
        roll_deg,
        gaps[finger],
    )


@pytest.mark.parametrize("roll_deg", _ROLLS_DEG)
def test_shadow_thumb_pad_divergence_documented_gap(roll_deg: float) -> None:
    """Thumb link geometry differs from capsules; record gap without widening digit tols."""
    spec = _spec(roll_deg)
    pose = spec.to_grip_pose()
    settings = pad_divergence_settings()
    for hand in (Hand.LEAD, Hand.TRAIL):
        _solved, gaps = solve_and_compare(
            hand, grip_spec=spec, grip_radius_m=_GRIP_R_M, grip_pose=pose
        )
        thumb = gaps["thumb"]
        assert thumb <= settings.thumb_documented_max_m, (hand, roll_deg, thumb)
        if thumb <= settings.per_finger_tolerance_m.get("index", 0.014):
            pytest.fail(
                "Shadow thumb matched capsule within digit tolerance — "
                "revisit thumb_documented_max_m"
            )