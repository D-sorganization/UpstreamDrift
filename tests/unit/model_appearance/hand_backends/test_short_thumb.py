"""A hand's thumb may be held flexed, a short thumb (#15329, epic #15150).

Golf teaching tells a long lead thumb, laid straight down the grip, from a
short one, pulled up so that it barely passes the forefinger.
``thumb.held_flexion_deg_of`` holds the two free joints of a hand's thumb at
least so far flexed, and the thumb is then aimed as that bent shape, so its
tip stays on the grip nearer the hand. Empty as shipped, which draws both
hands as before.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends import skinned_human
from src.shared.python.model_appearance.hand_backends.joint_angles import HandSide
from src.shared.python.model_appearance.hand_backends.posing import cylinder_distance
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
)

pytestmark = pytest.mark.unit

_SETTINGS = default_settings()
_RADIUS_M = 0.0105
_HELD_DEG = (40.0, 10.0)
#: Least the held flexion above pulls the lone lead thumb's tip up the grip
#: (measured 2026-11-20 on the legacy grip: 5.3 mm; see
#: ``_held_flexion_deg_of`` in the hand settings for the owner's grip).
_LEAST_PULL_M = 0.004
#: Most the short thumb's tip may leave the long thumb's line round the grip.
_MOST_TURN_DEG = 5.0


def _settings(of_hand: object) -> skinned_human.SkinnedHumanSettings:
    raw = load_settings_mapping()
    raw["thumb"] = {**raw["thumb"], "held_flexion_deg_of": of_hand}
    return settings_from_mapping(raw)


def test_the_shipped_thumbs_are_held_at_no_flexion() -> None:
    assert _SETTINGS.thumb_held_flexion_of == ()
    assert _SETTINGS.thumb_held_flexion_deg == (0.0, 0.0)
    for side in ("lead", "trail"):
        assert _SETTINGS.with_short_thumb(side) is _SETTINGS


def test_a_hand_named_a_short_thumb_has_it_and_the_other_does_not() -> None:
    settings = _settings({"lead": list(_HELD_DEG)})
    assert settings.thumb_held_flexion_deg == (0.0, 0.0)
    assert settings.with_short_thumb("trail") is settings
    assert settings.with_short_thumb("lead").thumb_held_flexion_deg == _HELD_DEG
    straight = _settings({"lead": [0.0, 0.0]})
    assert straight.with_short_thumb("lead") is straight


@pytest.mark.parametrize(
    "of_hand",
    (
        {"left": [10.0, 10.0]},
        {"lead": [10.0]},
        {"lead": [-5.0, 0.0]},
        {"lead": [10.0, 400.0]},
        {"lead": [math.nan, 0.0]},
        {"lead": 10.0},
        ["lead"],
    ),
)
def test_a_flexion_the_thumb_cannot_take_is_refused(of_hand: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        _settings(of_hand)
    with pytest.raises(ValueError):
        _SETTINGS.with_short_thumb("left")


def test_a_short_lead_thumb_ends_nearer_the_hand_and_stays_on_the_grip() -> None:
    """On a lone hand: the lead thumb's tip is pulled up the grip, its joints
    are flexed at least as far as held, its skin is not in the grip, and the
    trail hand is solved exactly as before."""
    spec = GripSpec.legacy()
    settings = _settings({"lead": list(_HELD_DEG)})

    def solved(side: HandSide, by: skinned_human.SkinnedHumanSettings):
        return skinned_human._solve_right_hand(spec, side, _RADIUS_M, by, None)

    assert solved("trail", settings).angles == solved("trail", _SETTINGS).angles
    long, short = solved("lead", _SETTINGS), solved("lead", settings)

    def tip(hand) -> np.ndarray:
        return hand.hand.joint_position("thumb_dist", hand.angles, tail=True)

    pulled = float(tip(long)[2] - tip(short)[2])
    assert pulled >= _LEAST_PULL_M, f"tip pulled {pulled * 1e3:.1f} mm up the grip"
    round_grip = [math.degrees(math.atan2(p[1], p[0])) for p in (tip(long), tip(short))]
    turned = abs(round_grip[1] - round_grip[0])
    assert turned <= _MOST_TURN_DEG, f"tip {turned:.1f} deg off the long thumb's line"
    for bone, held in zip(("thumb_prox", "thumb_dist"), _HELD_DEG, strict=True):
        assert math.degrees(short.angles[bone]) >= held - 1e-9
    # The pads give: a thumb is pressed squeeze.finger_depth_m into the grip.
    deepest = _SETTINGS.squeeze_finger_depth_m + _SETTINGS.curl.penetration_allowance_m
    grip = cylinder_distance(_RADIUS_M, _SETTINGS.grip_taper)
    for bone in ("thumb_prox", "thumb_dist"):
        skin = short.hand.points(short.angles, short.hand.segment_ids(bone))
        depth = float(np.min(grip(skin)))
        assert depth >= -deepest - 1e-6, f"{bone} {depth * 1e3:.2f} mm into the grip"
