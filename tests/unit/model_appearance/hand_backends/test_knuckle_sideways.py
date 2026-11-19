"""A bent knuckle has little sideways play (#15278, epic #15150).

Owner review 2026-11-19: the left forefinger had a crooked kink, "bending in a
way that fingers don't bend": a finger moves almost wholly in flexion and
extension, cannot rotate, and has limited lateral motion. The fit was swinging
knuckles up to 20 degrees sideways while they were bent 90 to 115 degrees. A
knuckle's sideways range now shrinks as it flexes; every number comes from
``scripts/config/skinned_human_hand.json`` (``side_close``).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends import skinned_human
from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.knuckle_sideways import (
    sideways_limit_rad,
    split_swing_rad,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
)
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
)
from tests.unit.model_appearance.hand_backends.test_skinned_human import (
    _hand,
    _pair,
)

pytestmark = pytest.mark.unit

#: The anatomical knuckle; ``side_close.spread_in_palm`` is still off by default.
_SETTINGS = dataclasses.replace(default_settings(), side_close_spread_in_palm=True)
_ASSET = skinned_human._sized_assets(0.19, 0.085, _SETTINGS)[0]
_MOST = math.radians(_SETTINGS.side_close_knuckle_max_deg)
_FLEXED = math.radians(_SETTINGS.side_close_flexed_at_deg)
_SHARE = _SETTINGS.side_close_flexed_share
_SLACK_DEG = float(load_hand_visual_quality()["knuckle_sideways_tolerance_deg"])


def test_the_sideways_range_shrinks_as_the_knuckle_flexes() -> None:
    assert 0.0 <= _SHARE < 1.0, "a bent knuckle has less play than a straight one"
    assert sideways_limit_rad(0.0, _SETTINGS) == pytest.approx(_MOST)
    assert sideways_limit_rad(-0.2, _SETTINGS) == pytest.approx(_MOST)
    assert sideways_limit_rad(_FLEXED, _SETTINGS) == pytest.approx(_SHARE * _MOST)
    assert sideways_limit_rad(2.0 * _FLEXED, _SETTINGS) == pytest.approx(_SHARE * _MOST)
    half = sideways_limit_rad(0.5 * _FLEXED, _SETTINGS)
    assert half == pytest.approx(0.5 * (1.0 + _SHARE) * _MOST)
    with pytest.raises(ValueError, match="finite"):
        sideways_limit_rad(math.nan, _SETTINGS)


def test_a_swing_is_a_turn_in_the_palm_with_a_lean_that_holds_it_in_range() -> None:
    asked = math.radians(20.0)
    assert split_swing_rad(asked, 0.0, _SETTINGS) == pytest.approx((0.0, asked))
    for flexion in (0.3, 0.8, _FLEXED, 1.9):
        lean, turn = split_swing_rad(-asked, flexion, _SETTINGS)
        assert turn == pytest.approx(-asked), "the fit keeps the whole spread"
        sideways = turn * math.cos(flexion) + lean
        assert abs(sideways) <= sideways_limit_rad(flexion, _SETTINGS) + 1e-12
    # Past the straight-finger range the swing is held.
    assert split_swing_rad(2.0 * _MOST, 0.0, _SETTINGS)[1] == pytest.approx(_MOST)


def _sideways_deg(hand: skinned_human._GripHand, flexion: float) -> float:
    """How far the ring finger's first bone leaves the path it flexes along."""
    angles = {**skinned_human._straight_angles(), "ring_prox": flexion}
    swung = hand.joint_position("ring_prox", angles, tail=True)
    base = hand.joint_position("ring_prox", angles)
    hand.swing_sideways(None)
    plain = hand.joint_position("ring_prox", angles, tail=True)
    cos = (swung - base) @ (plain - base)
    cos /= np.linalg.norm(swung - base) * np.linalg.norm(plain - base)
    return math.degrees(math.acos(min(1.0, float(cos))))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_swing_asked_of_a_bent_knuckle_is_held_to_its_range(sign: float) -> None:
    asked = 20.0
    for flexion_deg in (0.0, 45.0, 90.0, 110.0):
        hand = skinned_human._GripHand(_ASSET, _SETTINGS)
        hand.swing_sideways("ring_prox", sign * math.radians(asked))
        flexion = math.radians(flexion_deg)
        most = math.degrees(sideways_limit_rad(flexion, _SETTINGS))
        leaning = _sideways_deg(hand, flexion)
        assert leaning <= min(asked, most) + _SLACK_DEG, f"knuckle bent {flexion_deg}"
        if flexion_deg == 0.0:
            assert leaning == pytest.approx(asked, abs=0.05)


def test_with_the_setting_off_the_whole_swing_is_a_lean() -> None:
    asked = math.radians(20.0)
    assert split_swing_rad(asked, _FLEXED, default_settings()) == (asked, 0.0)


@pytest.mark.xfail(
    strict=True,
    reason="#15278: the gripping hands still lean bent knuckles sideways; "
    "spread_in_palm stays off until the trail hand's seat is refitted",
)
@pytest.mark.parametrize("side", ["lead", "trail"])
def test_no_knuckle_of_a_gripping_hand_moves_sideways_past_its_range(side: str) -> None:
    """The lean each hand is posed with, against its knuckle's range."""
    worst = 0.0
    for hand in (_hand(side), _pair()[side]):
        angles = hand.joint_angles.as_mapping()
        for finger in FINGERS:
            flexion = angles[f"{finger}_prox"]
            lean = abs(angles.get(f"{finger}_side", 0.0))
            worst = max(worst, lean - sideways_limit_rad(flexion, _SETTINGS))
    assert worst <= 0.0, f"{math.degrees(worst):.1f} degrees past the range"
