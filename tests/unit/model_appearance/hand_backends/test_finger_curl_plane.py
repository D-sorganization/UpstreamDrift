"""Fingers that curl in a plane turned from the palm's (epic #15150).

Owner, 2026-11-19: the club lies diagonally across the trail hand, yet the
fingers wrap it close to square, because each finger is turned at its knuckle
and not because the whole hand turns. ``preshape.curl_plane_turn_deg`` is that
turn, per hand and finger; the palm and the forearm stay where they were.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.curl_plane import (
    turn_curl_planes,
)
from src.shared.python.model_appearance.hand_backends.joint_angles import FINGERS
from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
    wrist_joint_m,
)
from src.shared.python.model_appearance.hand_backends.skinned_human import (
    build_hand_visual,
)
from src.shared.python.model_appearance.hand_backends.skinned_landmarks import (
    hand_landmarks,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
)
from tests.unit.model_appearance.hand_backends.test_skinned_human import (
    _RADIUS,
    _hand,
    _spec,
)

pytestmark = pytest.mark.unit

_T = cast(dict[str, Any], load_hand_visual_quality()["trail_grip"])


def _tilt_deg(hand: Any, finger: str) -> float:
    hinge = hand_landmarks(hand).joint(f"{finger}_mid").hinge_axis
    assert hinge is not None
    return math.degrees(math.acos(min(1.0, abs(float(hinge[2])))))


def test_every_hand_and_finger_has_a_turn_inside_the_knuckles_range() -> None:
    settings = default_settings()
    turns = {side: dict(per) for side, per in settings.curl_plane_turn_deg}
    assert set(turns) == {"lead", "trail"}
    most = settings.curl_plane_turn_max_deg
    for per in turns.values():
        assert set(per) == set(FINGERS)
        assert all(abs(turn) <= most for turn in per.values())
    assert not any(turns["lead"].values()), "the lead hand is not to change"
    assert any(turns["trail"].values())


def test_a_turn_no_knuckle_can_make_is_refused() -> None:
    raw = load_settings_mapping()
    most = float(raw["preshape"]["curl_plane_turn_max_deg"])
    raw["preshape"]["curl_plane_turn_deg"]["trail"]["ring"] = most + 1.0
    with pytest.raises(ValueError, match="curl_plane_turn_deg"):
        settings_from_mapping(raw)
    raw = load_settings_mapping()
    del raw["preshape"]["curl_plane_turn_deg"]["lead"]["little"]
    with pytest.raises(ValueError, match="curl_plane_turn_deg"):
        settings_from_mapping(raw)


def test_the_turn_is_about_the_palms_normal_and_of_the_named_finger_only() -> None:
    fixed = {f"{finger}_prox": np.eye(3) for finger in FINGERS}
    kept = fixed["index_prox"]
    turn_curl_planes(fixed, {"index": 0.0, "middle": 20.0, "ring": 0.0, "little": 0.0})
    assert fixed["index_prox"] is kept
    c, s = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
    np.testing.assert_allclose(
        fixed["middle_prox"], [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], atol=1e-12
    )
    with pytest.raises(ValueError, match="thumb"):
        turn_curl_planes(fixed, {"thumb": 5.0})


def test_turned_fingers_wrap_squarer_and_the_palm_stays() -> None:
    raw = load_settings_mapping()
    turns = raw["preshape"]["curl_plane_turn_deg"]["trail"]
    turn = float(turns["middle"])
    raw["preshape"]["curl_plane_turn_deg"]["trail"] = dict.fromkeys(turns, 0.0)
    flat = build_hand_visual(
        _spec(), "trail", grip_radius_m=_RADIUS, settings=settings_from_mapping(raw)
    )
    turned = _hand("trail")
    gain = _tilt_deg(flat, "middle") - _tilt_deg(turned, "middle")
    share = float(_T["min_curl_turn_share_squared"])
    assert gain >= share * turn, f"{turn:.0f} deg of turn squared {gain:.1f} deg"
    moved = float(np.linalg.norm(wrist_joint_m(turned) - wrist_joint_m(flat)))
    assert moved <= float(_T["max_wrist_shift_from_curl_turn_m"]), (
        f"{moved * 1e3:.1f} mm"
    )
