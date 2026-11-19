"""The trail forefinger is a trigger finger (#15278, epic #15150).

Owner review 2026-11-18: the right thumb reached too far down the shaft. Its
tip belongs about level with the centre of the forefinger's first joint, and
that joint lies on the away-from-target side of the shaft. Every number comes
from ``scripts/config/hand_visual_quality.json``.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.qa import (
    finger_side_gaps_m,
    load_hand_visual_quality,
)
from src.shared.python.model_appearance.hand_backends.skinned_landmarks import (
    hand_landmarks,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
)
from src.shared.python.model_appearance.hand_backends.trigger_finger import (
    spread_trigger_finger,
    wrap_trigger_finger,
)
from tests.unit.model_appearance.hand_backends.test_skinned_human import (
    _hand,
    _pair,
)

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group("skinned_hand_backends")]

_Q = load_hand_visual_quality()


class _FakeHand:
    """A forefinger whose end joint moves away from the middle finger's as it
    swings positive; enough to drive :func:`spread_trigger_finger`."""

    def __init__(self, spread_deg: float, closed: float) -> None:
        self.settings = SimpleNamespace(
            side_close_trigger_spread_deg=(("lead", 0.0), ("trail", spread_deg)),
            side_close_trigger_back_off_deg=6.0,
            curl=SimpleNamespace(step_rad=math.radians(2.0), contact_tolerance_m=0.001),
        )
        self.side = {"index_prox": closed}
        self.closed = closed
        self.wraps = 0

    def swing_sideways(self, bone: str | None, swing: float = 0.0) -> None:
        assert bone == "index_prox"
        self.side[bone] = swing

    def joint_position(self, bone: str, angles: object) -> np.ndarray:
        if bone == "middle_dist":
            return np.zeros(3)
        return np.array([0.02 + 0.05 * math.sin(self.side["index_prox"]), 0.0, 0.0])

    def segment_ids(self, bone: str) -> np.ndarray:
        assert bone == "index_dist"
        return np.array([1])

    def points(self, angles: object) -> np.ndarray:
        """Two points; the second is the fingertip, as far off the grip (x) as
        ``lift_m_per_deg`` times the spread it was wrapped at."""
        return np.array([[0.0, 0.0, 0.0], [float(angles["lift"]), 0.0, 0.0]])  # type: ignore[index]

    def wrap(self, lift_m_per_deg: float) -> dict[str, float]:
        self.wraps += 1
        spread = abs(math.degrees(self.side["index_prox"] - self.closed))
        return {"spread": spread, "lift": lift_m_per_deg * max(spread - 20.0, 0.0)}


def _off_grip(points: np.ndarray) -> np.ndarray:
    return np.asarray(points[:, 0])


def test_the_trigger_spread_is_a_setting_per_hand() -> None:
    spread = dict(default_settings().side_close_trigger_spread_deg)
    assert set(spread) == {"lead", "trail"}
    assert spread["lead"] == 0.0, "the lead forefinger is never a trigger finger"
    assert spread["trail"] > 0.0, "owner, 2026-11-18: the trail trigger finger is on"
    assert default_settings().side_close_trigger_back_off_deg > 0.0
    raw = load_settings_mapping()
    bad = {**raw["side_close"], "trigger_spread_deg": {"lead": 0.0, "trail": -1.0}}
    with pytest.raises(ValueError, match="trail"):
        settings_from_mapping({**raw, "side_close": bad})
    bad = {**raw["side_close"], "trigger_back_off_deg": 0.0}
    with pytest.raises(ValueError, match="trigger_back_off_deg"):
        settings_from_mapping({**raw, "side_close": bad})


def test_the_spread_swings_the_forefinger_away_from_the_middle_finger() -> None:
    closed = math.radians(-20.0)
    hand = _FakeHand(30.0, closed)
    spread_trigger_finger(hand, {}, "trail")
    assert hand.side["index_prox"] == pytest.approx(closed + math.radians(30.0))
    spread_trigger_finger(hand, {}, "lead")  # zero spread: untouched
    assert hand.side["index_prox"] == pytest.approx(closed + math.radians(30.0))
    with pytest.raises(ValueError, match="hand_side"):
        spread_trigger_finger(hand, {}, "left")


def test_the_spread_is_kept_while_the_fingertip_seats() -> None:
    hand = _FakeHand(30.0, math.radians(-20.0))
    angles = wrap_trigger_finger(
        hand, "trail", {"closed": 0.0}, _off_grip, lambda: hand.wrap(0.0)
    )
    assert angles["spread"] == pytest.approx(30.0) and hand.wraps == 1


def test_the_spread_backs_off_until_the_fingertip_seats() -> None:
    """A finger held so far apart that its tip hangs off the grip is brought
    back a step at a time; the pose returned is the one wrapped at that spread."""
    hand = _FakeHand(30.0, math.radians(-20.0))
    angles = wrap_trigger_finger(
        hand, "trail", {"closed": 0.0}, _off_grip, lambda: hand.wrap(0.0005)
    )
    # 30 lifts 5 mm, 24 lifts 2 mm, 18 seats.
    assert angles["spread"] == pytest.approx(18.0) and hand.wraps == 3
    assert hand.side["index_prox"] == pytest.approx(math.radians(-2.0))
    never = _FakeHand(8.0, 0.0)
    angles = wrap_trigger_finger(
        never, "trail", {"closed": 0.0}, lambda p: p[:, 0] + 1.0, lambda: never.wrap(0)
    )
    assert angles["spread"] == pytest.approx(0.0), "no further back than closed"


def test_without_side_closing_there_is_no_trigger_finger() -> None:
    hand = _FakeHand(30.0, 0.0)
    wrap_trigger_finger(hand, "trail", {}, _off_grip, lambda: hand.wrap(0.0))
    assert hand.side["index_prox"] == 0.0 and hand.wraps == 1


@pytest.mark.parametrize("name", ["lone", "pair"])
def test_the_trigger_finger_stands_apart_and_its_tip_stays_on_the_grip(
    name: str,
) -> None:
    """Owner, 2026-11-18: the trigger finger is exempt from the rule that
    neighbouring fingers touch. On the pair it is held apart within the
    configured band. A lone hand's finger is brought back until its tip seats,
    so it need only stand further apart than a closed finger may. On both the
    tip stays on the grip."""
    closed_most = float(_Q["max_index_side_gap_m"])
    low, high = (float(v) for v in _Q["trigger_index_side_gap_m"])
    assert low > closed_most
    floors = {"lone": closed_most, "pair": low}
    hand = _hand("trail") if name == "lone" else _pair()["trail"]
    gap = finger_side_gaps_m(hand)["index-middle"]
    assert floors[name] < gap <= high, f"{name} index-middle {gap * 1e3:.1f} mm"
    tip = hand.contact_gaps_m["index_dist"]
    assert tip <= float(_Q["contact_tolerance_m"]), f"{name} tip {tip * 1e3:.2f}"


def test_the_trail_thumb_tip_is_about_level_with_the_forefinger_joint() -> None:
    """Along the shaft (+z runs down it) the thumb's tip is no more than the
    configured distance past the centre of the forefinger's first joint."""
    marks = hand_landmarks(_pair()["trail"])
    joint = marks.joint("index_mid").origin_m
    past = float(marks.points_m["thumb_tip"][2] - joint[2])
    limit = float(_Q["max_trail_thumb_past_index_joint_m"])
    assert past <= limit, f"thumb tip is {past * 1e3:.1f} mm past the joint"


def test_the_trail_forefinger_joint_is_on_the_away_from_target_side() -> None:
    """Positive azimuth is the away-from-target side of the grip's top, where
    the lead thumb lies; the joint sits within the configured band there."""
    joint = hand_landmarks(_pair()["trail"]).joint("index_mid").origin_m
    azimuth = math.degrees(math.atan2(joint[1], joint[0]))
    low, high = (float(v) for v in _Q["trail_index_joint_azimuth_deg"])
    assert low <= azimuth <= high, f"joint at {azimuth:.1f} degrees"
