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
)
from tests.unit.model_appearance.hand_backends.test_skinned_human import _pair

pytestmark = pytest.mark.unit

_Q = load_hand_visual_quality()


_PENDING = (
    "open owner decision (#15278): the trigger finger that meets this breaks "
    "the neighbouring-fingers-touch limit, so it is off by default"
)


class _FakeHand:
    """A forefinger whose end joint moves away from the middle finger's as it
    swings positive; enough to drive :func:`spread_trigger_finger`."""

    def __init__(self, spread_deg: float, closed: float) -> None:
        self.settings = SimpleNamespace(
            side_close_trigger_spread_deg=(("lead", 0.0), ("trail", spread_deg)),
            curl=SimpleNamespace(step_rad=math.radians(2.0)),
        )
        self.side = {"index_prox": closed}

    def swing_sideways(self, bone: str | None, swing: float = 0.0) -> None:
        assert bone == "index_prox"
        self.side[bone] = swing

    def joint_position(self, bone: str, angles: object) -> np.ndarray:
        if bone == "middle_dist":
            return np.zeros(3)
        return np.array([0.02 + 0.05 * math.sin(self.side["index_prox"]), 0.0, 0.0])


def test_the_trigger_spread_is_a_setting_per_hand() -> None:
    spread = dict(default_settings().side_close_trigger_spread_deg)
    assert set(spread) == {"lead", "trail"}
    assert spread["lead"] == 0.0, "the lead forefinger is never a trigger finger"
    raw = load_settings_mapping()
    bad = {**raw["side_close"], "trigger_spread_deg": {"lead": 0.0, "trail": -1.0}}
    with pytest.raises(ValueError, match="trail"):
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


@pytest.mark.xfail(strict=True, reason=_PENDING)
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
