"""One hand's thumb may be laid to a target of its own (#15329, epic #15150).

``thumb.toward_index_target`` is what both thumbs are turned to;
``thumb.toward_index_target_of`` names a different one for a hand, so that the
trail thumb can close a V on its forefinger while the lead thumb stays on its
azimuth. Empty as shipped, which draws both hands as before.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.hand_backends import skinned_human
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    THUMB_TARGETS,
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
)

pytestmark = pytest.mark.unit

_SETTINGS = default_settings()
_RADIUS_M = 0.0105


def _settings(of_hand: dict[str, str]) -> skinned_human.SkinnedHumanSettings:
    raw = load_settings_mapping()
    raw["thumb"] = {**raw["thumb"], "toward_index_target_of": of_hand}
    return settings_from_mapping(raw)


def test_the_shipped_hands_share_one_thumb_target() -> None:
    assert _SETTINGS.thumb_target_of == ()
    for side in ("lead", "trail"):
        assert _SETTINGS.with_thumb_target(side) is _SETTINGS


def test_a_hand_given_its_own_target_is_laid_to_it_and_the_other_is_not() -> None:
    usual = _SETTINGS.thumb_toward_index_target
    other = next(target for target in THUMB_TARGETS if target != usual)
    settings = _settings({"trail": other})
    assert settings.with_thumb_target("lead") is settings
    assert settings.with_thumb_target("trail").thumb_toward_index_target == other
    assert settings.thumb_toward_index_target == usual
    same = _settings({"trail": usual})
    assert same.with_thumb_target("trail") is same


@pytest.mark.parametrize(
    "of_hand", ({"trail": "wrist"}, {"left": "knuckle"}, {"trail": 3}, ["trail"])
)
def test_a_target_or_a_hand_that_does_not_exist_is_refused(of_hand) -> None:
    with pytest.raises((ValueError, TypeError)):
        _settings(of_hand)
    with pytest.raises(ValueError):
        _SETTINGS.with_thumb_target("left")


def test_the_trail_thumb_closes_on_its_forefinger_and_the_lead_hand_stays() -> None:
    """On a lone hand: the trail thumb's end joint comes nearer the knuckle of
    its forefinger (the V), and the lead hand is solved exactly as before."""
    spec = GripSpec.legacy()
    settings = _settings({"trail": "knuckle"})

    def solved(side: str, by: skinned_human.SkinnedHumanSettings):
        return skinned_human._solve_right_hand(spec, side, _RADIUS_M, by, None)

    assert solved("lead", settings).angles == solved("lead", _SETTINGS).angles
    apart = []
    for by in (_SETTINGS, settings):
        hand = solved("trail", by)
        joint = hand.hand.joint_position("thumb_dist", hand.angles)
        knuckle = hand.hand.joint_position("index_prox", hand.angles)
        apart.append(float(np.linalg.norm(joint - knuckle)))
    assert apart[1] < apart[0], f"{apart[0] * 1e3:.1f} to {apart[1] * 1e3:.1f} mm"
