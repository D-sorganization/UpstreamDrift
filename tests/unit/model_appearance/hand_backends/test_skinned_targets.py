"""The turn search that puts a thumb on its line (epic #15150, #15181)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    load_settings_mapping,
    settings_from_mapping,
)
from src.shared.python.model_appearance.hand_backends.skinned_targets import (
    scanned_turn,
)

pytestmark = pytest.mark.unit


def _tracked(miss_at):
    tried: list[float] = []

    def missed(turn: float) -> tuple[float, dict[str, float]]:
        tried.append(turn)
        return miss_at(turn), {"turn": turn}

    return missed, tried


def test_a_miss_within_the_tolerance_is_left_alone() -> None:
    missed, tried = _tracked(lambda turn: 0.02 - turn)
    assert scanned_turn(missed, 1.0, 8, 0.05) == {"turn": 0.0}
    assert tried == [0.0]


def test_the_least_turn_toward_the_miss_that_is_close_enough_wins() -> None:
    missed, tried = _tracked(lambda turn: -0.4 - turn)
    pose = scanned_turn(missed, 1.0, 12, 0.05)
    assert abs(pose["turn"] + 0.4) <= 0.05
    assert tried == sorted(tried, reverse=True)  # from zero outward, in even steps
    assert tried[-1] == pose["turn"]
    np.testing.assert_allclose(np.diff(tried), -1.0 / 12)


def test_across_a_jump_the_closest_pose_tried_is_kept() -> None:
    """A thumb that lifts jumps past its line: no turn lands on it."""

    def miss_at(turn: float) -> float:
        return -0.5 + 0.1 * turn if turn > -0.33 else 0.2 + 0.1 * turn

    missed, tried = _tracked(miss_at)
    pose = scanned_turn(missed, 1.0, 8, 0.01)
    assert len(tried) == 9
    assert abs(miss_at(pose["turn"])) == min(abs(miss_at(turn)) for turn in tried)
    assert pose["turn"] <= -0.33


def test_no_room_to_turn_keeps_the_first_pose() -> None:
    missed, tried = _tracked(lambda turn: 0.3)
    assert scanned_turn(missed, 0.0, 8, 0.01) == {"turn": 0.0}
    assert tried == [0.0]


def test_the_tolerance_must_not_be_negative() -> None:
    missed, _ = _tracked(lambda turn: math.pi)
    with pytest.raises(ValueError, match="close_rad"):
        scanned_turn(missed, 1.0, 8, -0.1)


def test_the_trail_hand_of_a_pair_has_its_own_palm_cup_limit() -> None:
    """Its palm lies on the lead thumb, so it cannot cup as far (#15181)."""
    raw = load_settings_mapping()
    raw["thumb"] = {**raw["thumb"], "palm_cup_max_deg": 90.0}
    raw["pair"] = {**raw["pair"], "palm_cup_max_deg": 55.0}
    alone = settings_from_mapping(raw)
    paired = alone.beside_other_hand()
    assert (alone.palm_cup_max_deg, paired.palm_cup_max_deg) == (90.0, 55.0)
    assert paired.pair_palm_cup_max_deg == 55.0
    assert paired.thumb_azimuth_close_rad == alone.thumb_azimuth_close_rad
    raw["pair"] = {**raw["pair"], "palm_cup_max_deg": -1.0}
    with pytest.raises(ValueError, match="palm_cup_max_deg"):
        settings_from_mapping(raw)
