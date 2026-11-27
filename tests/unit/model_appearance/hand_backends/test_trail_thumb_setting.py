"""The trail thumb lies on the target side of the shaft (#15278, epic #15150)."""

from __future__ import annotations

import pytest

from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
)
from src.shared.python.model_appearance.skinned_club_hands import owner_grip_spec

pytestmark = pytest.mark.unit

_Q = load_hand_visual_quality()


def test_the_trail_thumb_is_set_well_to_the_target_side() -> None:
    """Owner review 2026-11-18: the right thumb sat too near the top of the
    shaft. The floor is a setting, and the grip the golfer is built with
    carries the configured angle."""
    floor = float(_Q["min_trail_thumb_target_side_deg"])
    trail = float(_Q["owner_thumb_azimuth_deg"]["trail"])
    assert trail >= floor
    assert owner_grip_spec().trail.thumb_azimuth_deg == pytest.approx(trail)
