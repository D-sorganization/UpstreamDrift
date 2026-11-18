"""The trail palm's channel rests on the lead thumb (#15265, epic #15150).

Every tolerance comes from ``scripts/config/hand_visual_quality.json``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.qa import (
    hand_pair_penetration_m,
    load_hand_visual_quality,
    thumb_crown_gaps_m,
)
from tests.unit.model_appearance.hand_backends.test_skinned_human import _pair

pytestmark = pytest.mark.unit

_Q = load_hand_visual_quality()
_STATION = float(_Q["thumb_channel_station_m"])
_FACING = float(_Q["thumb_channel_crown_facing"])


def _gaps(lead, trail) -> np.ndarray:
    return thumb_crown_gaps_m(lead, trail, station_m=_STATION, min_facing=_FACING)


def test_the_trail_palm_rests_on_the_lead_thumb_along_its_length() -> None:
    """Owner: the channel of the trail palm rests against the lead thumb, the
    hands in contact there, from the base of the thumb to its tip."""
    pair = _pair()
    gaps = _gaps(pair["lead"], pair["trail"])
    limit = float(_Q["max_thumb_channel_gap_m"])
    assert len(gaps) >= 8, "the thumb is cut into too few stations to judge"
    assert gaps.max() <= limit, f"gaps along the thumb (mm): {np.round(gaps * 1e3, 1)}"
    # Resting on it, not pushed through it.
    depth = hand_pair_penetration_m(pair["lead"], pair["trail"])
    assert depth <= float(_Q["pair_penetration_tolerance_m"])


def test_the_thumb_gap_grows_when_the_trail_hand_is_taken_away() -> None:
    """The measure sees a palm that is off the thumb, by about as much."""
    pair = _pair()
    trail = pair["trail"]
    away = dataclasses.replace(
        trail,
        mesh=dataclasses.replace(
            trail.mesh, vertices=trail.mesh.vertices + np.array([0.0, 0.0, 0.03])
        ),
    )
    near, far = _gaps(pair["lead"], trail), _gaps(pair["lead"], away)
    assert far[0] > near[0] + 0.01, "the base of the thumb is no longer covered"
    assert len(far) == len(near)


def test_the_thumb_measure_refuses_bad_settings() -> None:
    pair = _pair()
    for station, facing in ((0.0, 0.5), (0.005, 0.0), (0.005, 1.0)):
        with pytest.raises(ValueError, match="station_m"):
            thumb_crown_gaps_m(
                pair["lead"], pair["trail"], station_m=station, min_facing=facing
            )
