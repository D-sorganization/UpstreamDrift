"""The trail hand's seat on the grip is a named setting (epic #15150).

``owner_trail_seat`` in ``hand_visual_quality.json`` names one of
``owner_trail_seats``. ``palm_on_thumb`` is the seat drawn since #15265;
``square_wrap`` turns the hand squarer to the shaft and bends the wrist toward
the little finger so that the forearm still lines up. The measured targets of
``square_wrap`` are ``trail_grip.square_wrap`` in the same file.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance.hand_backends.qa import (
    hand_pair_penetration_m,
    load_hand_visual_quality,
    thumb_crown_gaps_m,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
)
from src.shared.python.model_appearance.hand_backends.trail_grip_qa import (
    TrailGripMeasures,
    trail_grip_measures,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

pytestmark = pytest.mark.unit

_Q = load_hand_visual_quality()
_T = cast(dict[str, Any], _Q["trail_grip"])
_S = cast(dict[str, Any], _T["square_wrap"])
_HOLDING = ("middle", "ring")
_OLD, _NEW = "palm_on_thumb", "square_wrap"


def test_the_default_seat_is_the_one_drawn_before() -> None:
    """Selecting nothing draws the hands as before: same grip, same wrist."""
    seat = sch.trail_seat()
    assert seat.name == _Q["owner_trail_seat"] == _OLD
    assert seat.club_cross_angle_deg == float(_Q["owner_trail_cross_angle_deg"])
    assert sch.owner_grip_spec() == sch.owner_grip_spec(seat=_OLD)
    assert sch.seat_settings(seat) is default_settings()


def test_the_square_seat_turns_the_hand_and_bends_the_wrist() -> None:
    old, new = sch.trail_seat(_OLD), sch.trail_seat(_NEW)
    assert new.club_cross_angle_deg < old.club_cross_angle_deg
    assert new.wrist_ulnar_deviation_deg > old.wrist_ulnar_deviation_deg
    spec = sch.owner_grip_spec(seat=_NEW)
    assert spec.trail.club_cross_angle_deg == new.club_cross_angle_deg
    assert spec.lead == sch.owner_grip_spec().lead
    wrists = dict(sch.seat_settings(new).wrist_deg)
    assert wrists["trail"] == (new.wrist_extension_deg, new.wrist_ulnar_deviation_deg)
    assert wrists["lead"] == dict(default_settings().wrist_deg)["lead"]


def test_a_seat_that_is_not_in_the_table_is_refused() -> None:
    with pytest.raises(ValueError, match="no trail seat named"):
        sch.trail_seat("hammer")
    with pytest.raises(TypeError):
        sch.trail_seat(3)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def square() -> dict[str, HandVisualResult]:
    return sch.build_seated_pair(_NEW)


@pytest.fixture(scope="module")
def measured(square: dict[str, HandVisualResult]) -> TrailGripMeasures:
    return trail_grip_measures(square["lead"], square["trail"])


def test_the_square_seat_wraps_the_fingers_squarer(measured) -> None:
    """The seat's own bounds on the curl planes, and the owner's knuckle line."""
    for finger in _HOLDING:
        tilt = measured.wrap_tilt_deg[finger]
        assert tilt <= float(_S["max_wrap_tilt_deg"]), f"{finger} {tilt:.1f} deg"
    index = measured.wrap_tilt_deg["index"]
    assert index <= float(_S["max_index_wrap_tilt_deg"]), f"index {index:.1f} deg"
    line = measured.knuckle_line_to_shaft_deg
    assert line <= float(_T["max_knuckle_line_to_shaft_deg"]), f"{line:.1f} deg"


def test_the_square_seat_puts_the_knuckles_behind_the_shaft_on_its_line(
    measured,
) -> None:
    """The owner's target, met: middle and ring knuckles on the shaft's line."""
    most = float(_T["max_knuckle_lateral_offset_m"])
    for finger in _HOLDING:
        off = measured.knuckle_lateral_m[finger]
        assert abs(off) <= most, f"trail {finger} knuckle {off * 1e3:.1f} mm off"
        assert measured.knuckle_behind_m[finger] > 0.0


def test_the_square_seat_keeps_the_hands_out_of_each_other(square) -> None:
    deep = hand_pair_penetration_m(square["lead"], square["trail"])
    assert deep <= float(_Q["pair_penetration_tolerance_m"]), f"{deep * 1e3:.2f} mm"


def test_the_square_seat_keeps_the_wrist_arch(measured) -> None:
    arch = measured.wrist_extension_deg
    assert arch >= float(_T["measured_most"]["least_wrist_extension_deg"]) - 1e-6


_WAS = _S["measured"]
_SQUARE = f"#15150: owner's target, not met by square_wrap: {_WAS['wrap_tilt']}. "
_SQUARE += "A rigid hand's fingers stand off square by about the angle the club "
_SQUARE += "lies across it, and squarer carries the palm off the lead thumb."


@pytest.mark.xfail(strict=True, reason=_SQUARE)
def test_the_square_seat_meets_the_owner_square_target(measured) -> None:
    most = float(_T["max_wrap_tilt_deg"])
    assert all(measured.wrap_tilt_deg[f] <= most for f in _HOLDING)
    assert measured.wrap_tilt_deg["index"] <= float(_T["max_index_wrap_tilt_deg"])


_HEEL = "#15150: given up for squarer fingers: the square_wrap heel is "
_HEEL += f"{_WAS['heel_to_lead_thumb_knuckle']} from the lead thumb's knuckle."


@pytest.mark.xfail(strict=True, reason=_HEEL)
def test_the_square_seat_heel_is_at_the_lead_thumb_knuckle(measured) -> None:
    heel = measured.heel_to_lead_thumb_knuckle_m
    assert heel <= float(_T["max_heel_to_lead_thumb_knuckle_m"])


_COVER = "#15150: given up for squarer fingers: square_wrap leaves the base of "
_COVER += f"the lead thumb bare by {_WAS['thumb_crown_gap']}; palm_on_thumb holds "
_COVER += "max_thumb_channel_gap_m and stays the default."


@pytest.mark.xfail(strict=True, reason=_COVER)
def test_the_square_seat_palm_lies_on_the_lead_thumb(square) -> None:
    gaps = thumb_crown_gaps_m(
        square["lead"],
        square["trail"],
        station_m=float(_Q["thumb_channel_station_m"]),
        min_facing=float(_Q["thumb_channel_crown_facing"]),
        min_station_vertices=int(_Q["thumb_channel_min_station_vertices"]),
    )
    assert float(max(gaps)) <= float(_Q["max_thumb_channel_gap_m"])
