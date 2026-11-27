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


def test_a_seat_may_roll_the_palm_about_the_shaft() -> None:
    """``square_on_thumb`` rolls the palm round the grip onto the lead thumb;
    the seats that say nothing keep the hand settings' roll."""
    usual = dict(default_settings().palm_centre_wrap_deg)
    assert sch.trail_seat(_OLD).palm_centre_wrap_deg == usual["trail"]
    assert sch.trail_seat(_NEW).palm_centre_wrap_deg == usual["trail"]
    rolled = sch.trail_seat("square_on_thumb")
    assert rolled.palm_centre_wrap_deg > usual["trail"]
    wraps = dict(sch.seat_settings(rolled).palm_centre_wrap_deg)
    assert wraps == {"lead": usual["lead"], "trail": rolled.palm_centre_wrap_deg}


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
    assert _crown_gap_m(square) <= float(_Q["max_thumb_channel_gap_m"])


# -- square_on_thumb: square_wrap with the palm rolled back onto the lead thumb

_R = cast(dict[str, Any], _T["square_on_thumb"])
_ROLLED = _R["measured"]


@pytest.fixture(scope="module")
def rolled() -> dict[str, HandVisualResult]:
    return sch.build_seated_pair("square_on_thumb")


@pytest.fixture(scope="module")
def rolled_measured(rolled: dict[str, HandVisualResult]) -> TrailGripMeasures:
    return trail_grip_measures(rolled["lead"], rolled["trail"])


def _crown_gap_m(pair: dict[str, HandVisualResult]) -> float:
    gaps = thumb_crown_gaps_m(
        pair["lead"],
        pair["trail"],
        station_m=float(_Q["thumb_channel_station_m"]),
        min_facing=float(_Q["thumb_channel_crown_facing"]),
        min_station_vertices=int(_Q["thumb_channel_min_station_vertices"]),
    )
    return float(max(gaps))


def test_the_rolled_seat_wraps_the_fingers_squarer(rolled_measured) -> None:
    tilts = rolled_measured.wrap_tilt_deg
    assert all(tilts[f] <= float(_R["max_wrap_tilt_deg"]) for f in _HOLDING), tilts
    assert tilts["index"] <= float(_R["max_index_wrap_tilt_deg"]), tilts
    line = rolled_measured.knuckle_line_to_shaft_deg
    assert line <= float(_T["max_knuckle_line_to_shaft_deg"]), f"{line:.1f} deg"


def test_the_rolled_seat_lays_the_palm_back_on_the_lead_thumb(
    rolled, rolled_measured
) -> None:
    """The heel meets the owner's target at the thumb's knuckle; the crown gap
    is held to the seat's own bound and the hands stay out of each other."""
    heel = rolled_measured.heel_to_lead_thumb_knuckle_m
    assert heel <= float(_T["max_heel_to_lead_thumb_knuckle_m"]), f"{heel * 1e3:.1f}"
    gap = _crown_gap_m(rolled)
    assert gap <= float(_R["max_thumb_crown_gap_m"]), f"{gap * 1e3:.1f} mm"
    deep = hand_pair_penetration_m(rolled["lead"], rolled["trail"])
    assert deep <= float(_Q["pair_penetration_tolerance_m"]), f"{deep * 1e3:.2f} mm"


def test_the_rolled_seat_holds_the_knuckles_near_the_shaft_line(
    rolled_measured,
) -> None:
    most = float(_R["max_knuckle_lateral_offset_m"])
    for finger in _HOLDING:
        off = rolled_measured.knuckle_lateral_m[finger]
        assert abs(off) <= most, f"trail {finger} knuckle {off * 1e3:.1f} mm off"
        assert rolled_measured.knuckle_behind_m[finger] > 0.0


_ROLL = "#15150: owner's targets, not met by square_on_thumb: curl planes "
_ROLL += f"{_ROLLED['wrap_tilt']}; knuckles {_ROLLED['knuckle_lateral_offset']} "
_ROLL += f"off the shaft line; lead thumb crown gap {_ROLLED['thumb_crown_gap']}."


@pytest.mark.xfail(strict=True, reason=_ROLL)
def test_the_rolled_seat_wraps_square(rolled_measured) -> None:
    most = float(_T["max_wrap_tilt_deg"])
    assert all(rolled_measured.wrap_tilt_deg[f] <= most for f in _HOLDING)


@pytest.mark.xfail(strict=True, reason=_ROLL)
def test_the_rolled_seat_knuckles_are_on_the_shaft_line(rolled_measured) -> None:
    most = float(_T["max_knuckle_lateral_offset_m"])
    assert all(abs(rolled_measured.knuckle_lateral_m[f]) <= most for f in _HOLDING)


@pytest.mark.xfail(strict=True, reason=_ROLL)
def test_the_rolled_seat_palm_covers_the_whole_lead_thumb(rolled) -> None:
    assert _crown_gap_m(rolled) <= float(_Q["max_thumb_channel_gap_m"])
