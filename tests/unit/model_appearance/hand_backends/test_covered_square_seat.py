"""The ``covered_square`` seat of the trail hand (#15329, epic #15150).

Owner, 2026-11-20: on the rigid palm, the trail palm over the lead thumb wins
over fingers square to the shaft. This seat keeps the palm on the thumb and,
inside that, wraps the fingers as square as the knuckles allow: the set
metacarpal arch, the middle knuckle abducted toward the little finger and the
middle and ring knuckles held inside their sideways range
(``side_close.knuckle_abduction_deg``). Its bounds, and what it measured, are
``trail_grip.covered_square`` in ``hand_visual_quality.json``. It is not the
default seat.
"""

from __future__ import annotations

import copy
import math
from typing import Any, cast

import pytest

from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance.hand_backends.knuckle_sideways import (
    sideways_limit_rad,
)
from src.shared.python.model_appearance.hand_backends.qa import (
    hand_pair_penetration_m,
    hand_to_finger_gaps_m,
    load_hand_visual_quality,
    thumb_crown_gaps_m,
)
from src.shared.python.model_appearance.hand_backends.skinned_pair import pair_fault_m
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
    load_settings_mapping,
    settings_from_mapping,
)
from src.shared.python.model_appearance.hand_backends.trail_grip_qa import (
    TrailGripMeasures,
    trail_grip_measures,
)
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult

pytestmark = pytest.mark.unit

_Q = load_hand_visual_quality()
_T = cast(dict[str, Any], _Q["trail_grip"])
_C = cast(dict[str, Any], _T["covered_square"])
_WAS = _C["measured"]
_SEAT = "covered_square"
_HOLDING = ("middle", "ring")


def test_no_knuckle_is_abducted_unless_a_hand_is_set_to() -> None:
    usual = dict(default_settings().side_close_knuckle_abduction_deg)
    assert usual == {"lead": (), "trail": ()}, "off by default"
    assert default_settings().side_close_abduction_hold_passes >= 1
    most = default_settings().side_close_knuckle_max_deg
    for bad in ({"thumb": 5.0}, {"middle": -1.0}, {"ring": most + 1.0}):
        raw = copy.deepcopy(dict(load_settings_mapping()))
        raw["side_close"]["knuckle_abduction_deg"]["trail"] = bad
        with pytest.raises(ValueError, match="knuckle_abduction_deg"):
            settings_from_mapping(raw)


def test_the_covered_seat_sets_the_trail_knuckles_and_nothing_else() -> None:
    seat = sch.trail_seat(_SEAT)
    assert _Q["owner_trail_seat"] != _SEAT, "the default seat is not changed"
    abducted = dict(seat.knuckle_abduction_deg)
    assert set(abducted) == set(_HOLDING) and abducted["middle"] > 0.0
    settings = sch.seat_settings(seat)
    swings = dict(settings.side_close_knuckle_abduction_deg)
    assert swings == {"lead": (), "trail": seat.knuckle_abduction_deg}
    assert settings.metacarpal_arch.of("lead") == {}
    usual = default_settings()
    assert dict(settings.wrist_deg)["lead"] == dict(usual.wrist_deg)["lead"]
    assert sch.owner_grip_spec(seat=_SEAT).lead == sch.owner_grip_spec().lead
    for other in ("palm_on_thumb", "square_wrap", "square_on_thumb", "arched_square"):
        assert sch.trail_seat(other).knuckle_abduction_deg == ()
    assert sch.seat_settings(sch.trail_seat()) is usual


# Building this seat's pair holds the knuckles in range over several wraps and
# takes over a minute, past the default lane's per-test limit, so every test
# that needs the built pair runs in the slow lane.
_BUILT = pytest.mark.slow


@pytest.fixture(scope="module")
def covered() -> dict[str, HandVisualResult]:
    return sch.build_seated_pair(_SEAT)


@pytest.fixture(scope="module")
def measured(covered: dict[str, HandVisualResult]) -> TrailGripMeasures:
    return trail_grip_measures(covered["lead"], covered["trail"])


@_BUILT
def test_the_covered_seat_keeps_the_palm_over_the_lead_thumb(
    covered: dict[str, HandVisualResult], measured: TrailGripMeasures
) -> None:
    """The hard constraint: the palm by the lead thumb's base, and no more of
    the thumb's crown bare than the default seat is allowed."""
    heel = measured.heel_to_lead_thumb_base_m
    assert heel <= float(_C["max_heel_to_lead_thumb_base_m"]), f"{heel * 1e3:.1f} mm"
    gaps = thumb_crown_gaps_m(
        covered["lead"],
        covered["trail"],
        station_m=float(_Q["thumb_channel_station_m"]),
        min_facing=float(_Q["thumb_channel_crown_facing"]),
        min_station_vertices=int(_Q["thumb_channel_min_station_vertices"]),
    )
    assert max(gaps) <= float(_Q["max_thumb_channel_gap_m"]), f"{max(gaps) * 1e3:.1f}"


@_BUILT
def test_the_covered_seat_wraps_the_holding_fingers_squarer_than_the_default(
    measured: TrailGripMeasures,
) -> None:
    before = float(_T["measured_most"]["wrap_tilt_deg"])
    gain = float(_C["min_wrap_tilt_gain_deg"])
    for finger in _HOLDING:
        tilt = measured.wrap_tilt_deg[finger]
        assert tilt <= float(_C["max_wrap_tilt_deg"][finger]), f"{finger} {tilt:.1f}"
        assert tilt <= before - gain, f"{finger} {tilt:.1f} deg off square"
    line = measured.knuckle_line_to_shaft_deg
    assert line <= float(_T["max_knuckle_line_to_shaft_deg"]), f"{line:.1f} deg"
    assert measured.wrist_extension_deg >= _T["wrist_extension_band_deg"][0] - 1e-6


@_BUILT
def test_the_covered_seat_keeps_the_hands_seated_and_out_of_each_other(
    covered: dict[str, HandVisualResult],
) -> None:
    lead, trail = covered["lead"], covered["trail"]
    deep = hand_pair_penetration_m(lead, trail)
    assert deep <= float(_Q["pair_penetration_tolerance_m"]), f"{deep * 1e3:.2f} mm"
    settings = sch.seat_settings(sch.trail_seat(_SEAT))
    fault = pair_fault_m(trail, lead, settings)
    assert fault <= float(_Q["contact_tolerance_m"]), f"{fault * 1e3:.2f} mm"
    ring = hand_to_finger_gaps_m(trail, lead, "index")["ring"]
    assert ring <= float(_Q["max_pair_ring_finger_gap_m"]), "ring beside forefinger"


@_BUILT
def test_the_covered_seat_swings_no_holding_knuckle_past_its_range(
    covered: dict[str, HandVisualResult],
) -> None:
    """Abduction is anatomical: inside the range a knuckle has as it is flexed."""
    settings = sch.seat_settings(sch.trail_seat(_SEAT))
    angles = covered["trail"].joint_angles.as_mapping()
    slack = math.radians(float(_Q["knuckle_sideways_tolerance_deg"]))
    assert abs(angles["middle_side"]) > 0.0, "the middle knuckle is abducted"
    for finger in _HOLDING:
        most = sideways_limit_rad(angles[f"{finger}_prox"], settings)
        swing = abs(angles[f"{finger}_side"])
        assert swing <= most + slack, f"{finger} {math.degrees(swing - most):.1f} past"


_LITTLE = "#15329: not met by covered_square: the little finger lies on the lead "
_LITTLE += f"middle finger, {_WAS['little_to_lead_forefinger']} off the forefinger."


@_BUILT
@pytest.mark.xfail(strict=True, reason=_LITTLE)
def test_the_covered_seat_little_finger_touches_the_lead_forefinger(
    covered: dict[str, HandVisualResult],
) -> None:
    gap = hand_to_finger_gaps_m(covered["trail"], covered["lead"], "index")["little"]
    assert gap <= float(_Q["max_pair_little_finger_gap_m"])


_SQUARE = (
    f"#15150: owner's targets, not met with the thumb covered: {_WAS['wrap_tilt']}"
)
_SQUARE += f"; knuckles {_WAS['knuckle_lateral_offset']} off the shaft line."


@_BUILT
@pytest.mark.xfail(strict=True, reason=_SQUARE)
def test_the_covered_seat_meets_the_owner_square_targets(
    measured: TrailGripMeasures,
) -> None:
    assert all(
        measured.wrap_tilt_deg[f] <= float(_T["max_wrap_tilt_deg"]) for f in _HOLDING
    )
    most = float(_T["max_knuckle_lateral_offset_m"])
    assert all(abs(measured.knuckle_lateral_m[f]) <= most for f in _HOLDING)
