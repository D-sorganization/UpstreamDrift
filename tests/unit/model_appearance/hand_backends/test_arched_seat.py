"""The ``arched_square`` seat of the trail hand (#15329, epic #15150).

``square_on_thumb`` with the club laid squarer still and the middle, ring and
little metacarpals leaned toward the little finger (``metacarpal_arch``), so
the fingers' curl planes turn with their rays. Its own bounds, and what it
measured, are ``trail_grip.arched_square`` in ``hand_visual_quality.json``;
the owner's targets it does not meet are strict expected failures here.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance.hand_backends.qa import (
    hand_pair_penetration_m,
    hand_to_finger_gaps_m,
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
_A = cast(dict[str, Any], _T["arched_square"])
_WAS = _A["measured"]
_SEAT = "arched_square"
_HOLDING = ("middle", "ring")


def test_the_arched_seat_sets_the_trail_arch_and_nothing_of_the_lead_hand() -> None:
    seat = sch.trail_seat(_SEAT)
    assert _Q["owner_trail_seat"] != _SEAT, "the default seat is not changed"
    rolled = sch.trail_seat("square_on_thumb")
    assert seat.club_cross_angle_deg < rolled.club_cross_angle_deg
    arch = dict(seat.metacarpal_arch)
    assert set(arch) == {"middle", "ring", "little"}
    assert all(flexion == 0.0 and lean > 0.0 for flexion, lean in arch.values())
    assert arch["middle"][1] < arch["ring"][1] < arch["little"][1]
    settings = sch.seat_settings(seat)
    assert settings.metacarpal_arch.of("trail") == arch
    assert settings.metacarpal_arch.of("lead") == {}
    assert (
        dict(settings.wrist_deg)["lead"] == dict(default_settings().wrist_deg)["lead"]
    )
    for other in ("palm_on_thumb", "square_wrap", "square_on_thumb"):
        assert sch.trail_seat(other).metacarpal_arch == ()
    assert sch.seat_settings(sch.trail_seat()) is default_settings()


def test_a_seat_whose_arch_the_joints_cannot_take_is_refused() -> None:
    seat = sch.trail_seat(_SEAT)
    bent = sch.TrailSeat(
        **{**seat.__dict__, "metacarpal_arch": (("ring", (0.0, 45.0)),)}
    )
    with pytest.raises(ValueError, match="metacarpal_arch"):
        sch.seat_settings(bent)


@pytest.fixture(scope="module")
def arched() -> dict[str, HandVisualResult]:
    return sch.build_seated_pair(_SEAT)


@pytest.fixture(scope="module")
def measured(arched: dict[str, HandVisualResult]) -> TrailGripMeasures:
    return trail_grip_measures(arched["lead"], arched["trail"])


def _crown_gap_m(pair: dict[str, HandVisualResult]) -> float:
    gaps = thumb_crown_gaps_m(
        pair["lead"],
        pair["trail"],
        station_m=float(_Q["thumb_channel_station_m"]),
        min_facing=float(_Q["thumb_channel_crown_facing"]),
        min_station_vertices=int(_Q["thumb_channel_min_station_vertices"]),
    )
    return float(max(gaps))


def test_the_arched_seat_wraps_the_holding_fingers_to_the_owner_bound(
    measured: TrailGripMeasures,
) -> None:
    """Met: the owner's bound on the curl planes of the middle and ring
    fingers, and on the knuckle line."""
    most = float(_T["max_wrap_tilt_deg"])
    for finger in _HOLDING:
        tilt = measured.wrap_tilt_deg[finger]
        assert tilt <= most, f"{finger} curls {tilt:.1f} deg off square"
    index = measured.wrap_tilt_deg["index"]
    assert index <= float(_A["max_index_wrap_tilt_deg"]), f"index {index:.1f} deg"
    line = measured.knuckle_line_to_shaft_deg
    assert line <= float(_T["max_knuckle_line_to_shaft_deg"]), f"{line:.1f} deg"


def test_the_arched_seat_has_the_knuckles_on_the_shaft_line(
    measured: TrailGripMeasures,
) -> None:
    """Met: the owner's 5 mm, for both holding fingers."""
    most = float(_T["max_knuckle_lateral_offset_m"])
    for finger in _HOLDING:
        off = measured.knuckle_lateral_m[finger]
        assert abs(off) <= most, f"trail {finger} knuckle {off * 1e3:.1f} mm off"
        assert measured.knuckle_behind_m[finger] > 0.0


def test_the_arched_seat_heel_is_at_the_lead_thumb_knuckle(
    measured: TrailGripMeasures,
) -> None:
    heel = measured.heel_to_lead_thumb_knuckle_m
    assert heel <= float(_T["max_heel_to_lead_thumb_knuckle_m"]), f"{heel * 1e3:.1f}"


def test_the_arched_seat_keeps_the_hands_together_and_out_of_each_other(
    arched: dict[str, HandVisualResult], measured: TrailGripMeasures
) -> None:
    """The standing bounds: no hand in the other, the ring finger beside the
    lead forefinger, the little finger on it, the wrist arch kept."""
    deep = hand_pair_penetration_m(arched["lead"], arched["trail"])
    assert deep <= float(_Q["pair_penetration_tolerance_m"]), f"{deep * 1e3:.2f} mm"
    gaps = hand_to_finger_gaps_m(arched["trail"], arched["lead"], "index")
    assert gaps["ring"] <= float(_Q["max_pair_ring_finger_gap_m"])
    assert gaps["little"] <= float(_Q["max_pair_little_finger_gap_m"])
    least = float(_T["measured_most"]["least_wrist_extension_deg"])
    assert measured.wrist_extension_deg >= least - 1e-6
    gap = _crown_gap_m(arched)
    assert gap <= float(_A["max_thumb_crown_gap_m"]), f"{gap * 1e3:.1f} mm"


_RING = f"#15329: not met by arched_square: {_WAS['wrap_tilt']}; the issue asks "
_RING += "15 degrees of both holding fingers."


@pytest.mark.xfail(strict=True, reason=_RING)
def test_the_arched_seat_wraps_within_the_issue_fifteen_degrees(
    measured: TrailGripMeasures,
) -> None:
    most = float(_A["issue_max_wrap_tilt_deg"])
    assert all(measured.wrap_tilt_deg[f] <= most for f in _HOLDING)


_COVER = "#15329: given up for squarer fingers: arched_square leaves the tip "
_COVER += f"end of the lead thumb bare by {_WAS['thumb_crown_gap']}."


@pytest.mark.xfail(strict=True, reason=_COVER)
def test_the_arched_seat_palm_covers_the_whole_lead_thumb(
    arched: dict[str, HandVisualResult],
) -> None:
    assert _crown_gap_m(arched) <= float(_Q["max_thumb_channel_gap_m"])


_FORE = f"#15329: not met by arched_square: forefinger {_WAS['index_wrap_tilt']}."


@pytest.mark.xfail(strict=True, reason=_FORE)
def test_the_arched_seat_forefinger_wraps_to_the_owner_bound(
    measured: TrailGripMeasures,
) -> None:
    assert measured.wrap_tilt_deg["index"] <= float(_T["max_index_wrap_tilt_deg"])


_ARCH = "#15150: owner's band, not met by any seat: the drawn wrist is arched "
_ARCH += f"{_WAS['wrist_extension']}."


@pytest.mark.xfail(strict=True, reason=_ARCH)
def test_the_arched_seat_wrist_is_arched_into_the_owner_band(
    measured: TrailGripMeasures,
) -> None:
    low, high = _T["wrist_extension_band_deg"]
    assert low <= measured.wrist_extension_deg <= high
