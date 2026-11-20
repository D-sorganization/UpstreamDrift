"""The trail hand's grip at address, as the owner described it (epic #15150).

Owner, 2026-11-19: the trail wrist is arched more; the fingers wrap closer to
square to the shaft; the middle and ring knuckles lie on the shaft's line,
behind it from the target; the heel of the palm sits high on the lead thumb.
Every band and target is ``trail_grip`` in ``hand_visual_quality.json``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from src.shared.python.model_appearance.hand_backends.qa import (
    load_hand_visual_quality,
)
from src.shared.python.model_appearance.hand_backends.trail_grip_qa import (
    TrailGripMeasures,
    trail_grip_measures,
)
from tests.unit.model_appearance.hand_backends.test_skinned_human import _pair

pytestmark = pytest.mark.unit

_T = cast(dict[str, Any], load_hand_visual_quality()["trail_grip"])
_HOLDING = ("middle", "ring")


def _measured() -> TrailGripMeasures:
    pair = _pair()
    return trail_grip_measures(pair["lead"], pair["trail"])


def test_the_measures_need_a_lead_and_a_trail_hand() -> None:
    pair = _pair()
    with pytest.raises(TypeError):
        trail_grip_measures(pair["lead"], object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lead hand, then the trail"):
        trail_grip_measures(pair["trail"], pair["lead"])


_WRIST_NOT_MET = "#15150: owner's target, not met. Past about 22 degrees the wrist "
_WRIST_NOT_MET += "crease's relaxing moves finger skin more than the 2.3 nm that "
_WRIST_NOT_MET += "test_wrist_bend_turns_the_forearm_and_leaves_the_hand_alone allows."


@pytest.mark.xfail(strict=True, reason=_WRIST_NOT_MET)
def test_the_trail_wrist_is_arched_at_address() -> None:
    """Item 1: the drawn hand's wrist is extended inside the owner's band."""
    low, high = (float(v) for v in _T["wrist_extension_band_deg"])
    extension = _measured().wrist_extension_deg
    assert low <= extension <= high, f"trail wrist extended {extension:.1f} deg"


def test_the_measures_are_those_of_the_posed_bones() -> None:
    """The knuckles are behind the shaft and the fingers named are measured."""
    measured = _measured()
    assert set(measured.wrap_tilt_deg) == {"index", "middle", "ring", "little"}
    assert all(0.0 <= tilt <= 90.0 for tilt in measured.wrap_tilt_deg.values())
    assert 0.0 <= measured.knuckle_line_to_shaft_deg <= 90.0
    assert all(measured.knuckle_behind_m[f] > 0.0 for f in _HOLDING)
    assert measured.heel_to_lead_thumb_base_m > 0.0
    assert np.isfinite(measured.wrist_below_lead_thumb_base_m)


def test_the_trail_grip_is_no_further_from_the_reference_than_measured() -> None:
    """Ratchets on what is measured today: none of the five may get worse."""
    measured, most = _measured(), _T["measured_most"]
    arch = measured.wrist_extension_deg
    assert arch >= float(most["least_wrist_extension_deg"]) - 1e-6, f"{arch:.1f} deg"
    tilt = max(measured.wrap_tilt_deg[f] for f in _HOLDING)
    assert tilt <= float(most["wrap_tilt_deg"]), f"wrap tilt {tilt:.1f} deg"
    line = measured.knuckle_line_to_shaft_deg
    assert line <= float(most["knuckle_line_to_shaft_deg"]), f"{line:.1f} deg"
    off = max(abs(measured.knuckle_lateral_m[f]) for f in _HOLDING)
    assert off <= float(most["knuckle_lateral_offset_m"]), f"{off * 1e3:.1f} mm"
    heel = measured.heel_to_lead_thumb_base_m
    assert heel <= float(most["heel_to_lead_thumb_base_m"]), f"{heel * 1e3:.1f} mm"


_NOT_MET = "#15150: owner's target, not met. Squaring the trail fingers carries "
_NOT_MET += "the palm down the shaft off the base of the lead thumb "
_NOT_MET += "(max_thumb_channel_gap_m) while the ring finger stays beside the lead "
_NOT_MET += "forefinger; see _trail_grip in hand_visual_quality.json."


@pytest.mark.xfail(strict=True, reason=_NOT_MET)
def test_the_trail_fingers_wrap_near_square_to_the_shaft() -> None:
    """Item 2: the two holding fingers and the knuckle line."""
    measured = _measured()
    most = float(_T["max_wrap_tilt_deg"])
    for finger in _HOLDING:
        tilt = measured.wrap_tilt_deg[finger]
        assert tilt <= most, f"trail {finger} wraps {tilt:.1f} deg off square"
    line = measured.knuckle_line_to_shaft_deg
    assert line <= float(_T["max_knuckle_line_to_shaft_deg"]), f"{line:.1f} deg"


@pytest.mark.xfail(strict=True, reason=_NOT_MET)
def test_the_middle_and_ring_knuckles_lie_behind_the_shaft_on_its_line() -> None:
    """Item 3: seen along the target line both knuckles are on the shaft."""
    measured = _measured()
    most = float(_T["max_knuckle_lateral_offset_m"])
    for finger in _HOLDING:
        off = measured.knuckle_lateral_m[finger]
        assert abs(off) <= most, f"trail {finger} knuckle {off * 1e3:.1f} mm off"


@pytest.mark.xfail(strict=True, reason=_NOT_MET)
def test_the_heel_of_the_trail_palm_is_at_the_base_of_the_lead_thumb() -> None:
    """Item 4: the palm's heel reaches the lead thumb's base joint."""
    heel = _measured().heel_to_lead_thumb_base_m
    most = float(_T["max_heel_to_lead_thumb_base_m"])
    assert heel <= most, f"heel {heel * 1e3:.1f} mm from the lead thumb's base"
