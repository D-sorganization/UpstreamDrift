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
    skinning_residual_m,
    thumb_crown_gaps_m,
    thumb_wrap_shares,
)
from src.shared.python.model_appearance.hand_backends.skinned_pair import (
    build_grip_hands,
)
from src.shared.python.model_appearance.hand_backends.skinned_settings import (
    default_settings,
)
from tests.unit.model_appearance.hand_backends.test_skinned_human import (
    _RADIUS,
    _pair,
    _spec,
)

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


def _wrap(lead, trail) -> np.ndarray:
    return thumb_wrap_shares(
        lead, trail, station_m=_STATION, contact_m=float(_Q["thumb_wrap_contact_m"])
    )


def _trial_settings():
    """The trail palm pads folded round the thumb, on the palm without its
    fold joint: the trial pad values were measured on that palm (#15265), and
    on the folded one they push the hands 1.15 mm into each other."""
    settings = default_settings()
    wide = dataclasses.replace(settings.palm_pad, **_Q["thumb_wrap_trial_pad"])
    flat = dataclasses.replace(settings.pair_palm_fold, max_deg=0.0)
    return dataclasses.replace(settings, pair_palm_wrap=wide, pair_palm_fold=flat)


@pytest.mark.xfail(
    strict=True,
    reason="#15265, #15270: with the palm fold joint the thenar side closes on "
    "the thumb (stations under the middle of the palm 0.5 to 0.65) but the "
    "median share is 0.20: the base and the tip of the thumb stay uncovered "
    "on their flanks, and so does the flank on the little-finger side.",
)
def test_the_trail_palm_wraps_round_the_lead_thumb() -> None:
    """Owner ("cup the palm"): the palm folds down both flanks of the thumb,
    it does not only touch the thumb's top line."""
    pair = _pair()
    shares = _wrap(pair["lead"], pair["trail"])
    assert len(shares) >= 8, "the thumb is cut into too few stations to judge"
    assert float(np.median(shares)) >= float(_Q["min_thumb_wrap_share"]), (
        f"wrap share along the thumb: {np.round(shares, 2)}"
    )


def test_the_palm_fold_joint_closes_the_palm_on_the_thumb() -> None:
    """#15270: the fold joint brings more of the thumb into contact than the
    same grip without it, without the hands overlapping and with the skin
    still on its bones."""
    settings = default_settings()
    fold = settings.pair_palm_fold
    assert fold.max_deg > 0.0, "the palm fold joint is switched off"
    flat = dataclasses.replace(
        settings, pair_palm_fold=dataclasses.replace(fold, max_deg=0.0)
    )
    without = build_grip_hands(_spec(), grip_radius_m=_RADIUS, settings=flat)
    pair = _pair()
    gain = float(
        _wrap(pair["lead"], pair["trail"]).mean()
        - _wrap(without["lead"], without["trail"]).mean()
    )
    assert gain >= float(_Q["min_palm_fold_wrap_gain"]), f"wrap gain {gain:.3f}"
    # The fold is in the rest hand, so the bones carry it: the skin is moved
    # off them no further than the correctives may move it.
    assert not np.array_equal(
        pair["trail"].rest_asset.rest_vertices,
        without["trail"].rest_asset.rest_vertices,
    )
    assert skinning_residual_m(pair["trail"]) <= float(_Q["max_corrective_shift_m"])
    depth = hand_pair_penetration_m(pair["lead"], pair["trail"])
    assert depth <= float(_Q["pair_penetration_tolerance_m"])


def test_the_palm_pads_can_be_set_to_fold_round_the_thumb() -> None:
    """``pair.palm_wrap`` is the lever: with the trial pad values the palm
    reaches the wrap the owner asks for, without the hands overlapping. It is
    not the default because the skin then leaves its bones too far."""
    settings = default_settings()
    assert settings.palm_pad_over(False) == settings.palm_pad
    assert settings.palm_pad_over(True) == settings.palm_pad, "the default is off"
    trial = _trial_settings()
    assert trial.palm_pad_over(False) == settings.palm_pad
    pair = build_grip_hands(_spec(), grip_radius_m=_RADIUS, settings=trial)
    shares, plain = (
        _wrap(pair["lead"], pair["trail"]),
        _wrap(_pair()["lead"], _pair()["trail"]),
    )
    assert float(np.median(shares)) >= float(_Q["min_thumb_wrap_share"]), shares
    assert np.median(shares) > np.median(plain) + 0.2
    depth = hand_pair_penetration_m(pair["lead"], pair["trail"])
    assert depth <= float(_Q["pair_penetration_tolerance_m"])
    assert skinning_residual_m(pair["trail"]) > float(_Q["max_corrective_shift_m"])


def test_the_wrap_measure_refuses_bad_settings() -> None:
    pair = _pair()
    for station, contact in ((0.0, 0.001), (0.005, 0.0)):
        with pytest.raises(ValueError, match="contact_m"):
            thumb_wrap_shares(
                pair["lead"], pair["trail"], station_m=station, contact_m=contact
            )
