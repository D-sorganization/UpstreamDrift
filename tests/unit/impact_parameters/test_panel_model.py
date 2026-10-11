"""Tests for the shared impact-parameters card model (GCV-17)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.impact_parameters import (
    ClubheadSeries,
    TargetFrame,
    extract_impact_parameters,
)
from src.shared.python.impact_parameters.panel_model import (
    build_impact_card,
    parse_target_dir,
    target_frame_from_heading,
)

pytestmark = pytest.mark.unit

N = 41


def _result(with_face=True):
    t = np.arange(N) * 0.002
    vel = np.tile([0.0, -40.0, -3.0], (N, 1))
    pos = np.cumsum(vel * 0.002, axis=0)
    kwargs = {}
    if with_face:
        n = np.array([0.0, -0.97, 0.22])
        kwargs = {
            "face_normal": np.tile(n, (N, 1)),
            "toe_axis": np.tile([1.0, 0.0, 0.0], (N, 1)),
            "grip_axis": np.tile([0.0, 0.22, 0.97], (N, 1)),
        }
    else:
        kwargs = {"face_unobservable_reason": "mocap club roll unobservable"}
    series = ClubheadSeries(t, pos, vel, **kwargs)
    return extract_impact_parameters(series, TargetFrame(), impact_index=30)


def _row(card, key):
    return next(r for r in card.rows if r.key == key)


def test_card_has_all_launch_monitor_rows_with_units():
    card = build_impact_card(_result(), units="mph")
    keys = [r.key for r in card.rows]
    for k in (
        "clubhead_speed",
        "attack_angle_deg",
        "club_path_deg",
        "face_angle_deg",
        "face_to_path_deg",
        "dynamic_loft_deg",
        "spin_loft_deg",
        "low_point_ahead_of_ball_m",
        "impact_location",
        "smash_factor",
    ):
        assert k in keys, k
    assert _row(card, "clubhead_speed").unit == "mph"
    assert _row(card, "clubhead_speed").value == pytest.approx(
        40.1 * 2.2369362920544, rel=0.01
    )
    assert _row(card, "attack_angle_deg").value == pytest.approx(
        -math.degrees(math.atan(3 / 40)), abs=1e-6
    )


def test_units_toggle_changes_only_speed():
    mph = build_impact_card(_result(), units="mph")
    mps = build_impact_card(_result(), units="m/s")
    assert _row(mps, "clubhead_speed").unit == "m/s"
    assert _row(mps, "clubhead_speed").value * 2.2369362920544 == pytest.approx(
        _row(mph, "clubhead_speed").value
    )
    assert _row(mps, "attack_angle_deg").value == _row(mph, "attack_angle_deg").value


def test_unavailable_rows_carry_reason_and_no_value():
    card = build_impact_card(_result(with_face=False))
    face = _row(card, "face_angle_deg")
    assert face.value is None and "unobservable" in face.reason
    assert _row(card, "smash_factor").value is None
    assert _row(card, "smash_factor").reason
    assert card.d_plane["face_angle_deg"] is None
    assert card.available is True


def test_card_to_dict_is_json_serialisable():
    import json

    payload = build_impact_card(_result()).to_dict()
    json.dumps(payload)
    assert payload["units"] == "mph" and payload["d_plane"]["club_path_deg"] is not None


def test_invalid_units_rejected():
    with pytest.raises(ValueError, match="units"):
        build_impact_card(_result(), units="kph")


def test_parse_target_dir_normalises_and_validates():
    np.testing.assert_allclose(parse_target_dir("0,-2"), [0, -1, 0])
    np.testing.assert_allclose(parse_target_dir("3,4,0"), [0.6, 0.8, 0])
    assert parse_target_dir(None) == (0.0, -1.0, 0.0)
    for bad in ("a,b", "0,0", "1", "1,2,3,4", "nan,1"):
        with pytest.raises(ValueError):
            parse_target_dir(bad)


def test_heading_rotates_default_target_about_up():
    frame = target_frame_from_heading(90.0)
    np.testing.assert_allclose(frame.x_t, [1.0, 0.0, 0.0], atol=1e-12)
    assert target_frame_from_heading(0.0).x_t.tolist() == [0.0, -1.0, 0.0]
    with pytest.raises(ValueError):
        target_frame_from_heading(float("nan"))
