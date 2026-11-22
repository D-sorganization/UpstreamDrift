"""Aim line of shot-shape views: ball to finish versus the address line (#15443)."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from src.tools.native_viewer_export import (
    aim_line,
    cli,
    flight_inset as fi,
    flight_tracer as tr,
)
from src.tools.native_viewer_export.rollout import Rollout
from tests.unit.tools.native_viewer_export import test_flight_tracer as base

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

AIM = aim_line.load_aim_settings()
BUNDLE = ["--bundle", "b.npz", "--swing", "s", "--out", "o"]


def _settings(mode: str) -> tr.TracerSettings:
    return replace(base.DEFAULTS, aim=replace(AIM, aim_line=mode))


def _aimed(mode: str) -> tr.FlightPlayback:
    play = base._playback(_settings(mode))
    return tr.with_aim(play, base._flight())


def test_settings_file_is_documented_and_validated() -> None:
    raw = json.loads(aim_line.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}
    assert AIM.aim_line == "auto"
    with pytest.raises(ValueError):
        replace(AIM, aim_line="middle")
    with pytest.raises(ValueError):
        replace(AIM, line_width_px=0)


def test_default_is_finish_with_a_shot_shape_and_unchanged_without() -> None:
    assert AIM.resolved("draw", None) == "finish"
    assert AIM.resolved(None, None) == "auto"
    assert AIM.resolved("draw", "address") == "address"
    assert AIM.resolved(None, "finish") == "finish"
    with pytest.raises(ValueError):
        AIM.resolved(None, "sideways")


def test_cli_flag_overrides_the_default() -> None:
    args = [*BUNDLE, "--ball-flight", "--shot-shape", "draw"]
    parser = cli.build_parser()
    tracer = cli._ball_flight_for(parser.parse_args(args), base.DEFAULTS.display_units)
    assert tracer.aim.aim_line == "finish"
    parsed = parser.parse_args([*args, "--aim-line", "address"])
    tracer = cli._ball_flight_for(parsed, base.DEFAULTS.display_units)
    assert tracer.aim.aim_line == "address"
    parsed = parser.parse_args([*BUNDLE, "--ball-flight"])
    assert (
        cli._ball_flight_for(parsed, base.DEFAULTS.display_units).aim.aim_line == "auto"
    )


def test_auto_leaves_the_playback_untouched() -> None:
    play = base._playback(_settings("auto"))
    assert tr.with_aim(play, base._flight()) is play
    assert play.aim_yaw_rad is None and play.aim_text is None


def test_finish_aim_turns_the_plan_view_onto_the_finish_point() -> None:
    play = _aimed("finish")
    data = fi.inset_data(play, 1e9)
    assert play.aim_text == AIM.finish_text
    assert data.lateral_m == pytest.approx(0.0, abs=1e-9)
    assert data.full[-1, 1] == pytest.approx(0.0, abs=1e-9)
    assert data.full[-1, 0] == pytest.approx(math.hypot(160.0, 8.0))  # tee to landing


def test_address_aim_keeps_the_target_line_and_names_it() -> None:
    play = _aimed("address")
    data = fi.inset_data(play, 1e9)
    assert play.aim_text == AIM.address_text
    assert data.lateral_m == pytest.approx(8.0)
    assert data.full[-1, 1] == pytest.approx(-8.0)


def test_rest_point_defines_the_finish_line_when_a_roll_is_modelled() -> None:
    play = base._playback(_settings("finish"))
    roll = Rollout(20.0, 180.0, (19.0, 6.0), "driver", "fairway", "flat", "x")
    play = tr.with_aim(replace(play, rollout=roll), base._flight())
    data = fi.inset_data(play, 1e9)
    assert data.roll_end[1] == pytest.approx(0.0, abs=1e-9)
    assert data.lateral_m != 0.0  # landing is off the rest-point line


def test_hud_states_side_against_the_aim_line_and_names_it() -> None:
    finish, address = _aimed("finish"), _aimed("address")
    assert finish.lines[-1] == AIM.finish_text
    assert "Side 0 yd" in finish.lines[1]
    assert address.lines[-1] == AIM.address_text
    assert "Side 9 yd" in address.lines[1] and "right" in address.lines[1]


def test_plan_title_names_the_line() -> None:
    play = _aimed("finish")
    data = fi.inset_data(play, 1e9)
    plots = fi.plots_for(
        (0, 0, 400, 300), fi.load_inset_settings(), 12, (200, 20, 20), 1
    )
    titles, _ = fi.label_specs(plots, data, fi.load_inset_settings(), 12)
    assert AIM.finish_text in titles[1].text


def test_wide_view_draws_the_aim_line_along_the_finish() -> None:
    play = _aimed("finish")
    pts = tr._aim_line_points(play)
    path = np.asarray(play.path_m)
    ball, end = path[0, :2], path[-1, :2]
    along = pts[1, :2] - pts[0, :2]
    cross = along[0] * (end - ball)[1] - along[1] * (end - ball)[0]
    assert cross == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(along) > np.linalg.norm(end - ball)
    assert tr._aim_line_points(base._playback(_settings("auto"))) is None
    address = tr._aim_line_points(_aimed("address"))
    target = np.asarray(play.axes_m)[0, :2]
    along = address[1, :2] - address[0, :2]
    assert along[0] * target[1] - along[1] * target[0] == pytest.approx(0.0, abs=1e-6)


def test_aim_line_changes_no_flight_number() -> None:
    play, aimed = base._playback(_settings("auto")), _aimed("finish")
    assert aimed.carry_m == play.carry_m and aimed.lateral_m == play.lateral_m
    assert np.array_equal(aimed.path_m, play.path_m)


def test_width_scales_with_resolution_and_zero_side_has_no_direction() -> None:
    assert AIM.width_for(544) == 3
    assert AIM.width_for(1080) > AIM.width_for(544)
    with pytest.raises(ValueError):
        AIM.width_for(0)
    assert aim_line.side_text_args(1e-12) == (0.0, "")
    assert aim_line.side_text_args(-2.0) == (2.0, "left")
    assert "Side 0 yd   Time" in _aimed("finish").lines[1]
