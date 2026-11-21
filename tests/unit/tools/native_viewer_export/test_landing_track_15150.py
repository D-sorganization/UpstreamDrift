"""Ground track, drop line and carry label of the landing marker (#15150)."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from src.tools.native_viewer_export import flight_tracer as tr, landing_marker as lm
from src.tools.native_viewer_export.overlay2d import pinhole_for_view
from tests.unit.tools.native_viewer_export import test_flight_tracer as base

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

D, IMPACT_S, LOOKAT = base.DEFAULTS, base.IMPACT_S, base.LOOKAT
WIDE, SIZE = "down_the_line_flight", (1280, 720)
TRACK, DROP, TEXT = (10, 250, 10), (250, 10, 250), (10, 250, 250)
SOLID = {
    "ground_track": True,
    "ground_track_color_rgba": (*TRACK, 255),
    "ground_track_width_px": 3,
    "drop_line": True,
    "drop_line_color_rgba": (*DROP, 255),
    "drop_line_width_px": 3,
    "carry_label": True,
    "carry_label_color_rgba": (*TEXT, 255),
    "occlude_track": False,
}
OFF = dict.fromkeys(("ground_track", "drop_line", "carry_label"), False)
CAMERA = pinhole_for_view(WIDE, LOOKAT, None, math.radians(45.0), SIZE)


def _style(**changes):
    marker = replace(D.landing_marker, **{**SOLID, **changes})
    return replace(D, landing_marker=marker, inset=None)  # inset owns the carry


def _draw(style, t, depth=None, playback=None):
    flight = playback or base._playback(style)
    tile = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8)
    depth_m = None if depth is None else {WIDE: depth}
    return flight.draw({WIDE: tile}, t, LOOKAT, None, depth_m)[WIDE]


def _count(img, colour):
    return int((img == colour).all(axis=2).sum())


def _centroid(img, colour):
    rows, cols = np.nonzero((img == colour).all(axis=2))
    return cols.mean(), rows.mean()


def _px(point):
    return tr.project(CAMERA, np.array([point]), 0.05)[0][0]


def test_defaults_are_off_and_validated() -> None:
    marker = lm.LandingMarkerSettings()
    assert not (marker.ground_track or marker.drop_line or marker.carry_label)
    bad = [
        ({"ground_track": 1}, TypeError, "ground_track"),
        ({"drop_line": "yes"}, TypeError, "drop_line"),
        ({"ground_track_color_rgba": (1, 2, 3)}, TypeError, "ground_track_color"),
        ({"drop_line_color_rgba": (1, 2, 3, 999)}, ValueError, "0..255"),
        ({"ground_track_width_px": 0}, ValueError, "ground_track_width_px"),
        ({"carry_label_scale": 0.0}, ValueError, "carry_label_scale"),
        ({"carry_label_size_px": 0}, ValueError, "carry_label_size_px"),
        ({"carry_label_format": "Carry {nope}"}, ValueError, "carry_label_format"),
        ({"carry_label_unit": 5}, TypeError, "carry_label_unit"),
    ]
    for changes, error, match in bad:
        with pytest.raises(error, match=match):
            lm.LandingMarkerSettings(**changes)


def test_shipped_settings_turn_the_aids_on() -> None:
    marker = D.landing_marker
    assert marker.ground_track and marker.drop_line and marker.carry_label


def test_ground_track_projects_the_path_onto_the_ground() -> None:
    path = np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 9.0]])
    ground = lm.ground_track_points(path, 0.1)
    assert ground.shape == (2, 3) and (ground[:, 2] == 0.1).all()
    assert (ground[:, :2] == path[:, :2]).all() and path[1, 2] == 9.0
    with pytest.raises(ValueError, match="finite"):
        lm.ground_track_points(np.array([[0.0, np.nan, 0.0]]), 0.0)


def test_carry_text_uses_format_unit_and_scale() -> None:
    marker = lm.LandingMarkerSettings(carry_label=True)
    assert lm.carry_text(marker, 206.6) == "Carry 207 m"
    yards = replace(
        marker,
        carry_label_unit="yd",
        carry_label_scale=1.0936,
        carry_label_format="{carry:.1f} {unit} carry",
    )
    assert lm.carry_text(yards, 100.0) == "109.4 yd carry"
    with pytest.raises(ValueError, match="carry_m"):
        lm.carry_text(marker, -1.0)


def test_playback_keeps_the_flight_carry() -> None:
    assert base._playback().carry_m == 160.0
    flight = replace(base._flight(), carry_m=123.4)
    play = tr.playback_from_flight(flight, base.FRAME, IMPACT_S, base.BALL, D)
    assert play.carry_m == 123.4
    img = _draw(_style(), IMPACT_S + 99.0, playback=replace(play, settings=_style()))
    assert _count(img, TEXT) > 30  # "Carry 123 m" drawn from the stored number


def test_ground_track_is_revealed_in_step_with_the_tracer() -> None:
    style, end = _style(drop_line=False), base._playback().path_m[-1]
    spot = _px([end[0], end[1], 0.0])
    early = _draw(style, IMPACT_S + 1.0)
    late = _draw(style, IMPACT_S + 99.0)
    assert 0 < _count(early, TRACK) < _count(late, TRACK)
    quiet = replace(style, landing_marker=replace(style.landing_marker, **OFF))
    assert _count(_draw(quiet, IMPACT_S + 99.0), TRACK) == 0
    cols = np.nonzero((late == TRACK).all(axis=2).any(axis=0))[0]
    assert abs(cols.max() - spot[0]) < 120 and cols.min() < cols.max()


def test_drop_line_joins_the_ball_to_the_ground_only_in_flight() -> None:
    style = _style(ground_track=False, carry_label=False)
    mid = base._playback(style).flown(IMPACT_S + 2.0)
    ball = mid[-1]
    assert ball[2] > 5.0  # well above the ground
    top, foot = _px(ball), _px([ball[0], ball[1], 0.0])
    img = _draw(style, IMPACT_S + 2.0)
    cx, cy = _centroid(img, DROP)
    assert abs(cx - top[0]) < 4 and min(top[1], foot[1]) - 2 < cy < max(top[1], foot[1])
    assert _count(img, DROP) > 5
    assert _count(_draw(style, IMPACT_S + 99.0), DROP) == 0  # landed


def test_carry_label_appears_after_landing_beneath_the_ring() -> None:
    style = _style(ground_track=False, drop_line=False)
    assert _count(_draw(style, IMPACT_S + 2.0), TEXT) == 0
    late = _draw(style, IMPACT_S + 99.0)
    end = base._playback().path_m[-1]
    u, v = _px([end[0], end[1], 0.0])
    cx, cy = _centroid(late, TEXT)
    assert _count(late, TEXT) > 30 and abs(cx - u) < 80 and v < cy < v + 80
    quiet = replace(style, landing_marker=replace(style.landing_marker, **OFF))
    assert _count(_draw(quiet, IMPACT_S + 99.0), TEXT) == 0


def test_label_stays_inside_the_tile() -> None:
    tile = np.full((60, 80, 3), 40, np.uint8)
    marker = replace(D.landing_marker, **SOLID)
    style = replace(D, landing_marker=marker)
    camera = pinhole_for_view(WIDE, LOOKAT, None, math.radians(45.0), (80, 60))
    spot = np.array([LOOKAT[0] + 0.3, LOOKAT[1], 0.0])
    out = lm.paint_label(tile, camera, spot, style, 160.0)
    assert out.shape == tile.shape and _count(out, TEXT) > 0


def test_ground_track_is_hidden_by_nearer_scene_when_occluding() -> None:
    style = _style(drop_line=False, occlude_track=True)
    wall = np.full((SIZE[1], SIZE[0]), 0.5, np.float32)
    assert _count(_draw(style, IMPACT_S + 99.0, wall), TRACK) == 0
    assert _count(_draw(style, IMPACT_S + 99.0, wall * 1e4), TRACK) > 0


def test_invalid_carry_in_playback_is_rejected() -> None:
    with pytest.raises(ValueError, match="carry_m"):
        replace(base._playback(), carry_m=float("nan"))
