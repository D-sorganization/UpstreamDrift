"""Scene-depth occlusion and the landing marker of the shot tracer (#15150)."""

from __future__ import annotations

from dataclasses import replace
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.tools.native_viewer_export import (
    contact_frame,
    flight_depth as fd,
    flight_tracer as tr,
)
from src.tools.native_viewer_export.backends import _subprocess
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.ball import BallPlacement
from src.tools.native_viewer_export.core import export_swing
from src.tools.native_viewer_export.overlay2d import pinhole_for_view
from tests.unit.tools.native_viewer_export import test_flight_tracer as base

# flat ball: these tests count pixels of the exact ball colour (shading: its own file)
D = replace(base.DEFAULTS, ball_shading=None)
IMPACT_S, LOOKAT = base.IMPACT_S, base.LOOKAT
_playback, _tile = base._playback, base._tile

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SIZE, WIDE, M = (640, 360), "down_the_line_flight", D.landing_marker
RED, MARK = D.tracer_color_rgba[:3], M.color_rgba[:3]
CAM = pinhole_for_view("down_the_line", LOOKAT, None, math.radians(45.0), SIZE)
LINE = np.array([[LOOKAT[0] - 1, *LOOKAT[1:]], [LOOKAT[0] + 1, *LOOKAT[1:]]])


def _depth(value: float) -> np.ndarray:
    return np.full((SIZE[1], SIZE[0]), value, np.float32)


def _count(img: np.ndarray, colour=RED) -> int:
    return int((img == colour).all(axis=2).sum())


def test_occlusion_densifies_and_tests_the_depth_under_each_point() -> None:
    dense = fd.Occlusion(_depth(5.0), 0.05, 0.0, 10.0, 0.05).dense(CAM, LINE)
    pix, depth = tr.project(CAM, dense, 0.05)
    assert len(dense) > 20 and np.hypot(*np.diff(pix, axis=0).T).max() <= 10.5
    scene = _depth(float(depth[0]) + 1.0)  # a surface behind the line ...
    scene[:, SIZE[0] // 2 :] = float(depth[0]) - 1.0  # ... and one in front, right
    seen = fd.Occlusion(scene, 0.05, 0.0, 10.0, 0.05).visible(CAM, dense)[1]
    assert seen[pix[:, 0] < SIZE[0] // 2 - 1].all()
    assert not seen[pix[:, 0] > SIZE[0] // 2 + 1].any()


def test_tracer_and_ball_hidden_per_view_else_one_warning(caplog) -> None:
    tiles, t, flight = {"down_the_line": _tile()}, IMPACT_S + 2.0, _playback(D)
    plain = _playback(D).draw(tiles, t, LOOKAT, None)["down_the_line"]
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        flight.draw(tiles, t, LOOKAT, None, depth_m={})
        flight.draw(tiles, t, LOOKAT, None, depth_m={})
    assert sum("scene depth" in r.message for r in caplog.records) == 1
    hidden = {"down_the_line": _depth(0.5)}
    behind = flight.draw(tiles, t, LOOKAT, None, hidden)["down_the_line"]
    assert _count(plain) > 30 and _count(plain, D.ball_color_rgb) > 3
    assert _count(behind) == 0 and _count(behind, D.ball_color_rgb) == 0
    off = _playback(replace(D, occlude_by_scene=False))
    assert (off.draw(tiles, t, LOOKAT, None, hidden)["down_the_line"] == plain).all()


def test_settings_are_validated() -> None:
    with pytest.raises(TypeError, match="occlude_by_scene"):
        replace(D, occlude_by_scene=1)
    with pytest.raises(ValueError, match="when"):
        replace(M, when="never")


def test_worker_job_depth_reaches_the_exporter(tmp_path: Path) -> None:
    args = ("b", "q", [0, 1], ["face_on"], 4, 3, [0, 0, 0], None, str(tmp_path))
    job = WorkerJob(*args, depth_positions=[False, True])
    for pos in (0, 1):
        np.save(job.frame_path("face_on", pos), np.zeros((3, 4, 3), np.uint8))
    np.save(job.depth_path("face_on", 1), np.full((3, 4), 2.0, np.float32))
    first, second = list(_subprocess.read_frames(job))
    assert first.depth_m == {} and float(second.depth_m["face_on"][0, 0]) == 2.0


def test_depth_is_requested_only_from_the_impact_on() -> None:
    swing, off = (
        SimpleNamespace(time_s=lambda k: 0.1 * k),
        replace(D, occlude_by_scene=False),
    )

    def ask(flight, impact=0.5):
        settings = SimpleNamespace(ball_flight=flight, impact_time_s=impact)
        return fd.depth_request(swing, settings, [0, 5, 6, 10])

    assert ask(D) == [False, True, True, True] and ask(None) is None
    assert ask(off) is None


def _wide(flight, t):
    tile = np.full((720, 1280, 3), 40, np.uint8)
    return flight.draw({WIDE: tile}, t, LOOKAT, None)[WIDE]


def test_marker_sits_on_the_ground_at_the_landing_point() -> None:
    camera = pinhole_for_view(WIDE, LOOKAT, None, math.radians(45.0), (1280, 720))
    end = _playback().path_m[-1]
    spot = np.array([[end[0], end[1], M.ground_z_m]])
    (u, v), _ = (a[0] for a in tr.project(camera, spot, 0.05))
    rows, cols = np.nonzero((_wide(_playback(), IMPACT_S + 99.0) == MARK).all(axis=2))
    assert rows.size > 20 and abs(cols.mean() - u) < 9 and abs(rows.mean() - v) < 9


def test_marker_timing_follows_the_settings() -> None:
    mid, late = IMPACT_S + 2.0, IMPACT_S + 99.0
    assert _count(_wide(_playback(), mid), MARK) == 0  # not landed yet

    def marked(**changes):
        return _playback(replace(D, landing_marker=replace(M, **changes)))

    assert _count(_wide(marked(when="from_start"), mid), MARK) > 20
    off = replace(D, landing_marker=None)
    assert _count(_wide(_playback(off), late), MARK) == 0


def _refine(swing, impact_s):
    return replace(swing, ball=replace(swing.ball, hide_after_s=0.123)), 0.123


def test_export_result_records_the_hide_time_actually_used(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(contact_frame, "refine", _refine)
    ball = BallPlacement((1.0, 0.0, 0.03), 0.021335, "test", 0.05)
    swing = replace(base._swing(), ball=ball)
    settings = replace(base._settings(impact_time_s=0.1), ball=True)
    factory = lambda path, fps: base._Writer([])  # noqa: E731
    done = export_swing(
        base._Backend(), swing, settings, tmp_path, writer_factory=factory
    )
    assert done.ball is not None and done.ball["hide_after_s"] == 0.123


def _ball_scene(near_side: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Tile, depth with the club nearer on one side of the ball, and the ball's u."""
    spot = np.array([LOOKAT, LOOKAT], dtype=float)
    pix, depth = tr.project(CAM, spot, 0.05)
    u = int(round(pix[0, 0]))
    scene = _depth(float(depth[0]) + 5.0)  # open ground far behind the ball
    if near_side == "left":
        scene[:, :u] = float(depth[0]) - 1.0
    else:
        scene[:, u:] = float(depth[0]) - 1.0
    return spot, scene, float(u)


def test_ball_marker_is_clipped_where_the_club_is_nearer() -> None:
    spot, scene, u = _ball_scene("left")
    radius = 0.1  # large enough to span many pixels
    out = tr.draw_trace(_tile(), CAM, spot, radius, D, scene)
    cols = np.nonzero((out == D.ball_color_rgb).all(axis=2).any(axis=0))[0]
    assert cols.size > 5 and cols.min() >= u  # only the far-side half is painted
    whole = tr.draw_trace(_tile(), CAM, spot, radius, D, _depth(1e3))
    wide = np.nonzero((whole == D.ball_color_rgb).all(axis=2).any(axis=0))[0]
    assert wide.min() < u - 3 and wide.size > 1.6 * cols.size


def test_ball_marker_clipping_follows_the_setting() -> None:
    spot, scene, _ = _ball_scene("right")
    off = replace(D, occlude_by_scene=False)
    free = tr.draw_trace(_tile(), CAM, spot, 0.1, off, scene)
    clipped = tr.draw_trace(_tile(), CAM, spot, 0.1, D, scene)
    assert _count(clipped, D.ball_color_rgb) < _count(free, D.ball_color_rgb)
    assert _count(clipped, D.ball_color_rgb) > 0
