"""Flight inset of native flight clips: side profile and plan view (#15150)."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.native_viewer_export import (
    compositor,
    ellipsoid_panel as ep,
    flight_inset as fi,
    flight_tracer as tr,
)
from src.tools.native_viewer_export.core import ClipOverlays
from tests.unit.tools.native_viewer_export import test_flight_tracer as base

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SHAPE = (720, 1280, 3)
CURVE, TEXT = (10, 250, 10), (250, 10, 250)
ON = fi.load_inset_settings()
LOUD = replace(
    ON,
    enabled=True,
    opacity=1.0,
    curve_color_rgba=(*CURVE, 255),
    text_color_rgb=TEXT,
    dot_color_rgba=(*CURVE, 255),
    marker_color_rgba=(*CURVE, 255),
)
OTHER = replace(base.DEFAULTS, inset=LOUD)


def _count(img: np.ndarray, colour: tuple[int, ...]) -> int:
    return int((img == np.array(colour, np.uint8)).all(axis=2).sum())


def _frame() -> np.ndarray:
    return np.full(SHAPE, 60, np.uint8)


def _rect(settings=LOUD, obstacles=(), top=40, limit=640):
    return fi.layout_rect(SHAPE, settings, list(obstacles), top, limit)


def _playback(settings=OTHER):
    return base._playback(settings)


def _inside(inner, outer) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _clear(a, b) -> bool:
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_settings_are_validated_and_documented(tmp_path: Path) -> None:
    assert not fi.FlightInsetSettings().enabled  # off by default in code
    raw = json.loads(fi.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}
    for bad, error in (
        ({"position": "middle"}, ValueError),
        ({"width_fraction": 0.0}, ValueError),
        ({"min_height_fraction": 0.9}, ValueError),
        ({"opacity": 1.5}, ValueError),
        ({"vertical_exaggeration": 0.5}, ValueError),
        ({"curve_width_px": 0}, ValueError),
        ({"curve_color_rgba": (1, 2, 3)}, TypeError),
        ({"curve_color_rgba": (1, 2, 3, 400)}, ValueError),
        ({"text_color_rgb": (1, 2)}, TypeError),
        ({"enabled": "yes"}, TypeError),
        ({"carry_label_format": "{nope}"}, ValueError),
        ({"apex_label_format": "{apex:.0f"}, ValueError),
        ({"side_left_text": ""}, ValueError),
    ):
        with pytest.raises(error):
            replace(ON, **bad)
    del raw["unit"]
    short = tmp_path / "inset.json"
    short.write_text(json.dumps(raw), encoding="utf-8")
    assert fi.load_inset_settings(short).unit == "m"  # defaults fill a missing key
    short.write_text(json.dumps({**raw, "bogus": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        fi.load_inset_settings(short)


def test_tracer_settings_carry_the_inset_file() -> None:
    assert tr.load_tracer_settings().inset == ON


def test_layout_is_inside_frame_and_clear_of_view_label_and_hud() -> None:
    for position in fi.POSITIONS:
        rect = _rect(replace(LOUD, position=position))
        assert rect is not None
        assert _inside(rect, (0, 40, SHAPE[1], 640))


def test_layout_stacks_below_a_panel_on_a_shared_edge() -> None:
    panel = (1000, 40, 1272, 260)
    rect = _rect(obstacles=[panel])
    assert rect is not None and _clear(rect, panel)
    assert rect[1] == panel[3] + LOUD.panel_gap_px
    bottom = _rect(replace(LOUD, position="bottom_right"), [(1000, 400, 1272, 500)])
    assert bottom is not None and bottom[3] == 400 - LOUD.panel_gap_px


def test_layout_ignores_a_panel_on_the_other_edge_and_shrinks_or_hides() -> None:
    far = (8, 40, 300, 600)
    assert _rect(obstacles=[far])[1] == 40  # type: ignore[index]
    tight = _rect(obstacles=[(1000, 40, 1272, 450)])
    assert tight is not None and tight[3] - tight[1] < 0.4 * SHAPE[0]
    assert _rect(obstacles=[(1000, 40, 1272, 600)]) is None


def test_default_place_is_free_with_every_panel_on() -> None:
    """Impact factors top right, ellipsoid panel opposite, HUD rows below."""

    @dataclass
    class Cfg:
        position: str = "right"

    class Factors:
        settings = Cfg()

        def bbox(self, shape):
            return compositor.text_panel_box(
                shape, "Impact Factors", [("Row", "0.00", False)] * 12, "right"
            )[0]

    panel = ep.EllipsoidPanel(
        ep.load_panel_settings(), ("mobility", "effective-mass"), True, Factors()
    )
    rows = tuple(ep.PanelRow(f"point_{i}", 1.0, 2.0, 3.0) for i in range(4))

    class Feed:
        def hand_panel_rows(self, index):
            return rows

    hud = compositor.HudInfo(3.0, "MuJoCo", "Driver", "", extra_lines=("a", "b", "c"))
    off = replace(base.DEFAULTS, inset=replace(ON, enabled=False))
    overlays = ClipOverlays(
        flight=_playback(off),
        factors=Factors(),  # type: ignore[arg-type]
        ellipsoids=panel,
    )
    limit = ep.hud_top_px(SHAPE, hud)
    boxes = fi.obstacles_for(overlays, Feed(), 0, SHAPE, limit)
    assert len(boxes) == 2
    rect = fi.layout_rect(SHAPE, ON, boxes, 40, limit)
    assert rect is not None and _inside(rect, (0, 0, SHAPE[1], limit))
    assert all(_clear(rect, b) for b in boxes)
    frame = _frame()
    out = fi.draw_all(overlays, Feed(), 0, hud, {"down_the_line_flight": frame})
    assert out["down_the_line_flight"] is frame  # disabled by default: untouched
    on = replace(overlays, flight=_playback())
    shown = fi.draw_all(on, Feed(), 0, hud, {"down_the_line_flight": frame})
    changed = np.argwhere((shown["down_the_line_flight"] != frame).any(axis=2))
    assert changed.size
    y0, x0 = changed.min(axis=0)
    y1, x1 = changed.max(axis=0) + 1
    assert _inside((x0, y0, x1, y1), rect)


def test_flight_coordinates_match_the_model_frame() -> None:
    play = _playback()
    local = fi.flight_coordinates(play.path_m, play.axes_m)
    assert np.allclose(local, base._flight().trajectory_m)
    assert play.lateral_m == 8.0 and local[-1, 1] < 0.0  # right of the line: -left
    with pytest.raises(ValueError, match="axes_m"):
        fi.flight_coordinates(play.path_m, np.eye(2))


def test_curve_stays_inside_its_plots_for_any_exaggeration() -> None:
    rect = _rect()
    play = _playback()
    local = fi.flight_coordinates(play.path_m, play.axes_m)
    extent = (local[-1, 0], play.apex_m, np.abs(local[:, 1]).max())
    for exag in (None, 1.0, 6.0, 500.0):
        cfg = replace(LOUD, vertical_exaggeration=exag)
        profile, plan = fi.plots_for(rect, cfg, 16, extent)
        for plot, vertical in ((profile, local[:, 2]), (plan, local[:, 1])):
            pix = plot.pixels(local[:, 0], vertical)
            x0, y0, x1, y1 = plot.box
            assert (pix[:, 0] >= x0).all() and (pix[:, 0] <= x1).all()
            assert (pix[:, 1] >= y0).all() and (pix[:, 1] <= y1).all()
    true = fi.plots_for(rect, replace(LOUD, vertical_exaggeration=1.0), 16, extent)
    assert true[0].scale_y == pytest.approx(true[0].scale_x)
    tall = fi.plots_for(rect, replace(LOUD, vertical_exaggeration=1.5), 16, extent)
    assert tall[0].scale_y == pytest.approx(1.5 * tall[0].scale_x)


def test_reveal_follows_the_tracer_and_labels_persist() -> None:
    play, rect = _playback(), _rect()
    t0 = base.IMPACT_S
    frame = _frame()
    early = fi.paint(frame, play, t0 + 0.5, rect, LOUD)
    mid = fi.paint(frame, play, t0 + 3.0, rect, LOUD)
    landed = fi.paint(frame, play, t0 + 4.0, rect, LOUD)
    later = fi.paint(frame, play, t0 + 6.0, rect, LOUD)
    assert _count(early, CURVE) < _count(mid, CURVE) < _count(landed, CURVE)
    assert _count(early, TEXT) == 0  # the apex is not reached yet
    assert _count(landed, TEXT) > _count(mid, TEXT) > 0
    assert np.array_equal(landed, later)  # full curve and labels persist


def test_before_launch_axes_are_empty_or_hidden_and_only_the_rect_changes() -> None:
    play, rect = _playback(), _rect()
    frame = _frame()
    empty = fi.paint(frame, play, 0.2, rect, LOUD)
    assert _count(empty, CURVE) == 0 and not np.array_equal(empty, frame)
    hidden = fi.paint(frame, play, 0.2, rect, replace(LOUD, hide_before_launch=True))
    assert np.array_equal(hidden, frame)
    landed = fi.paint(frame, play, 9.0, rect, LOUD)
    x0, y0, x1, y1 = rect
    outside = np.ones(SHAPE[:2], bool)
    outside[y0:y1, x0:x1] = False
    assert np.array_equal(landed[outside], frame[outside])
    with pytest.raises(ValueError, match="outside the frame"):
        fi.paint(frame, play, 9.0, (0, 0, 2000, 10), LOUD)


def test_labels_use_the_model_numbers_and_units() -> None:
    cfg = replace(LOUD, unit="yd", unit_scale=1.0936)
    assert fi._fmt(cfg, "carry_label_format", carry=160.0 * cfg.unit_scale) == (
        "Carry 175 yd"
    )
    assert fi._fmt(cfg, "side_label_format", side=8.0, direction="R") == "8 yd R"
    play = _playback()
    assert fi.inset_data(play, 9.0).lateral_m == 8.0  # type: ignore[union-attr]
    none = tr.FlightPlayback(OTHER, "no ball placed")
    assert fi.inset_data(none, 1.0) is None
    frame = _frame()
    assert fi.paint(frame, none, 1.0, _rect(), LOUD) is frame
