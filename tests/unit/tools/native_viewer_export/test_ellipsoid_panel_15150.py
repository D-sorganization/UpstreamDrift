"""Compact side panel for the export ellipsoids, replacing the HUD rows (#15150)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.tools.native_viewer_export import compositor, ellipsoid_panel as ep
from src.tools.native_viewer_export.core import ExportSettings, OverlayFeed
from src.tools.native_viewer_export.impact_factors import (
    load_factor_settings,
    no_impact_panel,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SHAPE, SMALL, BOTH = (720, 1280, 3), (360, 640, 3), ("mobility", "effective-mass")
NAMES = ("lead_hand", "trail_hand", "lead_foot", "trail_foot", "clubhead")
ROWS = tuple(ep.PanelRow(n, 0.5 + i, 2.0 + i, 3.0 + i) for i, n in enumerate(NAMES))
SETTINGS = ep.load_panel_settings()
LIMIT = 720 - (4 * 22 + 6)  # four bottom HUD rows at 1280x720


def _factors(position: str = "right"):
    return no_impact_panel(replace(load_factor_settings(), position=position))


def _panel(by_kind=True, kinds=BOTH, factors="right", **changes):
    cfg, other = replace(SETTINGS, **changes), factors and _factors(factors)
    return ep.EllipsoidPanel(cfg, kinds, by_kind, other or None)


def test_settings_and_rows_are_validated() -> None:
    with pytest.raises(ValueError, match="layout"):
        replace(SETTINGS, layout="grid")
    with pytest.raises(ValueError, match="finite"):
        ep.PanelRow("x", float("nan"), 1.0, 2.0)


def test_table_has_a_row_per_point_a_column_per_kind_and_a_legend() -> None:
    lines = _panel().lines(ROWS)
    assert lines[0][1] == ("Mobility", "Eff. Mass", "Cond.")
    assert lines[1][0] == "Units" and "kg^1/2" in lines[1][1]
    assert lines[2][:2] == ("Lead Hand", ("0.50", "2.00", "3.0"))
    assert [ln[0] for ln in lines[2:7]][-2:] == ["Trail Foot", "Clubhead"]
    assert [ln[0] for ln in lines[7:]] == ["Mobility Ellipsoid", "Force Ellipsoid"]
    assert len(_panel(by_kind=False).lines(ROWS)) == 7
    assert _panel(kinds=("mobility",)).lines(ROWS)[2][1] == ("0.50", "3.0")


def test_panel_sits_opposite_the_factors_panel_and_stacks_on_a_shared_edge() -> None:
    assert _panel().position == _panel(factors=None).position == "left"
    assert _panel(factors="left").position == "right"
    for shape in (SHAPE, SMALL):
        left = _panel().fit(shape, ROWS, 10_000)[0]
        assert left[2] <= _factors().bbox(shape)[0]  # clear of the right-hand panel
    same = _panel(position="right").fit(SHAPE, ROWS, 10_000)[0]
    assert same[1] == _factors().bbox(SHAPE)[3] + compositor.HUD_LAYOUT.panel_gap_px


@pytest.mark.parametrize("shape", [SHAPE, SMALL, (480, 854, 3)])
def test_panel_stays_inside_the_frame_and_above_the_bottom_rows(shape) -> None:
    limit = int(shape[0]) - (4 * 22 + 6)
    for pos in ("left", "right"):
        box = _panel(position=pos).fit(shape, ROWS, limit)[0]
        assert 0 <= box[0] < box[2] <= shape[1] and 0 <= box[1] < box[3] <= limit


def test_a_panel_that_cannot_fit_sheds_lines_not_space() -> None:
    assert _panel().fit(SMALL, ROWS, 120)[0][3] <= 120
    assert _panel().draw(np.full(SMALL, 9, np.uint8), ROWS, 10).max() == 9


def test_drawing_changes_only_pixels_inside_the_box() -> None:
    img, panel = np.full(SHAPE, 90, np.uint8), _panel()
    out = panel.draw(img, ROWS, LIMIT)
    x0, y0, x1, y1 = panel.fit(SHAPE, ROWS, LIMIT)[0]
    keep = np.ones(SHAPE[:2], bool)
    keep[y0:y1, x0:x1] = False
    assert (out[keep] == 90).all() and (img == 90).all()
    assert (out[y0:y1, x0:x1] != 90).any()
    assert (out != _panel(by_kind=False).draw(img, ROWS, LIMIT)).any()  # legend


def _feed() -> OverlayFeed:
    return OverlayFeed(
        lambda i: None,
        hand_hud_lines=lambda i: (f"hud {i}",),
        hand_panel_rows=lambda i: ROWS if i < 5 else (),
    )


def test_feed_carries_rows_and_the_panel_replaces_the_text_rows() -> None:
    feed, export = _feed(), ExportSettings(interface_ellipsoids=BOTH)
    assert feed.hand_panel_rows(1) == ROWS and feed.hud_lines_at(1) == ("hud 1",)
    assert feed.held_after(2).hand_panel_rows(9) == ROWS
    follow = replace(feed, ellipsoids_for=lambda s: (None, None, lambda i: ROWS[:1]))
    assert follow.following(object(), lambda k: k).hand_panel_rows(0) == ROWS[:1]
    shown = ep.with_panel(None, export, feed, None)
    assert isinstance(shown.ellipsoids, ep.EllipsoidPanel)
    assert ep.hud_lines(shown, feed, 1) == () and ep.hud_lines(None, feed, 1) == (
        "hud 1",
    )
    text = replace(export, ellipsoid_panel=replace(SETTINGS, layout="text"))
    for other, settings in ((feed, text), (OverlayFeed(lambda i: None), export)):
        assert ep.with_panel(None, settings, other, None) is None  # HUD rows stay


def test_replay_note_survives_the_panel_and_text_layout_is_unchanged() -> None:
    note = "Kinematic Replay: Forces Not Shown"
    plain, replay = _feed(), replace(_feed(), forces_note=note)
    shown = ep.with_panel(None, ExportSettings(interface_ellipsoids=BOTH), plain, None)
    assert ep.hud_lines(shown, replay, 1) == (note,)  # panel shown + replay
    assert ep.hud_lines(shown, plain, 1) == ()  # panel shown, not a replay
    assert ep.hud_lines(None, replay, 1) == ("hud 1", note)  # text layout
    assert ep.hud_lines(None, plain, 1) == ("hud 1",)


def test_legend_row_is_dropped_only_when_a_replay_draws_no_vectors(
    monkeypatch,
) -> None:
    from types import SimpleNamespace as NS

    monkeypatch.setattr(OverlayFeed, "legend", staticmethod(lambda g: "Ref Force"))
    bare = NS(arrows=(), torque_arcs=(), ellipsoids=())
    drawn = NS(arrows=(1,), torque_arcs=(), ellipsoids=())
    plain = _feed()
    replay = replace(plain, forces_note="note")
    assert replay.legend_for(None) == "" and replay.legend_for(bare) == ""
    assert replay.legend_for(drawn) == plain.legend_for(drawn) == "Ref Force"
    assert plain.legend_for(bare) == "Ref Force"
