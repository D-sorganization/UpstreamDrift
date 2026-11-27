"""Grid-export placement of the impact panel and the carry label (#15150)."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.tools.native_viewer_export import (
    compositor,
    impact_factors as factors,
    landing_marker as lm,
)
from src.tools.native_viewer_export.overlay2d import pinhole_for_view
from tests.unit.tools.native_viewer_export import (
    test_flight_tracer as base,
    test_impact_card as ct,
    test_impact_factors as ft,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

GRID, TILE = (1440, 2560, 3), (720, 1280, 3)
D, LOOKAT, WIDE = base.DEFAULTS, base.LOOKAT, "down_the_line_flight"
INK, EDGE = (10, 250, 250), (3, 2, 1)
SIZE = (1280, 720)
CAMERA = pinhole_for_view(WIDE, LOOKAT, None, math.radians(45.0), SIZE)


def _panel(**changes):
    cfg = replace(ft.DEFAULTS, **changes)
    return factors.panel_from_series(ft._series(), ft.IMPACT_S, ft.BALL, cfg)


def test_grid_box_lies_in_the_anchor_tile_and_clears_the_edges() -> None:
    h, w = GRID[:2]
    for anchor, quad in {
        "bottom_right": (w // 2, h // 2, w, h),
        "bottom_left": (0, h // 2, w // 2, h),
        "top_right": (w // 2, 0, w, h // 2),
        "top_left": (0, 0, w // 2, h // 2),
    }.items():
        x0, y0, x1, y1 = _panel(grid_anchor=anchor).bbox(GRID, grid=True)
        assert quad[0] <= x0
        assert x0 < x1 <= quad[2], anchor
        assert quad[1] <= y0
        assert y0 < y1 <= quad[3], anchor
        assert x0 > 0 and y0 > 0
        assert x1 < w and y1 < h, anchor


def test_grid_panel_fits_one_tile_and_is_readable() -> None:
    panel = _panel()
    x0, y0, x1, y1 = panel.bbox(GRID, grid=True)
    assert x1 - x0 < GRID[1] // 2 and y1 - y0 < GRID[0] // 2
    assert panel._box(GRID, grid=True)[1] >= compositor.HUD_SIZE_PX


def test_grid_box_shrinks_to_fit_a_small_tile() -> None:
    tiny = (160, 280, 3)  # a tile of 140x80 px: the panel is clipped to it
    x0, y0, x1, y1 = _panel().bbox(tiny, grid=True)
    assert 140 <= x0 < x1 <= 280 and 80 <= y0 < y1 <= 160


def test_single_view_box_is_unchanged() -> None:
    panel = _panel()
    assert panel.bbox(TILE) == (860, 38, 1264, 278)  # +1 smash row
    assert panel.bbox(TILE) == panel.bbox(TILE, grid=False)
    assert panel.bbox(GRID) == (1746, 38, 2528, 454)  # grid frame, grid=False


def test_draw_all_changes_only_pixels_inside_each_boxes() -> None:
    panel = _panel()
    outputs = {"2x2": np.full(GRID, 90, np.uint8), "front": np.full(TILE, 90, np.uint8)}
    drawn = panel.draw_all(outputs, ft.IMPACT_S + 0.1, "2x2")
    for name, grid in (("2x2", True), ("front", False)):
        x0, y0, x1, y1 = panel.bbox(outputs[name].shape, grid=grid)
        mask = (drawn[name] != outputs[name]).any(axis=2)
        assert mask.any()
        assert not mask[:y0].any() and not mask[y1:].any()
        assert not mask[:, :x0].any() and not mask[:, x1:].any()


def test_grid_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="grid_anchor"):
        replace(ft.DEFAULTS, grid_anchor="middle")
    with pytest.raises(ValueError, match="grid_font_tile_fraction"):
        replace(ft.DEFAULTS, grid_font_tile_fraction=1.5)
    with pytest.raises(ValueError, match="grid_min_font_px"):
        replace(ft.DEFAULTS, grid_min_font_px=0)
    raw = json.loads(factors.SETTINGS_PATH.read_text(encoding="utf-8"))
    for key in ("grid_anchor", "grid_font_tile_fraction", "grid_min_font_px"):
        assert key in raw and key in raw["sources"]


def _marker(**changes):
    solid = {
        "carry_label": True,
        "carry_label_color_rgba": (*INK, 255),
        "carry_label_outline_rgba": (*EDGE, 255),
    }
    return replace(D, landing_marker=replace(D.landing_marker, **solid, **changes))


def _label_rect(img):
    ink = (img == INK).all(axis=2) | (img == EDGE).all(axis=2)
    rows, cols = np.nonzero(ink)
    return cols.min(), rows.min(), cols.max() + 1, rows.max() + 1


def _px(spot: np.ndarray) -> tuple[float, float]:
    uv, _ = lm.project(CAMERA, spot.reshape(1, 3), 0.05)
    return float(uv[0][0]), float(uv[0][1])


def _spot(dx: float, dy: float) -> np.ndarray:
    return np.array([LOOKAT[0] + dx, LOOKAT[1] + dy, 0.0])


@pytest.mark.parametrize(
    "dx, dy",
    [(-60.0, -80.0), (60.0, -80.0), (-60.0, -20.0), (60.0, -20.0), (400.0, -80.0)],
)
def test_grid_label_is_inside_the_tile_with_the_margin(dx: float, dy: float) -> None:
    style = _marker()
    margin = style.landing_marker.grid_carry_label_margin_px
    tile = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8)
    out = lm.paint_label(tile, CAMERA, _spot(dx, dy), style, 207.0, grid=True)
    x0, y0, x1, y1 = _label_rect(out)
    assert margin <= x0 and x1 <= SIZE[0] - margin
    assert margin <= y0 and y1 <= SIZE[1] - margin


def test_label_below_the_tile_bottom_is_lifted_inside_the_margin() -> None:
    style = _marker(grid_carry_label_keep_out=None)
    margin = style.landing_marker.grid_carry_label_margin_px
    tile = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8)
    spot = _spot(0.05, 2.5)  # near the camera: projects below the tile
    u, v = _px(spot)
    assert v > SIZE[1] - margin
    out = lm.paint_label(tile, CAMERA, spot, style, 207.0, grid=True)
    assert _label_rect(out)[3] <= SIZE[1] - margin


def test_single_view_label_keeps_its_old_placement() -> None:
    style = _marker()
    tile = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8)
    at_edge = lm.paint_label(tile, CAMERA, _spot(400.0, -40.0), style, 207.0)
    assert _label_rect(at_edge)[0] < 6  # flush with the edge, as before


def test_label_that_would_sit_on_the_central_subject_is_left_out_of_a_grid_tile() -> (
    None
):
    style = _marker()
    keep = style.landing_marker.grid_carry_label_keep_out
    assert keep is not None
    tile = np.full((SIZE[1], SIZE[0], 3), 40, np.uint8)
    centre = _spot(0.0, -80.0)
    u, v = _px(centre)
    assert keep[0] * SIZE[0] < u < keep[2] * SIZE[0]
    assert keep[1] * SIZE[1] < v < keep[3] * SIZE[1]
    assert (lm.paint_label(tile, CAMERA, centre, style, 207.0, grid=True) == tile).all()
    assert (lm.paint_label(tile, CAMERA, centre, style, 207.0) != tile).any()
    off = _marker(grid_carry_label_keep_out=None)
    drawn = lm.paint_label(tile, CAMERA, centre, off, 207.0, grid=True)
    assert (drawn != tile).any()


def test_label_settings_are_validated_and_documented() -> None:
    marker = D.landing_marker
    with pytest.raises(ValueError, match="grid_carry_label_margin_px"):
        replace(marker, grid_carry_label_margin_px=-1)
    with pytest.raises(TypeError, match="grid_carry_label_margin_px"):
        replace(marker, grid_carry_label_margin_px=2.5)
    for bad in ((0.5, 0.1, 0.4, 0.9), (0.1, 0.1, 0.9), (-0.1, 0, 1, 1), (0, 0, 1, 2)):
        with pytest.raises(ValueError, match="grid_carry_label_keep_out"):
            replace(marker, grid_carry_label_keep_out=bad)
    with pytest.raises(ValueError, match="grid_carry_label_keep_out"):
        replace(marker, grid_carry_label_keep_out="centre")
    path = Path(lm.__file__).with_name("flight_tracer_settings.json")
    notes = json.loads(path.read_text(encoding="utf-8"))
    block = notes["landing_marker"]
    assert "grid_carry_label_margin_px" in block
    assert "grid_carry_label_keep_out" in block
    assert "grid_carry_label" in notes["sources"]["landing_marker"]


def test_impact_card_takes_the_grid_placement_only_on_the_grid_frame() -> None:
    card, layout = ct._card(), ft.DEFAULTS.grid_layout
    h, w = GRID[:2]
    x0, y0, x1, y1 = compositor.impact_card_bbox(GRID, card, grid=layout)
    assert x0 >= w // 2 and x1 < w and y0 >= h // 2 and y1 < h
    single = compositor.impact_card_bbox(TILE, card)
    assert single == compositor.impact_card_bbox(TILE, card, grid=None)
    outputs = {"2x2": np.full(GRID, 90, np.uint8), "front": np.full(TILE, 90, np.uint8)}
    drawn = factors.draw_card_all(outputs, card, "2x2")
    for name, box in (("2x2", (x0, y0, x1, y1)), ("front", single)):
        mask = (drawn[name] != outputs[name]).any(axis=2)
        assert mask.any()
        assert not mask[: box[1]].any() and not mask[box[3] :].any()
        assert not mask[:, : box[0]].any() and not mask[:, box[2] :].any()
