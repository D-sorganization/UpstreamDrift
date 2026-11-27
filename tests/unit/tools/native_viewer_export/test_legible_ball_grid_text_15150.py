"""Legible flying ball and tile-scaled grid panel text (#15150 review notes)."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from src.tools.native_viewer_export import (
    compositor,
    flight_tracer as tracer,
    impact_factors as factors,
)
from src.tools.native_viewer_export.overlay2d import pinhole_for_view
from tests.unit.tools.native_viewer_export import (
    test_flight_tracer as base,
    test_impact_factors as ft,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REVIEWED_GRID = (720, 1280, 3)  # the reviewed four-view clip: 360 px tiles


def test_ball_radius_is_the_projection_when_it_is_larger_than_the_minimum() -> None:
    assert tracer.drawn_ball_radius_px(7.5, 5.0) == 7.5


def test_ball_radius_never_drops_below_the_minimum() -> None:
    assert tracer.drawn_ball_radius_px(1.5, 5.0) == 5.0
    assert tracer.drawn_ball_radius_px(0.0, 5.0) == 5.0


@pytest.mark.parametrize(
    "projected, minimum", [(1.0, 0.0), (1.0, -2.0), (-1.0, 5.0), (math.nan, 5.0)]
)
def test_ball_radius_rule_refuses_bad_inputs(projected: float, minimum: float) -> None:
    with pytest.raises(ValueError):
        tracer.drawn_ball_radius_px(projected, minimum)


def test_shipped_minimum_radius_is_legible_and_documented() -> None:
    assert base.DEFAULTS.min_ball_radius_px >= 4  # review: a 3-4 px ball was lost
    raw = json.loads(tracer.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert "min_ball_radius_px" in raw["sources"]
    assert "review" in raw["sources"]["min_ball_radius_px"]


def test_distant_ball_covers_at_least_the_minimum_disc() -> None:
    size = (48, 36)
    camera = pinhole_for_view(
        "down_the_line", base.LOOKAT, None, math.radians(45.0), size
    )
    style = replace(base.DEFAULTS, tracer_color_rgba=(0, 0, 0, 0))
    minimum = style.min_ball_radius_px
    point = np.asarray([base.LOOKAT, base.LOOKAT], dtype=float)
    tile = np.full((size[1], size[0], 3), 40, np.uint8)
    out = tracer.draw_trace(tile, camera, point, 1e-4, style)  # projects << 1 px
    rows, cols = np.mgrid[0 : size[1], 0 : size[0]]
    disc = np.hypot(cols - size[0] / 2.0, rows - size[1] / 2.0) <= minimum - 0.5
    changed = (out != tile).any(axis=2)
    assert changed[disc].all()
    assert changed.sum() >= disc.sum()


def test_grid_font_scales_with_tile_height_above_the_floor() -> None:
    assert compositor.grid_font_px(360, 0.056, 16) == 20
    assert compositor.grid_font_px(720, 0.056, 16) == 40
    assert compositor.grid_font_px(100, 0.056, 16) == 16  # floor


@pytest.mark.parametrize(
    "tile_h, fraction, floor_px",
    [(0, 0.05, 16), (360, 0.0, 16), (360, -0.1, 16), (360, 0.05, 0), (360, 1.5, 16)],
)
def test_grid_font_rule_refuses_bad_inputs(
    tile_h: int, fraction: float, floor_px: int
) -> None:
    with pytest.raises(ValueError):
        compositor.grid_font_px(tile_h, fraction, floor_px)


def test_reviewed_grid_panel_text_is_larger_and_fits_its_tile() -> None:
    panel = factors.panel_from_series(ft._series(), ft.IMPACT_S, ft.BALL, ft.DEFAULTS)
    (x0, y0, x1, y1), size = panel._box(REVIEWED_GRID, grid=True)
    assert size > compositor.HUD_SIZE_PX  # was pinned to the 16 px floor
    tile_h, tile_w = REVIEWED_GRID[0] // 2, REVIEWED_GRID[1] // 2
    natural = compositor.text_panel_box(
        REVIEWED_GRID, factors.TITLE, panel._widest(), "right", size
    )[0]
    assert (x1 - x0, y1 - y0) == (natural[2] - natural[0], natural[3] - natural[1])
    assert x0 >= tile_w and y0 >= tile_h  # inside the bottom-right tile


def test_grid_font_fraction_is_validated_and_documented() -> None:
    with pytest.raises(ValueError, match="grid_font_tile_fraction"):
        replace(ft.DEFAULTS, grid_font_tile_fraction=0.0)
    with pytest.raises(TypeError, match="grid_font_tile_fraction"):
        replace(ft.DEFAULTS, grid_font_tile_fraction="big")  # type: ignore[arg-type]
    raw = json.loads(factors.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert (
        "grid_font_tile_fraction" in raw and "grid_font_tile_fraction" in raw["sources"]
    )
    assert "grid_font_height_divisor" not in raw
