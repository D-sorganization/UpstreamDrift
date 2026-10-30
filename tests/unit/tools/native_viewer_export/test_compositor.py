"""2x2 multi-view compositor with Pillow labels and HUD (NV-6, #11679)."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.native_viewer_export.compositor import (
    HudInfo,
    compose_grid,
    draw_hud,
    draw_label,
    event_caption_for_time,
    grid_geometry,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _tile(value: int, h: int = 60, w: int = 80) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_grid_geometry_2x2() -> None:
    geo = grid_geometry(n_tiles=4, tile_h=60, tile_w=80, cols=2)
    assert (geo.rows, geo.cols) == (2, 2)
    assert (geo.height, geo.width) == (120, 160)
    assert geo.origin(0) == (0, 0)  # (x, y)
    assert geo.origin(1) == (80, 0)
    assert geo.origin(2) == (0, 60)
    assert geo.origin(3) == (80, 60)


def test_compose_grid_places_tiles_in_reading_order() -> None:
    tiles = [_tile(10), _tile(20), _tile(30), _tile(40)]
    out = compose_grid(tiles)
    assert out.shape == (120, 160, 3) and out.dtype == np.uint8
    assert out[10, 10, 0] == 10 and out[10, 100, 0] == 20
    assert out[90, 10, 0] == 30 and out[90, 100, 0] == 40


def test_compose_grid_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compose_grid([_tile(1), _tile(1, w=40), _tile(1), _tile(1)])
    with pytest.raises(ValueError, match="at least one"):
        compose_grid([])
    with pytest.raises(ValueError, match="uint8"):
        compose_grid([np.zeros((4, 4, 3), dtype=float)])


def test_label_is_drawn_with_pillow_and_leaves_input_untouched() -> None:
    img = _tile(0)
    out = draw_label(img, "Face-on", (4, 4))
    assert img.max() == 0, "input must not be mutated"
    assert out[:24, :70].max() == 255, "white label pixels expected"
    assert (out[:24, :70] == 0).any(), "black outline/background retained"
    assert draw_label(img, "", (4, 4)).max() == 0, "empty label draws nothing"


def test_label_text_changes_the_pixels() -> None:
    a = draw_label(_tile(0), "Face-on", (4, 4))
    b = draw_label(_tile(0), "Overhead", (4, 4))
    assert not np.array_equal(a, b)


def test_hud_shows_time_engine_club_and_legend() -> None:
    hud = HudInfo(
        time_s=0.612,
        engine="Drake",
        club="Driver",
        legend="green=GRF  orange=torque",
    )
    lines = hud.lines()
    assert lines[0] == "Drake | Driver | t = 0.612 s"
    assert lines[1] == "green=GRF  orange=torque"
    out = draw_hud(_tile(0, h=120, w=240), hud)
    assert out[-60:].max() == 255, "HUD drawn along the bottom edge"
    assert out[:40].max() == 0, "top edge untouched"
    assert HudInfo(0.0, "Drake", "Driver", "").lines() == [
        "Drake | Driver | t = 0.000 s"
    ]


def test_hud_validates_time() -> None:
    with pytest.raises(ValueError, match="finite"):
        HudInfo(float("nan"), "Drake", "Driver", "")


def test_hud_event_caption_names_the_current_event() -> None:
    """HUD event captions default off; when enabled, append 'Event: <name>' as the last line."""
    default_hud = HudInfo(0.25, "Drake", "Driver", "green=GRF")
    assert default_hud.event_caption == ""
    assert all(not line.startswith("Event:") for line in default_hud.lines())

    document = {
        "schema": "upstreamdrift.anatomical_event_table",
        "events": {
            "address": {"frame": 0, "time_s": 0.0},
            "top": {"frame": 40, "time_s": 0.40},
            "pelvis_reversal": {"frame": 45, "time_s": 0.45},
            "impact": {"frame": 80, "time_s": 0.80},
            "finish": {"frame": 100, "time_s": 1.00},
        },
    }
    expected_sequence = [
        (0.0, "address"),
        (0.20, "address"),
        (0.40, "top"),
        (0.45, "pelvis reversal"),
        (0.80, "impact"),
        (1.00, "finish"),
    ]
    for t_s, expected_name in expected_sequence:
        name = event_caption_for_time(t_s, document)
        assert name == expected_name
        hud = HudInfo(
            time_s=t_s,
            engine="Drake",
            club="Driver",
            legend="green=GRF",
            shading_legend="red=tension",
            event_caption=name,
        )
        lines = hud.lines()
        assert lines[-1] == f"Event: {expected_name}"

    base_img = draw_hud(_tile(0, h=120, w=280), default_hud)
    captioned_img = draw_hud(
        _tile(0, h=120, w=280),
        HudInfo(0.25, "Drake", "Driver", "green=GRF", event_caption="impact"),
    )
    assert not np.array_equal(base_img, captioned_img)

    with pytest.raises(TypeError, match="event_caption"):
        HudInfo(0.0, "Drake", "Driver", "", event_caption=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        event_caption_for_time(float("nan"), document)
