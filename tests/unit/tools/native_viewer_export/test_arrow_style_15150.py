"""Slim, clean force arrows from a settings file (#15150 Z8 review).

The owner asked for smaller-diameter, less blocky arrows for shared
high-resolution renders. The looks are settings, not literals; the default
keeps the NV-9 720p legibility floor and ``slim`` is held to its own floor
at the height it is meant for.
"""

from __future__ import annotations

import json
import math

import pytest

from src.shared.python.golf_view_presets import (
    VIEW_ORDER,
    VIEWER_FOV_Y_RAD,
    get_view_preset,
)
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.arrow_style import (
    ARROW_STYLE_PATH,
    arrow_style_names,
    head_span_shaft_radii,
    load_arrow_style,
)
from src.tools.native_viewer_export.overlay_feed import default_glyph_style

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]


def test_the_glyph_style_takes_its_shaft_radius_from_the_settings() -> None:
    doc = json.loads(ARROW_STYLE_PATH.read_text(encoding="utf-8"))
    for name in arrow_style_names():
        radius = doc["styles"][name]["shaft_radius_m"]
        assert default_glyph_style(80.0, arrow_style=name).shaft_radius_m == radius
        assert default_glyph_style(arrow_style=name).shaft_radius_m == radius
    default = doc["styles"][doc["default_style"]]["shaft_radius_m"]
    assert default_glyph_style().shaft_radius_m == default
    assert load_arrow_style("slim").shaft_radius_m < default
    assert head_span_shaft_radii() == doc["head_span_shaft_radii"]


@pytest.mark.parametrize("name", ["standard", "slim"])
@pytest.mark.parametrize("view", VIEW_ORDER)
def test_each_style_is_legible_at_its_minimum_height(name: str, view: str) -> None:
    style = load_arrow_style(name)
    fy = (style.min_height_px / 2.0) / math.tan(VIEWER_FOV_Y_RAD / 2.0)
    shaft_px = (
        2.0 * style.shaft_radius_m * fy / get_view_preset(view).default_distance_m
    )
    assert shaft_px >= style.min_shaft_px


def test_an_unknown_style_or_bad_setting_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown arrow style"):
        load_arrow_style("chunky")
    doc = json.loads(ARROW_STYLE_PATH.read_text(encoding="utf-8"))
    doc["styles"]["slim"]["shaft_radius_m"] = -1.0
    bad = tmp_path / "style.json"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="shaft_radius_m"):
        load_arrow_style("slim", bad)
    doc["default_style"] = "missing"
    bad.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="default_style"):
        load_arrow_style(path=bad)


def test_the_arrow_style_flag_defaults_to_the_configured_style() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(BASE).arrow_style == arrow_style_names()[0]
    assert parser.parse_args([*BASE, "--arrow-style", "slim"]).arrow_style == "slim"
    with pytest.raises(SystemExit):
        parser.parse_args([*BASE, "--arrow-style", "chunky"])
