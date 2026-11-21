"""Settings, layout and geometry of the flight inset (#15150)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.tools.native_viewer_export import flight_inset as fi

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SHAPE = (720, 1280, 3)
ON = replace(fi.load_inset_settings(), enabled=True)


def _rect(settings=ON, obstacles=(), top=40, limit=640):
    return fi.layout_rect(SHAPE, settings, list(obstacles), top, limit)


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
    assert not fi.load_inset_settings().enabled  # nothing draws yet
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


def test_layout_is_inside_frame_and_clear_of_view_label_and_hud() -> None:
    for position in fi.POSITIONS:
        rect = _rect(replace(ON, position=position))
        assert rect is not None
        assert _inside(rect, (0, 40, SHAPE[1], 640))


def test_layout_stacks_below_a_panel_on_a_shared_edge() -> None:
    panel = (1000, 40, 1272, 260)
    rect = _rect(obstacles=[panel])
    assert rect is not None and _clear(rect, panel)
    assert rect[1] == panel[3] + ON.panel_gap_px
    bottom = _rect(replace(ON, position="bottom_right"), [(1000, 400, 1272, 500)])
    assert bottom is not None and bottom[3] == 400 - ON.panel_gap_px


def test_layout_ignores_a_panel_on_the_other_edge_and_shrinks_or_hides() -> None:
    far = (8, 40, 300, 600)
    assert _rect(obstacles=[far])[1] == 40  # type: ignore[index]
    tight = _rect(obstacles=[(1000, 40, 1272, 450)])
    assert tight is not None and tight[3] - tight[1] < 0.4 * SHAPE[0]
    assert _rect(obstacles=[(1000, 40, 1272, 600)]) is None


def test_flight_coordinates_use_the_target_frame() -> None:
    axes = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    path = np.array([[1.0, 2.0, 0.0], [1.0, 12.0, 5.0], [3.0, 22.0, 0.0]])
    local = fi.flight_coordinates(path, axes)
    assert np.allclose(local, [[0, 0, 0], [10, 0, 5], [20, -2, 0]])
    assert np.allclose(fi.flight_coordinates(path, axes, path[1])[0], [-10, 0, -5])
    with pytest.raises(ValueError, match="axes_m"):
        fi.flight_coordinates(path, np.eye(2))


def test_curve_stays_inside_its_plots_for_any_exaggeration() -> None:
    rect = _rect()
    assert rect is not None
    t = np.linspace(0.0, 1.0, 40)
    local = np.column_stack([180.0 * t, 25.0 * np.sin(3 * t), 12.0 * 4 * t * (1 - t)])
    extent = (local[-1, 0], 12.0, np.abs(local[:, 1]).max())
    for exag in (None, 1.0, 6.0, 500.0):
        cfg = replace(ON, vertical_exaggeration=exag)
        profile, plan = fi.plots_for(rect, cfg, 16, extent)
        for plot, vertical in ((profile, local[:, 2]), (plan, local[:, 1])):
            pix = plot.pixels(local[:, 0], vertical)
            x0, y0, x1, y1 = plot.box
            assert (pix[:, 0] >= x0).all() and (pix[:, 0] <= x1).all()
            assert (pix[:, 1] >= y0).all() and (pix[:, 1] <= y1).all()
    true = fi.plots_for(rect, replace(ON, vertical_exaggeration=1.0), 16, extent)
    assert true[0].scale_y == pytest.approx(true[0].scale_x)
    tall = fi.plots_for(rect, replace(ON, vertical_exaggeration=1.5), 16, extent)
    assert tall[0].scale_y == pytest.approx(1.5 * tall[0].scale_x)
