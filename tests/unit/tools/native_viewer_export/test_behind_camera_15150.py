"""The behind-the-golfer views sit midway between ball and feet (#15150)."""

from __future__ import annotations

import json
import logging
import math

import numpy as np
import pytest

from src.shared.python.golf_view_presets import get_view_preset
from src.shared.python.golf_view_presets.adapters import mujoco_camera_params
from src.tools.native_viewer_export import behind_camera as bc
from src.tools.native_viewer_export.core import ExportSettings, view_lookats
from src.tools.native_viewer_export.flight_tracer import grid_cells
from src.tools.native_viewer_export.overlay2d import pinhole_for_view

pytestmark = pytest.mark.unit

ANCHOR = (0.4, -0.1, 0.0)


def _settings(**kw: object) -> ExportSettings:
    base = {
        "views": ("face_on", "down_the_line", "down_the_line_flight"),
        "behind_anchor_m": ANCHOR,
        "multiview": False,
    }
    return ExportSettings(**{**base, **kw})  # type: ignore[arg-type]


def test_settings_file_names_every_number() -> None:
    raw = json.loads(bc.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert raw["behind_fraction"] == 0.5
    assert {"behind_views", "behind_fraction", "foot_bodies"} <= set(raw["sources"])
    loaded = bc.load_behind_settings()
    assert loaded.views == ("down_the_line", "down_the_line_flight")
    assert len(loaded.foot_bodies) == 4


def test_anchor_is_the_fraction_between_ball_and_feet() -> None:
    ball, feet = (0.0, 0.0, 0.0), (1.0, 2.0, 0.0)
    assert bc.anchor_point(ball, feet, 0.5) == (0.5, 1.0, 0.0)
    assert bc.anchor_point(ball, feet, 0.0) == ball
    assert bc.anchor_point(ball, feet, 1.0) == feet
    for bad in (-0.1, 1.5, math.nan):
        with pytest.raises(ValueError, match="fraction"):
            bc.anchor_point(ball, feet, bad)


def test_only_behind_views_move_and_height_and_depth_are_kept() -> None:
    s = _settings()
    far = bc.view_lookat("face_on", s)
    assert far == s.lookat_m
    for name in ("down_the_line", "down_the_line_flight"):
        look = bc.view_lookat(name, s)
        assert look[0] == pytest.approx(ANCHOR[0])
        assert look[1:] == pytest.approx(s.lookat_m[1:])


def test_explicit_lookat_and_missing_anchor_win() -> None:
    explicit = _settings(lookat_explicit=True, lookat_m=(2.0, 1.0, 0.3))
    assert bc.view_lookat("down_the_line", explicit) == (2.0, 1.0, 0.3)
    none = _settings(behind_anchor_m=None)
    assert bc.view_lookat("down_the_line", none) == none.lookat_m


def test_camera_sits_on_the_line_through_the_anchor() -> None:
    s = _settings()
    preset = get_view_preset("down_the_line")
    look = bc.view_lookat("down_the_line", s)
    pos = preset.camera_position(look, preset.default_distance_m)
    along = preset.view_direction()
    off = np.array(ANCHOR) - pos
    lateral = off - np.dot(off, along) * along
    assert abs(np.dot(lateral, preset.image_right())) < 1e-9


def test_overlay_and_render_cameras_agree() -> None:
    s = _settings()
    render = view_lookats(s, [0], None)  # fixed views need no swing
    mapping = bc.look_map(s)
    for name in s.views:
        look = render[name][0]
        assert tuple(mapping[name]) == pytest.approx(look)
        preset = get_view_preset(name)
        pin = pinhole_for_view(name, mapping[name], None, 0.8, (64, 48))
        params = mujoco_camera_params(name, look, None)
        az, el = math.radians(params.azimuth), math.radians(params.elevation)
        direction = np.array(
            [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]
        )
        position = np.array(params.lookat) - params.distance * direction
        assert pin.translation_world_from_camera_m == pytest.approx(position, abs=1e-6)
        assert pin.rotation_world_from_camera[:, 2] == pytest.approx(
            preset.view_direction(), abs=1e-9
        )


def test_flight_grid_uses_the_same_per_view_lookat() -> None:
    class _Flight:
        got: object = None

        def grid_tiles(self, tiles, time_s, lookat, distance):  # type: ignore[no-untyped-def]
            self.got = lookat
            return tiles

    flight = _Flight()
    s = _settings()
    grid_cells(flight, {}, 0.0, s)  # type: ignore[arg-type]
    assert bc.lookat_of(flight.got, "down_the_line")[0] == pytest.approx(ANCHOR[0])
    assert bc.lookat_of(flight.got, "face_on") == s.lookat_m


class _Model:
    def __init__(self, positions: dict[str, tuple[float, float, float]]) -> None:
        self.positions = positions

    def body_positions(self, coords, names):  # type: ignore[no-untyped-def]
        return {n: self.positions[n] for n in names}


def test_feet_midpoint_is_the_mean_of_the_foot_bodies() -> None:
    names = bc.load_behind_settings().foot_bodies
    pos = {n: (1.0 + i, 0.5 * i, 0.0) for i, n in enumerate(names)}
    mid = bc.feet_midpoint(_Model(pos), {}, names)
    assert mid == pytest.approx((2.5, 0.75, 0.0))


def test_missing_feet_fall_back_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _NoFeet:
        def body_positions(self, coords, names):  # type: ignore[no-untyped-def]
            raise KeyError("toes_l")

    with caplog.at_level(logging.WARNING):
        got = bc.anchor_from((0.0, 0.0, 0.0), _NoFeet(), {}, bc.load_behind_settings())
    assert got is None
    assert "feet" in caplog.text
