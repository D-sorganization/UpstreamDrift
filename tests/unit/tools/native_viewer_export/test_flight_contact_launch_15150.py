"""Tracer launched from the contact instant; wide-view look settings (#15150)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.golf_view_presets import get_view_preset
from src.shared.python.impact_parameters import ClubheadSeries
from src.shared.python.impact_parameters.flight_report import FlightSummary
from src.tools.native_viewer_export import (
    compositor,
    flight_tracer as tracer,
    impact_factors as factors,
)
from src.tools.native_viewer_export.ball import BallPlacement
from src.tools.native_viewer_export.backends import mujoco_worker
from src.tools.native_viewer_export.overlay2d import pinhole_for_view

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

RATE_HZ, RADIUS_M, SPEED_MPS = 360.0, 0.021335, 40.0
CONTACT_S = 0.1 + 0.4 / RATE_HZ  # 0.4 of a sample after sample 36
DEFAULTS = tracer.load_tracer_settings()


def _struck() -> tuple[ClubheadSeries, BallPlacement]:
    """Head moving +X at 40 m/s that keeps 70 % of its speed after the strike."""
    t = np.arange(0.0, 0.2, 1.0 / RATE_HZ)
    after = np.where(t > CONTACT_S, CONTACT_S + 0.7 * (t - CONTACT_S), t)
    centre = np.column_stack([SPEED_MPS * (after - CONTACT_S), 0 * t, 0 * t])
    face = np.tile([1.0, 0.0, 0.0], (len(t), 1))
    series = ClubheadSeries(
        times_s=t,
        face_center_m=centre,
        velocity_mps=np.gradient(centre, t, axis=0),
        face_normal=face,
        toe_axis=np.tile([0.0, -1.0, 0.0], (len(t), 1)),
        grip_axis=np.tile([0.0, 0.0, 1.0], (len(t), 1)),
    )
    return series, BallPlacement((RADIUS_M, 0.0, 0.0), RADIUS_M, "test")


def _flight() -> FlightSummary:
    t = np.linspace(0.0, 4.0, 41)
    local = np.column_stack([40.0 * t, -2.0 * t, 5.0 * t * (4.0 - t)])
    return FlightSummary(
        carry_m=160.0,
        lateral_m=8.0,
        max_height_m=20.0,
        flight_time_s=4.0,
        ball_speed_mph=140.0,
        launch_deg=12.0,
        spin_rpm=2600.0,
        down_range_m=160.0,
        trajectory_m=local,
        valid=True,
        invalid_reason="",
        times_s=t,
    )


def test_flight_launches_from_the_contact_state_and_time(monkeypatch) -> None:
    series, ball = _struck()
    seen = {}

    def fake(series_in, index, frame, club=None):
        seen.update(series=series_in, index=index)
        return _flight()

    monkeypatch.setattr(tracer, "flight_for_impact", fake)
    flight = tracer.playback_from_series(series, 0.1, ball, DEFAULTS)
    used, at = seen["series"], seen["index"]
    assert used.times_s[at] == pytest.approx(CONTACT_S, abs=2e-5)  # not a sample
    assert flight.impact_time_s == used.times_s[at]  # the tracer starts then
    assert used.velocity_mps[at][0] == pytest.approx(SPEED_MPS, rel=0.01)
    # the card reads that same state: one source of truth
    assert flight.flown(CONTACT_S - 1e-3) is None
    assert flight.flown(CONTACT_S + 1e-3) is not None


def test_ground_extent_is_a_setting_applied_to_the_floor() -> None:
    mujoco = pytest.importorskip("mujoco")
    xml = (
        '<mujoco><worldbody><geom name="visual_floor" type="plane" '
        'size="3 3 0.05"/></worldbody></mujoco>'
    )
    model = mujoco.MjModel.from_xml_string(xml)
    half = mujoco_worker.load_scene_extent().ground_half_extent_m
    assert half > 3.0  # the patch reaches past the landing point of a drive
    mujoco_worker.widen_ground(model, half)
    assert model.geom("visual_floor").size[:2].tolist() == [half, half]
    with pytest.raises(ValueError):
        mujoco_worker.widen_ground(model, -1.0)
    scene = Path(mujoco_worker.__file__).with_name("scene_extent_settings.json")
    assert set(json.loads(scene.read_text())) >= {"ground_half_extent_m", "sources"}


def test_golfer_fills_enough_of_the_wide_view() -> None:
    preset = get_view_preset("down_the_line_flight")
    lookat = (1.0, 0.0, 0.9)
    pinhole = pinhole_for_view(
        preset, lookat, None, math.radians(DEFAULTS.fov_y_deg), (1280, 720)
    )
    feet, head = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 1.8]])
    pixels, _ = tracer.project(pinhole, np.array([feet, head]), DEFAULTS.near_clip_m)
    assert (pixels[0, 1] - pixels[1, 1]) / 720 >= 0.4  # was 0.31 at 7 m


def _lines() -> list[tuple[str, str, bool]]:
    return [(f"Row {i}", "1.0 mph", False) for i in range(8)]


def test_hud_lines_are_laid_out_from_the_text_size() -> None:
    shape = (720, 1280, 3)
    lines = [f"Line {i} Agjpq|" for i in range(5)]
    boxes = compositor.hud_line_boxes(shape, lines, compositor.HUD_SIZE_PX)
    for upper, lower in zip(boxes, boxes[1:], strict=False):
        assert upper[3] <= lower[1], "text lines must not overlap"
    assert boxes[-1][3] <= shape[0] and boxes[0][1] >= 0
    hud = compositor.HudInfo(0.0, "E", "Driver", "", extra_lines=tuple(lines[:3]))
    out = compositor.draw_hud(np.zeros(shape, np.uint8), hud)
    ys, _ = np.nonzero(out.any(axis=2))
    assert ys.max() < shape[0]
    taller = compositor.hud_line_boxes(shape, lines, 2 * compositor.HUD_SIZE_PX)
    assert taller[1][1] - taller[0][1] > boxes[1][1] - boxes[0][1]  # scales


@pytest.mark.parametrize("position", ["left", "right"])
def test_panels_never_cover_the_view_label(position: str) -> None:
    shape = (720, 1280, 3)
    label = compositor.view_label_box(shape, "Down-the-line (wide, ball flight)")
    card = compositor.text_panel_box(shape, "Title", _lines(), position)[0]
    settings = factors.load_factor_settings()
    panel = factors.FactorsPanel(settings, None, "x", ()).bbox(shape)
    for box in (card, panel):
        apart = box[1] >= label[3] or box[0] >= label[2] or box[2] <= label[0]
        assert apart, (position, box, label)
