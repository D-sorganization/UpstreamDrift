"""Carry, roll and total of native flight exports (#15443, child 6)."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from src.shared.python.sg_optimizer.shot_model.ground_runout import SlopeAwareGround
from src.tools.native_viewer_export import (
    flight_inset as fi,
    flight_tracer as tracer,
    rollout as ro,
)
from src.tools.native_viewer_export.core import ClipOverlays
from tests.unit.tools.native_viewer_export import test_flight_tracer as base

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SETTINGS = tracer.load_tracer_settings()  # yards, rollout on
YD = 0.9144


def _playback(club: str | None = "driver", settings=SETTINGS, valid: bool = True):
    flight = base._flight(valid)
    play = tracer.playback_from_flight(flight, base.FRAME, 1.0, base.BALL, settings)
    return flight, tracer.with_rollout(play, flight, club)


def test_roll_is_the_existing_runout_model_on_the_landing_state() -> None:
    flight, play = _playback("driver")
    got = play.rollout
    assert got is not None
    # synthetic parabola: lands at (160, -8) m; the last 0.1 s step moves
    # (40, -2, -19.5) m/s (landing velocity is the last finite difference)
    spin = ro.landing_backspin_rads(flight.spin_rpm, flight.flight_time_s)
    state = np.array([[160.0 / YD, -8.0 / YD, 40.0, -2.0, -19.5, spin]])
    rest = SlopeAwareGround()(state, "fairway", "flat", "driver")[0]
    offset = (rest - state[0, :2]) * YD
    assert got.roll_m == pytest.approx(math.hypot(*offset), rel=1e-9)
    assert got.total_m == pytest.approx(flight.carry_m + got.roll_m)
    assert got.end_offset_m == pytest.approx(tuple(offset))
    assert got.roll_m > 0.0 and got.end_offset_m[0] > 0.0  # rolls on down-range
    assert got.skill == "driver" and got.model_id == "sg_runout_v2"


def test_a_scoring_iron_rolls_less_than_the_driver_and_unknown_clubs_none() -> None:
    _, driver = _playback("driver")
    _, iron = _playback("7-iron")
    assert iron.rollout is not None and driver.rollout is not None
    assert 0.0 < iron.rollout.roll_m < driver.rollout.roll_m
    assert _playback("putter")[1].rollout is None
    assert _playback(None)[1].rollout.skill == SETTINGS.rollout.default_skill


def test_no_roll_for_an_invalid_flight_or_when_switched_off() -> None:
    wide = replace(SETTINGS, draw_invalid_flight=True)
    assert _playback(valid=False, settings=wide)[1].rollout is None
    off = replace(SETTINGS, rollout=replace(SETTINGS.rollout, enabled=False))
    assert _playback(settings=off)[1].rollout is None
    assert _playback(settings=replace(SETTINGS, rollout=None))[1].rollout is None


def test_hud_states_carry_roll_total_and_the_model_assumption() -> None:
    flight, play = _playback("driver")
    text = "\n".join(play.hud_lines(5.0))
    roll, total = play.rollout.roll_m / YD, play.rollout.total_m / YD
    assert f"Roll {roll:.0f} yd" in text and f"Total {total:.0f} yd" in text
    assert "rollout: flat-fairway model" in text
    assert "Carry 175 yd" in text  # 160 m
    assert "Roll" not in "\n".join(
        _playback(None, replace(SETTINGS, rollout=None))[1].lines
    )


def test_manifest_has_si_roll_total_and_the_model_source() -> None:
    _, play = _playback("driver")
    record = tracer.manifest_record(ClipOverlays(flight=play))["rollout"]
    assert record["roll_m"] == pytest.approx(play.rollout.roll_m)
    assert record["total_m"] == pytest.approx(160.0 + record["roll_m"])
    assert record["assumption"] == "rollout: flat-fairway model"
    assert record["model"] == "sg_runout_v2" and record["skill"] == "driver"
    assert record["lie"] == "fairway" and record["slope"] == "flat"
    assert "Assumed prior" in record["source"]
    _, bare = _playback(None, replace(SETTINGS, rollout=None))
    assert "rollout" not in tracer.manifest_record(ClipOverlays(flight=bare))


def test_rollout_settings_are_documented_and_validated() -> None:
    raw = json.loads(ro.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}
    with pytest.raises(ValueError, match="lie"):
        ro.RolloutSettings(lie="moon")
    with pytest.raises(ValueError, match="default_skill"):
        ro.RolloutSettings(default_skill="putt")
    with pytest.raises(TypeError, match="enabled"):
        ro.RolloutSettings(enabled="yes")  # type: ignore[arg-type]


def test_inset_extends_to_the_rest_point_and_labels_the_roll() -> None:
    _, play = _playback("driver")
    data = fi.inset_data(play, 1.0 + 30.0)
    assert data is not None and data.roll is not None
    assert data.roll_end[0] == pytest.approx(160.0 + play.rollout.end_offset_m[0])
    cfg = replace(SETTINGS.inset, enabled=True)
    frame = np.full((720, 1280, 3), 60, np.uint8)
    loud = replace(cfg, roll_color_rgba=(10, 250, 10, 255), opacity=1.0)
    out = fi.paint(frame, play, 1.0 + 30.0, (900, 40, 1272, 330), loud)
    assert (out == np.array([10, 250, 10], np.uint8)).all(axis=2).sum() > 20
    mid = fi.paint(frame, play, 1.0 + 2.0, (900, 40, 1272, 330), loud)
    assert (mid == np.array([10, 250, 10], np.uint8)).all(axis=2).sum() == 0


def _specs(shape: tuple[int, int], time_s: float, limit: float = 0.85):
    """Label specs as ``paint`` lays them out for an inset in a frame of ``shape``."""
    _, play = _playback("driver")
    cfg, data = SETTINGS.inset, fi.inset_data(play, time_s)
    rect = fi.layout_rect((*shape, 3), cfg, [], 40, int(shape[0] * limit))
    size = fi.font_px_for((*shape, 3), cfg)
    reach = np.vstack([data.full, data.roll_end])
    extent = (reach[:, 0].max(), data.apex_m, np.abs(reach[:, 1]).max())
    width, height = rect[2] - rect[0], rect[3] - rect[1]
    plots = fi.plots_for((0, 0, width, height), cfg, size, extent, 2)
    return plots, fi.label_specs(plots, data, cfg, size)


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


@pytest.mark.parametrize("shape", [(544, 640), (720, 1280)])
@pytest.mark.parametrize("time_s", [1.0 + 2.5, 1.0 + 30.0])
@pytest.mark.parametrize("limit", [0.85, 0.35])  # a short inset beside the panels
def test_label_boxes_stay_in_their_axes_and_clear_of_titles(
    shape, time_s, limit
) -> None:
    plots, (titles, labels) = _specs(shape, time_s, limit)
    title_boxes = [fi.label_box(t) for t in titles]
    boxes = [(spec, fi.label_box(spec)) for spec in labels]
    landed = time_s > 2.0 + 10
    assert any(s.text.startswith("Carry") for s, _ in boxes) == landed
    for spec, box in boxes:
        x0, y0, x1, y1 = spec.bounds
        assert x0 <= box[0] and box[2] <= x1 and y0 <= box[1] and box[3] <= y1, spec
        assert not any(_overlap(box, t) for t in title_boxes), spec.text
    for i, (_, a) in enumerate(boxes):
        assert not any(_overlap(a, b) for _, b in boxes[i + 1 :])
