"""Ball flight playback and shot tracer of native exports (#15150)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.impact_parameters import ClubheadSeries
from src.shared.python.impact_parameters.flight_report import FlightSummary
from src.shared.python.impact_parameters.target_frame import TargetFrame
from src.tools.native_viewer_export import flight_tracer as tracer
from src.tools.native_viewer_export.ball import BallPlacement

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

DEFAULTS = tracer.load_tracer_settings()
BALL = BallPlacement((1.0, -0.2, 0.03), 0.021335, "test")
FRAME = TargetFrame(target_dir=(0.0, -1.0, 0.0))  # target -Y, left of it +X
IMPACT_S = 1.0
REASON = "impact spin exceeds the driver validity envelope (6464 rpm); more text"


def _flight(valid: bool = True) -> FlightSummary:
    """160 m carry in 4 s, 8 m right of the target line, 20 m apex."""
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
        valid=valid,
        invalid_reason="" if valid else REASON,
        times_s=t,
    )


def _playback(settings: tracer.TracerSettings = DEFAULTS, valid: bool = True):
    return tracer.playback_from_flight(_flight(valid), FRAME, IMPACT_S, BALL, settings)


def test_settings_file_is_validated(tmp_path: Path) -> None:
    assert DEFAULTS.fade_after_s is None and not DEFAULTS.draw_invalid_flight
    with pytest.raises(ValueError, match="0..255"):
        replace(DEFAULTS, tracer_color_rgba=(300, 0, 0, 255))
    with pytest.raises(TypeError, match="ball_color_rgb"):
        replace(DEFAULTS, ball_color_rgb=(255, 255))
    with pytest.raises(ValueError, match="tracer_width_px"):
        replace(DEFAULTS, tracer_width_px=0)
    with pytest.raises(ValueError, match="fade_after_s"):
        replace(DEFAULTS, fade_after_s=-1.0)
    with pytest.raises(ValueError, match="fov_y_deg"):
        replace(DEFAULTS, fov_y_deg=180.0)
    with pytest.raises(TypeError, match="draw_invalid_flight"):
        replace(DEFAULTS, draw_invalid_flight="yes")  # type: ignore[arg-type]
    raw = json.loads(tracer.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}  # every entry is explained
    del raw["max_hold_s"]
    short = tmp_path / "tracer.json"
    short.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="max_hold_s"):
        tracer.load_tracer_settings(short)


def test_ball_waits_on_the_tee_then_flies_in_real_time_and_rests() -> None:
    flight = _playback()
    assert flight.flown(IMPACT_S - 0.01) is None  # still the scene's own tee ball
    at = flight.flown(IMPACT_S)
    assert at is not None and at[-1] == pytest.approx(BALL.position_m)
    mid = flight.flown(IMPACT_S + 2.05)
    assert mid is not None
    # 2.05 s after impact: 82 m down the target line (-Y), 4.1 m right (-X)
    # (height: midway between the 20.0 m and 19.95 m samples either side)
    assert mid[-1] == pytest.approx((1.0 - 4.1, -0.2 - 82.0, 0.03 + 19.975))
    late = flight.flown(IMPACT_S + 30.0)
    assert late is not None and late[-1] == pytest.approx((-7.0, -160.2, 0.03))
    assert flight.duration_s == pytest.approx(4.0)


def test_tracer_persists_unless_a_fade_is_set() -> None:
    flight = _playback()
    early, later = flight.flown(IMPACT_S + 1.0), flight.flown(IMPACT_S + 3.0)
    assert early is not None and later is not None and len(later) > len(early)
    assert later[0] == pytest.approx(BALL.position_m)  # the whole trace is kept
    assert (later[: len(early) - 1] == early[:-1]).all()
    faded = _playback(replace(DEFAULTS, fade_after_s=0.5)).flown(IMPACT_S + 3.0)
    assert faded is not None and len(faded) == 6  # 2.5 .. 3.0 s in 0.1 s steps
    assert faded[0, 1] == pytest.approx(-0.2 - 100.0)
    assert faded[-1] == pytest.approx(later[-1])


def test_hud_lines_say_where_the_ball_is_and_what_the_model_predicts() -> None:
    flight = _playback()
    assert flight.hud_lines(0.5) == ("Ball flight: on tee",)
    flying, landed = flight.hud_lines(IMPACT_S + 1.0), flight.hud_lines(9.0)
    assert flying[0] == "Ball flight (model prediction): in flight"
    assert landed[0] == "Ball flight (model prediction): landed"
    assert flying[1:] == landed[1:]
    text = " ".join(flying)
    assert "Ball 140 mph" in text and "Launch 12.0 deg" in text
    assert "Spin 2600 rpm" in text and "Carry 160 m" in text
    assert "Apex 20 m" in text and "Side 8 m right" in text and "Time 4.0 s" in text


def test_a_flight_outside_the_model_envelope_is_not_drawn_by_default() -> None:
    hidden = _playback(valid=False)
    assert hidden.flown(IMPACT_S + 1.0) is None and hidden.duration_s == 0.0
    expected = "Ball flight: not shown, impact spin exceeds the driver validity"
    assert hidden.hud_lines(2.0)[0].startswith(expected)
    shown = _playback(replace(DEFAULTS, draw_invalid_flight=True), valid=False)
    assert shown.flown(IMPACT_S + 1.0) is not None
    assert "outside the model validity envelope" in shown.hud_lines(2.0)[0]


def test_flight_starts_from_the_state_the_impact_card_reads(monkeypatch) -> None:
    t = np.arange(0.0, 0.2, 0.001)
    centre = np.column_stack([40.0 * (t - 0.1), 0.0 * t, 0.0 * t])
    series = ClubheadSeries(
        times_s=t,
        face_center_m=centre,
        velocity_mps=np.gradient(centre, t, axis=0),
        face_normal=np.tile([1.0, 0.0, 0.0], (len(t), 1)),
        toe_axis=np.tile([0.0, -1.0, 0.0], (len(t), 1)),
        grip_axis=np.tile([0.0, 0.0, 1.0], (len(t), 1)),
    )
    seen: list[tuple[int, tuple[float, float, float]]] = []

    def fake(series_in, index, frame):
        seen.append((index, frame.target_dir))
        return _flight()

    monkeypatch.setattr(tracer, "flight_for_impact", fake)
    flight = tracer.playback_from_series(series, 0.1, BALL, DEFAULTS)
    assert seen == [(99, (1.0, 0.0, 0.0))]  # last sample before the impact time
    end = flight.flown(99.0)
    assert end is not None and end[-1, 0] == pytest.approx(1.0 + 160.0)  # along +X


def test_bad_input_is_rejected() -> None:
    flight = _playback()
    with pytest.raises(ValueError, match="finite"):
        flight.flown(float("nan"))
    with pytest.raises(TypeError, match="TracerSettings"):
        tracer.FlightPlayback({}, "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="one time per point"):
        replace(flight, times_s=np.arange(3.0))
    with pytest.raises(ValueError, match="start at 0"):
        replace(flight, times_s=np.asarray(flight.times_s) + 1.0)
