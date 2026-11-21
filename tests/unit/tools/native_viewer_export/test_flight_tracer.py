"""Ball flight playback and shot tracer of native exports (#15150)."""

from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.golf_view_presets import (
    VIEW_ORDER,
    get_view_preset,
    mujoco_camera_params,
)
from src.shared.python.impact_parameters import ClubheadSeries
from src.shared.python.impact_parameters.flight_report import FlightSummary
from src.shared.python.impact_parameters.target_frame import TargetFrame
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export import cli, flight_tracer as tracer
from src.tools.native_viewer_export.ball import BallPlacement
from src.tools.native_viewer_export.core import (
    ExportSettings,
    SwingInput,
    export_swing,
)
from src.tools.native_viewer_export.overlay2d import pinhole_for_view

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

DEFAULTS = tracer.load_tracer_settings()
BALL = BallPlacement((1.0, -0.2, 0.03), 0.021335, "test")
FRAME = TargetFrame(target_dir=(0.0, -1.0, 0.0))  # target -Y, left of it +X
IMPACT_S, LOOKAT = 1.0, (1.0, 0.0, 0.9)
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


def _tile() -> np.ndarray:
    return np.full((360, 640, 3), 40, np.uint8)


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


@pytest.mark.parametrize(
    ("view", "point"),
    [("down_the_line", (1.4, -8.0, 2.5)), ("face_on", (1.5, -1.5, 1.6))],
)
def test_projection_matches_the_mujoco_camera(view: str, point: tuple) -> None:
    mujoco = pytest.importorskip("mujoco")
    position = " ".join(str(c) for c in point)
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><light pos="0 0 9"/><geom type="sphere" size="0.1" '
        f'pos="{position}" rgba="1 0 0 1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    mujoco_camera_params(view, LOOKAT, None).apply_to(camera)
    renderer = mujoco.Renderer(model, 240, 320)
    try:
        renderer.update_scene(data, camera=camera)
        image = renderer.render().copy()
    finally:
        renderer.close()
    rows, cols = np.nonzero(image[:, :, 0] > 10)  # red sphere on black
    assert rows.size > 4  # the sphere is in the picture
    fov = math.radians(DEFAULTS.fov_y_deg)
    pinhole = pinhole_for_view(view, LOOKAT, None, fov, (320, 240))
    pixels, depth = tracer.project(pinhole, np.array([point]), DEFAULTS.near_clip_m)
    assert pixels[0] == pytest.approx((cols.mean() + 0.5, rows.mean() + 0.5), abs=1.0)
    assert depth[0] > 0.0


def test_draw_adds_the_tracer_and_ball_to_fixed_views_only() -> None:
    flight = _playback()
    tiles = {"down_the_line": _tile(), "hands_closeup": _tile()}
    before = flight.draw(tiles, IMPACT_S - 0.1, LOOKAT, None)
    assert all((before[name] == 40).all() for name in tiles)
    out = flight.draw(tiles, IMPACT_S + 2.0, LOOKAT, None)
    assert (tiles["down_the_line"] == 40).all()  # the input tile is not drawn on
    assert (out["hands_closeup"] == 40).all()  # tracking view: left alone
    drawn = out["down_the_line"]
    red, white = DEFAULTS.tracer_color_rgba[:3], DEFAULTS.ball_color_rgb
    assert (drawn == red).all(axis=2).sum() > 30
    fov = math.radians(DEFAULTS.fov_y_deg)
    pinhole = pinhole_for_view("down_the_line", LOOKAT, None, fov, (640, 360))
    points = flight.flown(IMPACT_S + 2.0)
    assert points is not None
    pixels, _ = tracer.project(pinhole, points, DEFAULTS.near_clip_m)
    u, v = (int(round(c)) for c in pixels[-1])
    assert tuple(drawn[v, u]) == white  # the ball sits on the end of the trace
    tee_u, tee_v = (int(round(c)) for c in pixels[0])
    assert tuple(drawn[tee_v, tee_u]) == red  # the trace starts at the tee
    behind = np.array([[1.0, 50.0, 1.0], [1.0, 60.0, 1.0]])  # behind the camera
    clean = tracer.draw_trace(_tile(), pinhole, behind, 0.02, DEFAULTS)
    assert (clean == 40).all()


def test_flight_view_keeps_the_whole_flight_in_the_picture() -> None:
    wide, plain = (
        get_view_preset("down_the_line_flight"),
        get_view_preset("down_the_line"),
    )
    assert wide.azimuth_deg == plain.azimuth_deg and wide.tracks is None
    assert wide.default_distance_m > plain.default_distance_m
    assert "down_the_line_flight" not in VIEW_ORDER  # the 2x2 grid stays four views
    points = _playback().flown(99.0)
    assert points is not None
    fov = math.radians(DEFAULTS.fov_y_deg)

    def inside(view: str) -> np.ndarray:
        pinhole = pinhole_for_view(view, LOOKAT, None, fov, (1280, 720))
        pixels, _ = tracer.project(pinhole, points, DEFAULTS.near_clip_m)
        return (pixels >= 0).all(axis=1) & (pixels < (1280, 720)).all(axis=1)

    assert inside("down_the_line_flight").all()
    assert not inside("down_the_line").all()  # the close view loses the apex


# --- export loop ---------------------------------------------------------------


class _Writer:
    def __init__(self, sink: list[np.ndarray]) -> None:
        self.append_data = sink.append

    def close(self) -> None:
        pass


class _Backend:
    engine = "fake"

    def unavailable_reason(self) -> None:
        return None

    def render(self, swing, settings, indices, overlay):
        for _k in indices:
            yield {v: _tile() for v in settings.views}


def _swing() -> SwingInput:
    names, q = ("A", "B"), np.zeros((21, 2))
    spec = json.dumps({"coordinate_order": list(names)}).encode()
    zeros = np.zeros((20, 2))
    bundle = InputBundle(spec, names, 0.01, q[0], q[0], zeros, q, q.copy(), "mujoco")
    return SwingInput(bundle, q, "driver", "Driver", "mujoco")


def _settings(**overrides: object) -> ExportSettings:
    base = ExportSettings(views=("face_on",), multiview=False, width=640, height=360)
    return replace(base, speeds=(1.0,), ball=False, **overrides)  # type: ignore[arg-type]


def _frames(tmp_path: Path, **overrides: object) -> list[np.ndarray]:
    sink: list[np.ndarray] = []
    export_swing(
        _Backend(),
        _swing(),
        _settings(**overrides),
        tmp_path,
        writer_factory=lambda path, fps: _Writer(sink),
    )
    return sink


def _short_flight() -> tracer.FlightPlayback:
    """Half a second of flight from an impact 0.1 s into the 0.2 s test swing."""
    t = np.linspace(0.0, 0.5, 26)
    local = np.column_stack([4.0 * t, 0.0 * t, 2.0 * t * (0.5 - t)])
    flight = replace(_flight(), trajectory_m=local, times_s=t, flight_time_s=0.5)
    return tracer.playback_from_flight(flight, FRAME, 0.1, BALL, DEFAULTS)


def test_export_flies_the_ball_and_holds_the_clip_until_it_lands(
    tmp_path: Path, monkeypatch
) -> None:
    plain = _frames(tmp_path, impact_time_s=0.1, hud=False)
    monkeypatch.setattr(tracer, "flight_for_export", lambda *_: _short_flight())
    bare = replace(DEFAULTS, inset=None)  # the inset has its own tests
    frames = _frames(tmp_path, impact_time_s=0.1, hud=False, ball_flight=bare)
    fps = ExportSettings().fps
    # swing ends at 0.2 s; landing 0.6 s plus the landing hold, in real time
    extra = math.ceil((0.4 + DEFAULTS.landing_hold_s) * fps - 1e-9)
    assert len(plain) == 0.2 * fps + 1 and len(frames) == len(plain) + extra
    impact_frame = math.ceil(0.1 * fps - 1e-9)
    assert all((f == 40).all() for f in frames[:impact_frame])  # ball on the tee
    assert all((f != 40).any() for f in frames[impact_frame + 1 :])
    landed = math.ceil(0.6 * fps) + 1
    assert (frames[landed - 3] != frames[landed]).any()  # still moving, then at rest
    assert all((f == frames[-1]).all() for f in frames[landed:])
    with_hud = _frames(tmp_path, impact_time_s=0.1, ball_flight=DEFAULTS)
    plain_hud = _frames(tmp_path, impact_time_s=0.1)
    assert (with_hud[0] != plain_hud[0]).any()  # "Ball flight: on tee" before impact


def test_export_is_unchanged_with_the_option_off(tmp_path: Path, monkeypatch) -> None:
    def boom(*args: object) -> None:
        raise AssertionError("no flight may be computed when the option is off")

    assert ExportSettings().ball_flight is None
    plain = _frames(tmp_path, impact_time_s=0.1)
    monkeypatch.setattr(tracer, "flight_for_export", boom)
    off = _frames(tmp_path, impact_time_s=0.1, ball_flight=None)
    assert len(off) == len(plain)
    assert all((a == b).all() for a, b in zip(off, plain, strict=True))


def test_export_without_a_flight_says_why_and_adds_no_frames(tmp_path: Path) -> None:
    unknown = tracer.flight_for_export(_swing(), _settings(ball_flight=DEFAULTS))
    assert unknown is not None
    assert unknown.hud_lines(0.1) == ("Ball flight: no impact detected",)
    timed = _settings(ball_flight=DEFAULTS, impact_time_s=0.1)
    no_ball = tracer.flight_for_export(_swing(), timed)
    assert no_ball is not None and no_ball.hud_lines(0.1)[0].endswith("no ball placed")
    failed = tracer.flight_for_export(replace(_swing(), ball=BALL), timed)
    assert failed is not None  # the two-coordinate test specification has no club
    assert failed.hud_lines(0.1)[0].startswith("Ball flight: unavailable")
    frames = _frames(tmp_path, impact_time_s=0.1, ball_flight=DEFAULTS, hud=False)
    assert len(frames) == 13 and all((f == 40).all() for f in frames)
    with pytest.raises(ValueError, match="step_s"):
        _short_flight().held(_swing(), None, 0.0)


def test_cli_flags_turn_the_flight_on(tmp_path: Path) -> None:
    base = ["--bundle", "b.npz", "--out", str(tmp_path), "--swing", "s"]
    parser = cli.build_parser()
    assert cli.build_settings(parser.parse_args(base)).ball_flight is None
    flags = [*base, "--ball-flight", "--no-grid", "--views", "down_the_line_flight"]
    on = cli.build_settings(parser.parse_args(flags))
    assert on.ball_flight == DEFAULTS and on.views == ("down_the_line_flight",)
    custom = tmp_path / "tracer.json"
    raw = json.loads(tracer.SETTINGS_PATH.read_text(encoding="utf-8"))
    custom.write_text(json.dumps({**raw, "tracer_width_px": 5}), encoding="utf-8")
    picked = cli.build_settings(
        parser.parse_args([*base, "--ball-flight-settings", str(custom)])
    )
    assert picked.ball_flight == replace(DEFAULTS, tracer_width_px=5)
