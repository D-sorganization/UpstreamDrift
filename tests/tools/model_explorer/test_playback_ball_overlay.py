"""Ball, flight and aim line of the live Results Playback panel (#15150).

Qt-free checks of :mod:`playback_ball_overlay`: its settings, the swing it
hands the exporter, and that it paints the exporter's own ``FlightPlayback``
(ball, tracer, aim line) only where and when it should.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.tools.model_explorer import playback_ball_overlay as pbo
from src.tools.model_explorer.pose_results import PoseFrameSource
from src.tools.native_viewer_export import flight_tracer
from src.tools.native_viewer_export.ball import BallPlacement

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC = REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
IMPACT_S = 0.05
SIZE = (240, 160)
#: target, left and up rows: the down-the-line camera looks along world -x
AXES = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])


def _pose(tmp_path: Path, start_s: float = 0.0) -> PoseFrameSource:
    spec = SPEC.read_bytes()
    order = tuple(json.loads(spec)["coordinate_order"])
    q = np.zeros((11, len(order)))
    times = start_s + np.arange(11) * 0.01
    return PoseFrameSource(tmp_path / "run.npz", "mujoco", times, q, order, spec)


def _layout(aim: bool = True) -> pbo.ShotLayout:
    """A 3 s flight 200 m down the target line (world -x), 10 m left of it."""
    tracer = flight_tracer.load_tracer_settings()
    times = np.linspace(0.0, 3.0, 31)
    s = times / 3.0
    local = np.column_stack([200.0 * s, 10.0 * s**2, 30.0 * 4 * s * (1 - s)])
    path = local @ AXES
    ball = BallPlacement((0.0, 0.0, 0.0213), 0.0213, "address_geometry", IMPACT_S)
    play = flight_tracer.FlightPlayback(
        tracer, "model prediction", IMPACT_S, times, path + ball.position_m, 0.0213
    )
    if aim:
        play = replace(play, axes_m=AXES, aim_yaw_rad=0.05)
    return pbo.ShotLayout("draw", ball, play, -4.0, "note")


def test_settings_load_and_reject_bad_values(tmp_path: Path) -> None:
    settings = pbo.load_overlay_settings()
    assert settings.shot_shape in pbo.SHAPES and settings.show_ball
    with pytest.raises(ValueError, match="shot_shape"):
        pbo.BallOverlaySettings(shot_shape="hook")
    with pytest.raises(ValueError, match="reason"):
        pbo.BallOverlaySettings(flight_unavailable_tooltip="no flight")
    with pytest.raises(TypeError, match="show_flight"):
        pbo.BallOverlaySettings(show_flight="yes")  # type: ignore[arg-type]


def test_flight_reason_names_the_rust_wheel_when_it_is_missing() -> None:
    target = "src.shared.python.physics.rust_kernel.is_rust_available"
    with patch(target, return_value=False):
        assert "upstream_physics" in (pbo.flight_unavailable_reason() or "")
    with patch(target, return_value=True):
        assert pbo.flight_unavailable_reason() is None


def test_swing_for_source_counts_time_from_the_first_sample(tmp_path: Path) -> None:
    pose = _pose(tmp_path, start_s=2.0)
    swing = pbo.swing_for_source(pose, "Driver")
    assert swing.n_states == pose.frame_count and swing.club == "Driver"
    assert swing.time_s(10) == pytest.approx(0.1)
    with pytest.raises(TypeError, match="PoseFrameSource"):
        pbo.swing_for_source("run.npz")  # type: ignore[arg-type]


def test_flight_and_aim_are_painted_only_when_asked(tmp_path: Path) -> None:
    overlay = pbo.BallOverlay(_pose(tmp_path), pbo.BallOverlaySettings())
    blank = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    layout, after = _layout(), IMPACT_S + 1.0

    def paint(time_s: float, flight: bool, aim: bool) -> np.ndarray:
        view = "down_the_line"
        return overlay.paint(blank, view, time_s, layout, flight=flight, aim=aim)

    assert np.array_equal(paint(after, False, False), blank)
    assert np.array_equal(paint(0.0, True, False), blank), "no trace before impact"
    traced = paint(after, True, False)
    assert traced.any()
    assert paint(0.0, False, True).any(), "the aim line shows before impact"
    assert not np.array_equal(paint(after, True, True), traced)
    assert overlay.paint(blank, "face_on", after, None, flight=True, aim=True) is blank
    overlay.close()


def test_ball_is_added_to_the_scene_until_impact(tmp_path: Path) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    overlay = pbo.BallOverlay(_pose(tmp_path, 1.0), pbo.BallOverlaySettings())
    counts = []
    for time_s in (1.0, 1.0 + IMPACT_S + 0.01):
        scene = mujoco.MjvScene(model, maxgeom=10)
        overlay.decorate_scene(scene, time_s, _layout())
        counts.append(scene.ngeom)
    assert counts == [1, 0]
    overlay.close()


def test_without_flight_model_the_layout_states_why(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    reason = "the upstream_physics Rust flight wheel is missing or out of date"
    overlay = pbo.BallOverlay(_pose(tmp_path), pbo.BallOverlaySettings(), reason)
    layout = overlay.request("fade", flight=True).result(timeout=120)
    assert layout.flight is None and layout.roll_deg == 0.0
    assert reason in layout.note or "impact" in layout.note, layout.note
    with pytest.raises(ValueError, match="shot shape"):
        overlay.request("hook")
    overlay.close()


def test_flight_is_computed_only_when_asked(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    overlay = pbo.BallOverlay(_pose(tmp_path), pbo.BallOverlaySettings())
    no_solve = AssertionError("no straight-roll solve without flight")
    with patch.object(overlay, "_straight_roll", side_effect=no_solve) as solver:
        ball_only = overlay.request("draw").result(timeout=120)
        assert overlay.request("fade") is overlay.request("draw")
    assert solver.call_count == 0 and ball_only.flight is None
    assert overlay.request("draw", flight=True) is not overlay.request("draw")
    overlay.close()


@pytest.mark.parametrize("view", ["down_the_line", "face_on"])
def test_shifted_lookat_is_used_for_aim_and_flight(tmp_path: Path, view: str) -> None:
    from src.tools.native_viewer_export.overlay2d import pinhole_for_view

    overlay = pbo.BallOverlay(_pose(tmp_path), pbo.BallOverlaySettings())
    blank = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    layout = _layout()
    lookat = (0.7, -0.4, 1.2)
    play = layout.flight
    camera = pinhole_for_view(
        view, lookat, None, math.radians(play.settings.fov_y_deg), SIZE
    )
    aim = np.asarray(play.paint_aim(blank, camera, None))
    expected = replace(play, aim_yaw_rad=None).draw(
        {view: aim}, IMPACT_S + 1.0, lookat, None, None
    )[view]
    actual = overlay.paint(
        blank, view, IMPACT_S + 1.0, layout, flight=True, aim=True, lookat_m=lookat
    )
    np.testing.assert_array_equal(actual, expected)
    default = overlay.paint(blank, view, IMPACT_S + 1.0, layout, flight=True, aim=True)
    assert not np.array_equal(actual, default)
    overlay.close()
