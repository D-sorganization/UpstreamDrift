"""Camera presets and playback phases of the putting tile (#14473)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.putting_dynamics import Mode, phase_marks
from src.tools.putting_green_gui import _scene_builder
from src.tools.putting_green_gui._camera import (
    CAMERA_PRESETS,
    camera_pose,
    camera_position,
)
from src.tools.putting_green_gui._scene_builder import PuttConfig, build_putt_scene

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def scene() -> _scene_builder.PuttScene:
    return build_putt_scene(PuttConfig(putter_speed_ms=1.8, slope_deg=1.0))


def _line(scene: _scene_builder.PuttScene) -> np.ndarray:
    along = np.asarray(scene.cup_xyz[:2]) - np.asarray(scene.start_xyz[:2])
    return along / np.linalg.norm(along)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    """The z of the 2D cross product: positive when ``b`` is left of ``a``."""
    return float(a[0] * b[1] - a[1] * b[0])


def test_the_presets_are_the_orbit_and_the_three_studio_views() -> None:
    assert CAMERA_PRESETS == ("orbit", "down_the_line", "face_on", "overhead")


def test_down_the_line_sits_behind_the_ball_on_the_target_line(scene) -> None:
    eye = camera_position(camera_pose(scene, "down_the_line"))[:2]
    to_eye = eye - np.asarray(scene.start_xyz[:2])
    # Behind the ball, looking along the ball-cup line at the cup.
    assert float(np.dot(to_eye, _line(scene))) < 0.0
    cross = _cross(_line(scene), to_eye / np.linalg.norm(to_eye))
    assert cross == pytest.approx(0.0, abs=1e-9)


def test_face_on_looks_square_at_a_right_handed_stroke(scene) -> None:
    pose = camera_pose(scene, "face_on")
    assert pose.center == pytest.approx(tuple(scene.start_xyz))
    to_eye = camera_position(pose)[:2] - np.asarray(scene.start_xyz[:2])
    # Square to the line, on the side away from a right-handed player.
    assert float(np.dot(to_eye, _line(scene))) == pytest.approx(0.0, abs=1e-9)
    assert _cross(_line(scene), to_eye) < 0.0


def test_overhead_looks_straight_down_on_the_whole_green(scene) -> None:
    pose = camera_pose(scene, "overhead")
    width, height = scene.green_size
    assert pose.center == pytest.approx((width / 2.0, height / 2.0, 0.0))
    assert pose.elevation_deg >= 89.0
    half_view = math.radians(30.0)  # half of pyqtgraph's 60 degree field of view
    assert pose.distance * math.tan(half_view) >= max(width, height) / 2.0


def test_orbit_keeps_the_auto_framed_view(scene) -> None:
    width, height = scene.green_size
    pose = camera_pose(scene, "orbit")
    assert pose.center == pytest.approx((width / 2.0, height / 2.0, 0.0))
    assert pose.distance == pytest.approx(max(width, height) * 1.5 + 4.0)


def test_an_unknown_preset_is_refused(scene) -> None:
    with pytest.raises(ValueError, match="camera preset"):
        camera_pose(scene, "grip")


def test_scene_carries_the_core_phase_marks(monkeypatch) -> None:
    # Golden parity: the scene's phases are the core's phase_marks of the
    # very samples it rolled.
    seen = []
    real = _scene_builder.simulate_strike

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        seen.append(result)
        return result

    monkeypatch.setattr(_scene_builder, "simulate_strike", spy)
    scene = build_putt_scene(PuttConfig(putter_speed_ms=1.8))
    assert scene.phases == phase_marks(seen[0].samples)
    assert [mark.mode for mark in scene.phases][-1] is Mode.REST
    assert len(scene.phases) >= 3
