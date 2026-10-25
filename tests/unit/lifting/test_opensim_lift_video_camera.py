"""Pure-data tests for lift video camera and schedule (OSL-8, #13045)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.msk_lift_video_camera import (
    LIFT_VIEWS,
    VIEWER_FOV_Y_RAD,
    LiftVideoConfig,
    compute_lift_framing,
    lift_camera_transforms,
    lift_video_schedule,
    pinhole_for_lifting_view,
)

pytestmark = pytest.mark.unit


def test_lift_video_config_validation() -> None:
    """LiftVideoConfig validates views, fps, speeds and dimensions."""
    cfg = LiftVideoConfig(lift="squat", view="side", fps=60, speed=1.0)
    assert cfg.lift == "squat"
    assert cfg.view == "side"
    assert cfg.fps == 60
    assert cfg.speed == 1.0
    assert cfg.overlay == "forces"

    with pytest.raises(ValueError, match="unknown view"):
        LiftVideoConfig(lift="squat", view="invalid_view")

    with pytest.raises(ValueError, match="fps must be positive"):
        LiftVideoConfig(lift="squat", fps=0)

    with pytest.raises(ValueError, match="speed must be positive"):
        LiftVideoConfig(lift="squat", speed=-1.0)

    with pytest.raises(ValueError, match="unknown overlay"):
        LiftVideoConfig(lift="squat", overlay="unsupported")


@pytest.mark.parametrize("view", LIFT_VIEWS)
def test_lift_camera_transforms_orthonormal(view: str) -> None:
    """Camera transforms produce valid orthonormal SO(3) matrices with det +1."""
    lookat = (0.0, 0.9, 0.0)
    distance = 2.8
    simbody_rot, pinhole_rot, cam_pos = lift_camera_transforms(view, lookat, distance)

    assert simbody_rot.shape == (3, 3)
    assert pinhole_rot.shape == (3, 3)
    assert cam_pos.shape == (3,)

    np.testing.assert_allclose(simbody_rot.T @ simbody_rot, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(pinhole_rot.T @ pinhole_rot, np.eye(3), atol=1e-6)
    assert math.isclose(float(np.linalg.det(simbody_rot)), 1.0, abs_tol=1e-6)
    assert math.isclose(float(np.linalg.det(pinhole_rot)), 1.0, abs_tol=1e-6)


@pytest.mark.parametrize("view", LIFT_VIEWS)
def test_lift_camera_transforms_lookat_centered(view: str) -> None:
    """Pinhole camera projects lookat point to image optical center."""
    lookat = (0.1, 0.85, -0.05)
    distance = 3.0
    _, pinhole_rot, cam_pos = lift_camera_transforms(view, lookat, distance)

    pt_cam = pinhole_rot.T @ (np.asarray(lookat) - cam_pos)
    assert math.isclose(float(pt_cam[0]), 0.0, abs_tol=1e-6)
    assert math.isclose(float(pt_cam[1]), 0.0, abs_tol=1e-6)
    assert pt_cam[2] > 0.0


def test_lift_camera_transforms_invalid_view() -> None:
    with pytest.raises(ValueError, match="unknown view"):
        lift_camera_transforms("diagonal", (0, 0, 0), 2.0)


def test_lift_video_schedule_interpolation() -> None:
    """Frame schedule creates 60 fps sample times and interpolates q."""
    times = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
    q = np.tile(np.linspace(0.0, 1.0, 5)[:, None], (1, 10))

    sched, q_interp, nearest = lift_video_schedule(times, q, fps=60, speed=1.0)
    assert len(sched.sample_times_s) >= 12
    assert q_interp.shape == (len(sched.sample_times_s), 10)
    assert len(nearest) == len(sched.sample_times_s)
    assert np.all(np.diff(sched.sample_times_s) > 0)


@pytest.mark.parametrize("view", LIFT_VIEWS)
def test_compute_lift_framing_lifter_span(view: str) -> None:
    """Lifter bounding box vertical span is >= 60% of frame height across views."""
    points = np.array(
        [
            [-0.2, 0.05, -0.2],
            [-0.2, 0.05, 0.2],
            [0.1, 1.78, 0.0],
            [-0.1, 0.95, -0.2],
            [-0.1, 0.95, 0.2],
            [0.0, 1.45, -0.25],
            [0.0, 1.45, 0.25],
        ]
    )
    bar_ends = np.array([[0.0, 1.45, -1.10], [0.0, 1.45, 1.10]])

    lookat, dist = compute_lift_framing(
        points, bar_ends, view=view, image_size_px=(800, 600)
    )
    cam = pinhole_for_lifting_view(view, lookat, dist, VIEWER_FOV_Y_RAD, (800, 600))

    uv, mask = cam.project(points)
    assert np.all(mask), f"points behind camera in view {view}"
    v_min, v_max = float(np.min(uv[:, 1])), float(np.max(uv[:, 1]))
    vertical_span_ratio = (v_max - v_min) / 600.0
    assert vertical_span_ratio >= 0.60, (
        f"view {view}: vertical span ratio {vertical_span_ratio:.3f} < 0.60"
    )


@pytest.mark.parametrize("view", LIFT_VIEWS)
def test_compute_lift_framing_barbell_endpoints_visible(view: str) -> None:
    """Barbell endpoints (+/- 1.10 m along bar) remain strictly inside the frame."""
    points = np.array([[0.0, 0.05, 0.0], [0.0, 1.78, 0.0]])
    bar_ends = np.array(
        [
            [0.05, 1.45, -1.10],
            [0.05, 1.45, 1.10],
            [-0.15, 0.85, -1.10],
            [-0.15, 0.85, 1.10],
        ]
    )

    lookat, dist = compute_lift_framing(
        points, bar_ends, view=view, image_size_px=(800, 600)
    )
    cam = pinhole_for_lifting_view(view, lookat, dist, VIEWER_FOV_Y_RAD, (800, 600))

    uv, mask = cam.project(bar_ends)
    assert np.all(mask), f"bar ends behind camera in view {view}"
    for i, (u, v) in enumerate(uv):
        assert 0.0 <= u <= 800.0, f"bar endpoint {i} u={u:.1f} out of bounds in {view}"
        assert 0.0 <= v <= 600.0, f"bar endpoint {i} v={v:.1f} out of bounds in {view}"
