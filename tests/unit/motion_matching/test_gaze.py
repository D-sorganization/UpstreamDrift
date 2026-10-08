"""Head gaze stabilisation: eye point, gaze error, schedule, metrics, neck IK."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import gaze
from src.shared.python.motion_matching.range_of_motion import UPPER_RANGES_DEG

pytestmark = pytest.mark.unit

BALL = np.array([0.6, 0.0, 0.021335])


def _aim(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation whose +x axis points from ``eye`` at ``target`` (roll-free)."""
    x = target - eye
    x = x / np.linalg.norm(x)
    y = np.cross([0.0, 0.0, 1.0], x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def test_eye_point_applies_head_frame_to_offset() -> None:
    rot = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    eye = gaze.eye_point(rot, [1.0, 2.0, 3.0], eye_offset=[0.1, 0.0, 0.2])
    assert eye == pytest.approx([1.0, 2.1, 3.2])


def test_default_eye_offset_is_documented_and_inside_head() -> None:
    x, y, z = gaze.EYE_OFFSET_HEAD_M
    assert 0.05 < x < 0.12 and y == 0.0 and 0.08 < z < gaze.HEAD_LENGTH_M


def test_gaze_error_zero_when_forward_axis_points_at_ball() -> None:
    eye = np.array([0.0, 0.0, 1.3])
    rot = _aim(eye, BALL)
    assert gaze.gaze_error(rot, eye, BALL) == pytest.approx(0.0, abs=1e-12)


def test_gaze_error_is_angle_between_forward_and_line_of_sight() -> None:
    eye = np.array([0.0, 0.0, 1.3])
    rot = _aim(eye, BALL) @ Rotation.from_euler("z", 10, degrees=True).as_matrix()
    assert np.degrees(gaze.gaze_error(rot, eye, BALL)) == pytest.approx(10.0)


def test_gaze_error_vectorised_over_frames() -> None:
    eye = np.tile([0.0, 0.0, 1.3], (4, 1))
    rot = np.stack([_aim(eye[0], BALL)] * 4)
    assert gaze.gaze_error(rot, eye, BALL).shape == (4,)


def test_gaze_error_rejects_eye_on_ball() -> None:
    with pytest.raises(ValueError):
        gaze.gaze_error(np.eye(3), BALL, BALL)


def test_min_jerk_boundary_conditions() -> None:
    s = np.array([0.0, 1.0])
    assert gaze.min_jerk(s) == pytest.approx([0.0, 1.0])
    h = 1e-4
    for edge in (0.0, 1.0):
        ds = gaze.min_jerk(np.array([edge + h])) - gaze.min_jerk(np.array([edge]))
        assert abs(float(ds[0])) < 1e-6 or edge == 1.0
    # clamped outside [0, 1]
    assert gaze.min_jerk(np.array([-1.0, 2.0])) == pytest.approx([0.0, 1.0])


def _schedule(t: np.ndarray, **kw: float) -> np.ndarray:
    eye = np.array([0.0, 0.0, 1.3])
    return gaze.gaze_target_direction(t, eye, BALL, [1.0, 0.0, 0.0], t_impact=1.0, **kw)


def test_schedule_holds_ball_until_impact_plus_hold() -> None:
    t = np.array([0.0, 0.5, 1.0, 1.03])
    expected = (BALL - [0.0, 0.0, 1.3]) / np.linalg.norm(BALL - [0.0, 0.0, 1.3])
    for row in _schedule(t):
        assert row == pytest.approx(expected)


def test_schedule_reaches_target_direction_after_release() -> None:
    d = _schedule(np.array([1.0 + 0.03 + 0.35, 5.0]))
    assert d == pytest.approx(np.array([[1.0, 0.0, 0.0]] * 2))


def test_schedule_is_unit_and_c2_at_both_ends() -> None:
    t0, t1 = 1.03, 1.03 + 0.35
    h = 1e-3
    dense = _schedule(np.linspace(t0, t1, 2001))
    dt = (t1 - t0) / 2000
    peak_acc = np.abs(np.diff(dense, 2, axis=0)).max() / dt**2
    for edge in (t0, t1):
        d = _schedule(edge + h * np.arange(-1, 2))
        assert np.linalg.norm(d, axis=1) == pytest.approx(1.0)
        vel = np.abs(d[2] - d[0]).max() / (2 * h)
        acc = np.abs(d[0] - 2 * d[1] + d[2]).max() / h**2
        # velocity and acceleration vanish at the knots (C2), relative to the
        # blend's own peak acceleration
        assert vel < 1e-3 * peak_acc * 0.35
        assert acc < 5e-3 * peak_acc


def test_schedule_rejects_antiparallel_directions() -> None:
    eye = np.array([0.0, 0.0, 1.3])
    with pytest.raises(ValueError):
        gaze.gaze_target_direction(
            np.array([2.0]),
            eye,
            eye + [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            t_impact=1.0,
        )


def test_schedule_validates_timing_arguments() -> None:
    with pytest.raises(ValueError):
        _schedule(np.array([0.0]), t_rel=0.0)
    with pytest.raises(ValueError):
        _schedule(np.array([0.0]), t_hold=-0.1)


def test_impact_index_uses_min_height_near_peak_speed() -> None:
    t = np.linspace(0.0, 1.0, 101)
    z = (t - 0.6) ** 2 * 5 + 0.02  # lowest at t=0.6
    x = np.tanh((t - 0.6) * 20)  # speed peaks at t=0.6
    head = np.column_stack([x, np.zeros_like(t), z])
    assert gaze.impact_index(t, head) == 60


def test_impact_index_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        gaze.impact_index(np.arange(5.0), np.zeros((4, 3)))


def test_head_stability_metrics_stationary_head_has_zero_range() -> None:
    n = 20
    eye0 = np.array([0.0, 0.0, 1.3])
    rot = np.stack([_aim(eye0, BALL)] * n)
    pos = np.tile(eye0 - rot[0] @ gaze.EYE_OFFSET_HEAD_M, (n, 1))
    m = gaze.head_stability_metrics(rot, pos, BALL, 0, n - 1)
    assert m.eye_translation_range_mm == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert m.head_yaw_range_deg == pytest.approx(0.0, abs=1e-9)
    assert m.theta_gaze_rms_deg < 5.0  # eye offset is not exactly on the line
    assert m.frames == n


def test_head_stability_metrics_reports_translation_and_rotation() -> None:
    n = 11
    rot = np.stack(
        [
            Rotation.from_euler("z", a, degrees=True).as_matrix()
            for a in np.linspace(0, 20, n)
        ]
    )
    pos = np.zeros((n, 3))
    pos[:, 0] = np.linspace(0, 0.05, n)
    m = gaze.head_stability_metrics(
        rot, pos, BALL, 0, n - 1, eye_offset=[0.0, 0.0, 0.0]
    )
    assert m.head_yaw_range_deg == pytest.approx(20.0)
    assert m.head_pitch_range_deg == pytest.approx(0.0, abs=1e-9)
    assert m.eye_translation_range_mm[0] > 49.0
    assert m.theta_gaze_max_deg >= m.theta_gaze_rms_deg


def test_head_stability_metrics_validates_window() -> None:
    rot = np.stack([np.eye(3)] * 3)
    with pytest.raises(ValueError):
        gaze.head_stability_metrics(rot, np.zeros((3, 3)), BALL, 2, 1)
    with pytest.raises(ValueError):
        gaze.head_stability_metrics(rot, np.zeros((3, 3)), BALL, 0, 3)


def test_neck_ik_reaches_reachable_direction_exactly() -> None:
    parent = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    want = Rotation.from_euler("XYZ", [5, -25, 30], degrees=True).as_matrix()
    direction = parent @ want @ np.array(gaze.GAZE_AXIS_HEAD)
    res = gaze.neck_ik(parent, direction)
    assert res.residual_deg < 1e-4
    assert not res.clamped.any()
    head = parent @ Rotation.from_euler("XYZ", res.angles_rad).as_matrix()
    assert head @ np.array(gaze.GAZE_AXIS_HEAD) == pytest.approx(direction, abs=1e-5)


def test_neck_ik_stays_inside_rom_and_reports_clamping() -> None:
    parent = np.eye(3)
    direction = np.array([-1.0, 0.1, 0.0])  # looking nearly behind: beyond +-80 deg
    res = gaze.neck_ik(parent, direction)
    lim = np.radians([UPPER_RANGES_DEG[n][1] for n in gaze.NECK_COORDINATES])
    assert (np.abs(res.angles_rad) <= lim + 1e-9).all()
    assert res.clamped.any()
    assert res.residual_deg > 1.0


def test_neck_ik_validates_direction() -> None:
    with pytest.raises(ValueError):
        gaze.neck_ik(np.eye(3), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        gaze.neck_ik(np.eye(2), [1.0, 0.0, 0.0])
