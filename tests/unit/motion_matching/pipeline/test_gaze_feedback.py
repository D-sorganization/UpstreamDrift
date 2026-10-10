"""Closed-loop gaze neck in the forward-dynamics replay (OSV-3d, #11729).

The open-loop ``--fd-neck gaze`` reference is solved on the *tracked* torso, so
replay torso error reaches the head. ``gaze-closed`` re-solves the neck target
from the *simulated* torso while the replay runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching import gaze
from src.shared.python.motion_matching.pipeline import gaze_residual as gr
from src.shared.python.motion_matching.pipeline import gaze_tracking as gt
from src.shared.python.motion_matching.range_of_motion import UPPER_RANGES_DEG

pytestmark = pytest.mark.unit

N = 60
TIMES = np.linspace(0.0, 0.59, N)
COORDS = ("frame", "NeckInputX", "NeckInputY", "NeckInputZ", "torso_yaw")
NECK_COLS = [1, 2, 3]
TORSO = 4
PARENT = Rotation.from_euler("ZYX", [0.4, -0.3, 0.2]).as_matrix()


def _club_xz(k: int) -> tuple[float, float]:
    k = int(np.clip(k, 0, N - 1))
    top, bottom, finish = (
        np.array([-0.8, 1.0]),
        np.array([0.0, 0.02]),
        np.array([0.8, 1.0]),
    )
    if k <= 25:
        p = bottom + (top - bottom) * (k / 25)
    elif k <= 40:
        p = top + (bottom - top) * ((k - 25) / 15) ** 2
    else:
        u = (k - 40) / 19
        p = bottom + (finish - bottom) * (1 - (1 - u) ** 2)
    return float(p[0]), float(p[1])


class TorsoKin:
    """Head parent turns with a torso yaw coordinate, so torso error moves the head."""

    coordinate_order = COORDS

    def body_poses(self, q, frames):  # noqa: ANN001
        k = int(round(q[0] * 1000))
        out = {}
        for f in frames:
            if f in (gr.CLUB_FRAME, gr.GRIP_FRAME):
                x, z = _club_xz(k)
                lift = 1.1 if f == gr.GRIP_FRAME else 0.0
                out[f] = (np.eye(3), np.array([x, 0.0, z + lift]))
            elif f == gr.HEAD_FRAME:
                torso = Rotation.from_euler("Z", q[TORSO]).as_matrix()
                neck = Rotation.from_euler("XYZ", q[1:4]).as_matrix()
                out[f] = (torso @ PARENT @ neck, np.array([-0.1, -0.6, 1.4]))
            else:
                raise ValueError(f"Unknown body {f}")
        return out


class FakeLane:
    gaze_plan = None
    gaze_weight = 0.0
    gaze_face_offset_m = 0.0
    times = TIMES
    rate_hz = 100.0

    class ground:  # noqa: N801
        height_m = 0.0


def _q() -> np.ndarray:
    q = np.zeros((N, len(COORDS)))
    q[:, 0] = np.arange(N) / 1000.0
    return q


def _feedback(kin: TorsoKin, q_track: np.ndarray, **kw) -> gt.GazeNeckFeedback:
    plan = gr.plan_gaze(kin, _q(), TIMES, ground_height_m=0.0, face_offset_m=0.0)
    return gt.GazeNeckFeedback(plan, kin, TIMES, **kw)


def _head_error_deg(fb: gt.GazeNeckFeedback, kin, t: float, row) -> float:  # noqa: ANN001
    directions, *_ = gr.schedule_directions(fb.plan, kin, row[None], [t])
    rot = kin.body_poses(row, [gr.HEAD_FRAME])[gr.HEAD_FRAME][0]
    a = rot @ fb.plan.gaze_axis_head
    d = directions[0] / np.linalg.norm(directions[0])
    return float(np.degrees(np.arccos(np.clip(a @ d, -1.0, 1.0))))


def _target(fb: gt.GazeNeckFeedback, t: float, q, q_t) -> np.ndarray:  # noqa: ANN001
    q_out, _ = fb(t, q, q_t, np.zeros_like(q_t))
    return q_out


def test_hook_resolves_the_neck_on_the_simulated_torso() -> None:
    kin = TorsoKin()
    q_track, _ = gt.apply_fd_neck("gaze", FakeLane(), kin, _q(), _q())
    fb = _feedback(kin, q_track)
    k = 30
    t = float(TIMES[k])
    sim_q = q_track[k].copy()
    sim_q[TORSO] = np.radians(12.0)  # the replay torso lags the reference
    open_loop = sim_q.copy()
    open_loop[NECK_COLS] = q_track[k, NECK_COLS]
    target = _target(fb, t, sim_q, q_track[k].copy())
    # only the neck columns of the reference are replaced
    other = [c for c in range(len(COORDS)) if c not in NECK_COLS]
    assert np.array_equal(target[other], q_track[k, other])
    closed = sim_q.copy()
    closed[NECK_COLS] = target[NECK_COLS]
    # gaze looks down at the ball, so 12 deg of torso yaw moves it ~4.5 deg
    assert _head_error_deg(fb, kin, t, open_loop) > 3.0
    assert _head_error_deg(fb, kin, t, closed) < 0.5
    for col, name in zip(NECK_COLS, gaze.NECK_COORDINATES, strict=True):
        lo, hi = np.radians(UPPER_RANGES_DEG[name])
        assert lo - 1e-9 <= target[col] <= hi + 1e-9


def test_hook_matches_the_open_loop_neck_on_the_tracked_torso() -> None:
    kin = TorsoKin()
    q_track, _ = gt.apply_fd_neck("gaze", FakeLane(), kin, _q(), _q())
    fb = _feedback(kin, q_track)
    k = 12
    target = _target(fb, float(TIMES[k]), q_track[k].copy(), q_track[k].copy())
    assert np.allclose(target[NECK_COLS], q_track[k, NECK_COLS], atol=np.radians(1.0))


def test_hook_updates_once_per_period_and_counts() -> None:
    kin = TorsoKin()
    q_track, _ = gt.apply_fd_neck("gaze", FakeLane(), kin, _q(), _q())
    fb = _feedback(kin, q_track, update_period_s=0.01)
    row = q_track[20].copy()
    first = _target(fb, 0.2, row, row.copy())
    moved = row.copy()
    moved[TORSO] = 0.3
    # inside the same period the held correction is reused (rate still zero)
    held = _target(fb, 0.205, moved, row.copy())
    assert np.array_equal(held[NECK_COLS], first[NECK_COLS])
    fresh = _target(fb, 0.21, moved, row.copy())
    assert not np.allclose(fresh[NECK_COLS], first[NECK_COLS])
    stats = fb.as_dict()
    assert stats["updates"] == 2
    assert set(stats) == {
        "updates",
        "clamped_updates",
        "max_residual_deg",
        "rms_residual_deg",
        "update_period_s",
    }


def test_hook_feeds_the_correction_rate_forward() -> None:
    kin = TorsoKin()
    q_track, _ = gt.apply_fd_neck("gaze", FakeLane(), kin, _q(), _q())
    fb = _feedback(kin, q_track, update_period_s=0.01)
    row = q_track[20].copy()
    v_t = np.full(len(COORDS), 0.5)
    q0, v0 = fb(0.2, row, row.copy(), v_t.copy())
    assert np.array_equal(v0, v_t)  # no rate before a second update
    moved = row.copy()
    moved[TORSO] = 0.05
    q1, v1 = fb(0.21, moved, row.copy(), v_t.copy())
    rate = (q1[NECK_COLS] - q0[NECK_COLS]) / 0.01
    assert np.abs(rate).max() > 0.1
    assert np.allclose(v1[NECK_COLS], v_t[NECK_COLS] + rate)
    other = [c for c in range(len(COORDS)) if c not in NECK_COLS]
    assert np.array_equal(v1[other], v_t[other])
    # between updates the correction is extrapolated at that rate
    q2, _ = fb(0.215, moved, row.copy(), v_t.copy())
    assert np.allclose(q2[NECK_COLS], q1[NECK_COLS] + 0.005 * rate)


def test_hook_preconditions() -> None:
    kin = TorsoKin()
    q_track, _ = gt.apply_fd_neck("gaze", FakeLane(), kin, _q(), _q())
    with pytest.raises(ValueError, match="update_period_s"):
        _feedback(kin, q_track, update_period_s=0.0)
    fb = _feedback(kin, q_track)
    with pytest.raises(ValueError, match="finite"):
        _target(fb, 0.1, np.full(len(COORDS), np.nan), q_track[0].copy())
    with pytest.raises(ValueError, match="shape"):
        _target(fb, 0.1, q_track[0][:-1], q_track[0].copy())


def test_closed_mode_feedforward_equals_gaze_and_report_label() -> None:
    kin = TorsoKin()
    lane = FakeLane()
    gaze_q, gaze_solve = gt.apply_fd_neck("gaze", lane, kin, _q(), _q())
    closed_q, closed_solve = gt.apply_fd_neck("gaze-closed", lane, kin, _q(), _q())
    assert np.array_equal(gaze_q, closed_q) and closed_solve is not None
    assert gt.fd_neck_feedback("ik", lane, kin, _q()) is None
    assert gt.fd_neck_feedback("gaze", lane, kin, _q()) is None
    fb = gt.fd_neck_feedback("gaze-closed", lane, kin, _q())
    assert isinstance(fb, gt.GazeNeckFeedback)
    assert fb.update_period_s == pytest.approx(1.0 / lane.rate_hz)
    rep = gt.fd_head_gaze_report(
        lane, kin, _q(), closed_q, closed_q, closed_solve, "gaze-closed", feedback=fb
    )
    assert rep["neck_reference"] == "gaze_schedule_closed_loop"
    assert rep["feedback"]["updates"] == 0
    assert "simulated torso" in rep["note"]


def test_controller_reference_hook_replaces_the_target(monkeypatch) -> None:  # noqa: ANN001
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs
    from src.shared.python.motion_matching import tracking_controller as tc

    seen: list[np.ndarray] = []

    def fake_ct(sim, q, v, q_t, v_t, a_t, gains, com_ref):  # noqa: ANN001
        seen.append(np.concatenate([q_t, v_t]))
        return np.zeros_like(q)

    monkeypatch.setattr(fs, "_computed_torque", fake_ct)

    class Sim:
        nv = 2

    times = np.array([0.0, 1.0])
    ref = np.array([[0.0, 0.0], [1.0, 1.0]])
    calls: list[tuple[float, np.ndarray]] = []

    def hook(t, q, q_t, v_t):  # noqa: ANN001
        calls.append((t, q.copy()))
        out = q_t.copy()
        out[1] = 7.0
        return out, v_t + 1.0

    ctl = tc.tracking_controller(
        Sim(), times, ref, omega_rad_s=10.0, reference_hook=hook
    )
    ctl(0.5, np.array([0.1, 0.2]), np.zeros(2))
    assert calls and calls[0][0] == 0.5
    assert np.allclose(seen[-1], [0.5, 7.0, 2.0, 2.0])

    def bad(t, q, q_t, v_t):  # noqa: ANN001
        return q_t[:1], v_t

    ctl = tc.tracking_controller(
        Sim(), times, ref, omega_rad_s=10.0, reference_hook=bad
    )
    with pytest.raises(ValueError, match="reference_hook"):
        ctl(0.5, np.zeros(2), np.zeros(2))


def test_cli_accepts_gaze_closed() -> None:
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    args = build_parser().parse_args(["--fd-neck", "gaze-closed"])
    assert args.fd_neck == "gaze-closed"
