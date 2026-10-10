"""Gaze-schedule neck tracking in the forward-dynamics replay (OSV-3c, #11729)."""

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
COORDS = ("frame", "NeckInputX", "NeckInputY", "NeckInputZ")
NECK_COLS = [1, 2, 3]
# A fixed non-identity parent rotation: the FK-based solve must not assume any
# Euler convention or that the parent frame is the torso.
PARENT = Rotation.from_euler("ZYX", [0.4, -0.3, 0.2]).as_matrix()


def _club_xz(k: int) -> tuple[float, float]:
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


class StubKin:
    coordinate_order = COORDS

    def __init__(self) -> None:
        self.head_t = np.array([-0.1, -0.6, 1.4])

    def body_poses(self, q, frames):  # noqa: ANN001
        k = int(round(q[0] * 1000))
        out = {}
        for f in frames:
            if f == gr.CLUB_FRAME:
                x, z = _club_xz(k)
                out[f] = (np.eye(3), np.array([x, 0.0, z]))
            elif f == gr.GRIP_FRAME:
                x, z = _club_xz(k)
                out[f] = (np.eye(3), np.array([x, 0.0, z + 1.1]))
            elif f == gr.HEAD_FRAME:
                rot = Rotation.from_euler("XYZ", q[1:4]).as_matrix()
                out[f] = (PARENT @ rot, self.head_t.copy())
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


def _q(neck: np.ndarray | None = None) -> np.ndarray:
    q = np.zeros((N, 4))
    q[:, 0] = np.arange(N) / 1000.0
    if neck is not None:
        q[:, 1:] = neck
    return q


def _head_dirs(kin: StubKin, q: np.ndarray, axis: np.ndarray) -> np.ndarray:
    rots, _ = gr.frame_poses(kin, q, gr.HEAD_FRAME)
    return rots @ axis


def _angles_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0)))


AXIS = np.array([0.8, 0.6, 0.0])


def _reachable_case() -> tuple[StubKin, np.ndarray, np.ndarray]:
    kin = StubKin()
    s = np.linspace(0.0, 1.0, N)
    truth = np.stack([0.3 * s, -0.4 * s, 0.5 * np.sin(3 * s)], axis=1)
    directions = _head_dirs(kin, _q(truth), AXIS)
    return kin, _q(0.2 * truth), directions


def test_solve_hits_reachable_directions_and_changes_only_neck() -> None:
    kin, q, directions = _reachable_case()
    res = gt.solve_neck_schedule(kin, q, directions, AXIS)
    achieved = _head_dirs(kin, res.q, AXIS)
    assert _angles_deg(achieved, directions).max() < 0.01
    assert res.max_residual_deg < 0.01
    assert res.clamped_frames == 0
    other = [c for c in range(q.shape[1]) if c not in NECK_COLS]
    assert np.array_equal(res.q[:, other], q[:, other])
    assert set(res.as_dict()) == {
        "clamped_frames",
        "max_residual_deg",
        "rms_residual_deg",
    }


def test_solve_clamps_unreachable_direction_inside_rom() -> None:
    kin, q, directions = _reachable_case()
    # straight behind the address line of sight is outside the neck range
    directions = directions.copy()
    directions[10:] = -_head_dirs(kin, _q(), AXIS)[10:]
    res = gt.solve_neck_schedule(kin, q, directions, AXIS)
    assert res.clamped_frames > 0
    assert res.max_residual_deg > 1.0
    for col, name in zip(NECK_COLS, gaze.NECK_COORDINATES, strict=True):
        lo, hi = np.radians(UPPER_RANGES_DEG[name])
        assert res.q[:, col].min() >= lo - 1e-9
        assert res.q[:, col].max() <= hi + 1e-9


def test_solve_preconditions() -> None:
    kin, q, directions = _reachable_case()

    class NoNeck(StubKin):
        coordinate_order = ("frame", "a", "b", "c")

    with pytest.raises(ValueError, match="NeckInputX"):
        gt.solve_neck_schedule(NoNeck(), q, directions, AXIS)
    with pytest.raises(ValueError, match="frames"):
        gt.solve_neck_schedule(kin, q, directions[:-1], AXIS)
    with pytest.raises(ValueError, match="prior_weight"):
        gt.solve_neck_schedule(kin, q, directions, AXIS, prior_weight=-1.0)
    with pytest.raises(ValueError):
        gt.solve_neck_schedule(kin, q[0], directions, AXIS)
    bad = directions.copy()
    bad[3] = 0.0
    with pytest.raises(ValueError, match="nonzero"):
        gt.solve_neck_schedule(kin, q, bad, AXIS)


def _plan(kin: StubKin) -> gr.GazePlan:
    return gr.plan_gaze(kin, _q(), TIMES, ground_height_m=0.0, face_offset_m=0.0)


def _solved_to_schedule(kin: StubKin, plan: gr.GazePlan) -> np.ndarray:
    q = _q()
    for _ in range(6):  # eye offset moves with the head: fixed-point iterate
        directions, *_ = gr.schedule_directions(plan, kin, q, TIMES)
        q = gt.solve_neck_schedule(kin, q, directions, plan.gaze_axis_head).q
    return q


def test_schedule_tracking_windows_and_none_for_empty() -> None:
    kin = StubKin()
    plan = _plan(kin)
    q = _solved_to_schedule(kin, plan)
    out = gt.schedule_tracking(plan, kin, q, TIMES)
    windows = ("address_to_impact", "hold", "release", "after_release")
    assert sum(out[w]["frames"] for w in windows) == N
    for w in windows:
        if out[w]["frames"]:
            assert out[w]["max_deg"] < 0.5
    # the 0.59 s trajectory ends before the release completes
    assert out["after_release"]["frames"] == 0
    assert out["after_release"]["rms_deg"] is None
    assert out["after_release"]["max_deg"] is None
    assert out["address_to_impact_metrics"]["theta_gaze_rms_deg"] < 0.5


def test_apply_fd_neck_ik_is_identity_and_gaze_improves() -> None:
    kin = StubKin()
    q_ref = _q()
    q_track = _q()
    same, solve = gt.apply_fd_neck("ik", FakeLane(), kin, q_ref, q_track)
    assert solve is None and np.array_equal(same, q_track)

    new, solve = gt.apply_fd_neck("gaze", FakeLane(), kin, q_ref, q_track)
    assert solve is not None and new.shape == q_track.shape
    for col, name in zip(NECK_COLS, gaze.NECK_COORDINATES, strict=True):
        lo, hi = np.radians(UPPER_RANGES_DEG[name])
        assert new[:, col].min() >= lo - 1e-9 and new[:, col].max() <= hi + 1e-9
    plan = _plan(kin)
    before = gt.schedule_tracking(plan, kin, q_track, TIMES)
    after = gt.schedule_tracking(plan, kin, new, TIMES)
    assert after["release"]["rms_deg"] < before["release"]["rms_deg"]
    assert after["hold"]["rms_deg"] < 0.5  # eye offset moves with the head
    with pytest.raises(ValueError, match="ik"):
        gt.apply_fd_neck("foo", FakeLane(), kin, q_ref, q_track)


def test_fd_head_gaze_report_available_and_unavailable() -> None:
    kin = StubKin()
    q = _q()
    new, solve = gt.apply_fd_neck("gaze", FakeLane(), kin, q, q)
    rep = gt.fd_head_gaze_report(FakeLane(), kin, q, new, new, solve, "gaze")
    assert rep["available"] is True and rep["neck_reference"] == "gaze_schedule"
    assert rep["neck_solve"] is not None and "tracked_reference" in rep
    assert "replay" in rep and "plan" in rep and rep["note"]
    ik = gt.fd_head_gaze_report(FakeLane(), kin, q, q, q, None, "ik")
    assert ik["neck_reference"] == "ik" and ik["neck_solve"] is None

    class NoHead(StubKin):
        def body_poses(self, q, frames):  # noqa: ANN001
            raise ValueError("Unknown body Head")

    bad = gt.fd_head_gaze_report(FakeLane(), NoHead(), q, q, q, None, "ik")
    assert bad["available"] is False and "Unknown body" in bad["reason"]


def test_cli_fd_neck_option() -> None:
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    parser = build_parser()
    assert parser.parse_args([]).fd_neck == "ik"
    assert parser.parse_args(["--fd-neck", "gaze"]).fd_neck == "gaze"
    with pytest.raises(SystemExit):
        parser.parse_args(["--fd-neck", "foo"])
