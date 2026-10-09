"""Soft gaze residual: plan, axis targets, receipt block, lane wiring (OSV-3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching import gaze
from src.shared.python.motion_matching.pipeline import gaze_residual as gr

pytestmark = pytest.mark.unit

N = 60
TIMES = np.linspace(0.0, 0.59, N)
COORDS = ("NeckInputX", "NeckInputY", "NeckInputZ")


def _club_xz(k: int) -> tuple[float, float]:
    """Address, backswing to the top (k=25), accelerating downswing to impact
    (k=40, peak speed), decelerating follow-through."""
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
    """Rigid scene: head fixed near the ball, club swinging through the ball."""

    coordinate_order = COORDS

    def __init__(self) -> None:
        self.head_t = np.array([-0.1, -0.6, 1.4])

    def body_poses(self, q, frames):  # noqa: ANN001
        k = int(round(q[0] * 1000))  # frame index smuggled through q[0]
        out = {}
        for f in frames:
            if f == gr.CLUB_FRAME:
                x, z = _club_xz(k)
                out[f] = (np.eye(3), np.array([x, 0.0, z]))
            elif f == gr.GRIP_FRAME:
                x, z = _club_xz(k)
                out[f] = (np.eye(3), np.array([x, 0.0, z + 1.1]))
            else:
                out[f] = (np.eye(3), self.head_t.copy())
        return out


def _q() -> np.ndarray:
    q = np.zeros((N, 3))
    q[:, 0] = np.arange(N) / 1000.0
    return q


def test_plan_uses_shared_ball_function_and_clubhead_impact() -> None:
    plan = gr.plan_gaze(StubKin(), _q(), TIMES, ground_height_m=0.0, face_offset_m=0.0)
    assert plan.ball_m[2] == pytest.approx(0.021335)
    assert plan.impact_index == 40  # lowest club point near peak speed
    assert plan.target_dir == pytest.approx([1.0, 0.0, 0.0])


def test_axis_targets_follow_schedule_and_weight() -> None:
    kin, q = StubKin(), _q()
    plan = gr.plan_gaze(kin, q, TIMES, ground_height_m=0.0, face_offset_m=0.0)
    targets = gr.gaze_axis_targets(plan, kin, q, TIMES, 7.0)
    assert len(targets) == N
    axis, direction, weight = targets[0][gr.HEAD_FRAME]
    assert weight == 7.0 and np.asarray(axis) == pytest.approx(plan.gaze_axis_head)
    eye = gaze.eye_point(np.eye(3), kin.head_t)
    sight = (plan.ball_m - eye) / np.linalg.norm(plan.ball_m - eye)
    assert direction == pytest.approx(sight)
    last = np.asarray(targets[-1][gr.HEAD_FRAME][1])
    assert last @ plan.target_dir > np.asarray(direction) @ plan.target_dir
    with pytest.raises(ValueError):
        gr.gaze_axis_targets(plan, kin, q, TIMES, -1.0)


def test_merge_axis_targets_unions_per_frame() -> None:
    base = [{"LS": 1}, None]
    extra = [{"Head": 2}, {"Head": 3}]
    assert gr.merge_axis_targets(base, extra) == [
        {"LS": 1, "Head": 2},
        {"Head": 3},
    ]
    assert gr.merge_axis_targets(None, extra) == extra
    with pytest.raises(ValueError):
        gr.merge_axis_targets([None], extra)


def test_report_records_weight_metrics_and_note() -> None:
    kin, q = StubKin(), _q()
    plan = gr.plan_gaze(kin, q, TIMES, ground_height_m=0.0, face_offset_m=0.0)
    off = gr.gaze_report(plan, kin, q, TIMES, 0.0, COORDS)
    on = gr.gaze_report(plan, kin, q, TIMES, 5.0, COORDS)
    assert off["gaze_weight"] == 0.0 and not off["regularised"]
    assert on["regularised"] and "not measured" in on["note"]
    # gaze axis is calibrated at address, so a static head has zero error
    assert off["address_to_impact"]["theta_gaze_rms_deg"] == pytest.approx(
        0.0, abs=1e-9
    )
    assert off["address_to_impact_nominal_axis"]["theta_gaze_rms_deg"] > 0.0
    assert set(off["neck_range_deg"]) == set(COORDS)
    assert off["plan"]["t_release_s"] == 0.35


def test_lane_default_leaves_axis_targets_unchanged() -> None:
    from src.shared.python.motion_matching.pipeline.lane import Lane

    lane = Lane.__new__(Lane)
    lane.gaze_weight = 0.0
    lane.anthropometric = False
    assert lane.axis_targets(None, np.zeros(3)) is None  # type: ignore[arg-type]


def test_cli_gaze_weight_defaults_to_marker_faithful_and_validates() -> None:
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    parser = build_parser()
    assert parser.parse_args([]).gaze_weight == 0.0
    assert parser.parse_args(["--gaze-weight", "10"]).gaze_weight == 10.0
    with pytest.raises(SystemExit):
        parser.parse_args(["--gaze-weight", "-1"])


def test_lane_plans_gaze_once_from_marker_faithful_pass() -> None:
    from src.shared.python.motion_matching.pipeline.lane import Lane

    kin, q = StubKin(), _q()
    lane = Lane.__new__(Lane)
    lane.gaze_weight = 4.0
    lane.anthropometric = False
    lane.times = TIMES
    lane.gaze_face_offset_m = 0.0
    lane.gaze_axis_targets_cache = None
    lane.gaze_plan = None

    class Ground:
        height_m = 0.0

    lane.ground = Ground()  # type: ignore[assignment]
    calls: list[int] = []

    def fake_solve(_kin, _q0, _frames, _axis):  # noqa: ANN001
        calls.append(1)
        return q, []

    lane._solve = fake_solve  # type: ignore[method-assign]
    # smoothing needs the capture rate; the stub trajectory is 100 Hz
    lane.__class__ = type("L", (Lane,), {"rate_hz": 100.0})
    first = lane.axis_targets(kin, q[0])
    second = lane.axis_targets(kin, q[0])
    assert len(calls) == 1 and first == second
    assert first[0][gr.HEAD_FRAME][2] == 4.0
    assert lane.gaze_plan.impact_index > 0


def test_receipt_reports_unavailable_when_the_model_has_no_head_frame() -> None:
    class NoHead(StubKin):
        def body_poses(self, q, frames):  # noqa: ANN001
            raise ValueError("Unknown body Head")

    class FakeLane:
        gaze_plan = None
        gaze_weight = 0.0
        gaze_face_offset_m = 0.0
        times = TIMES

        class ground:  # noqa: N801
            height_m = 0.0

    block = gr.head_gaze_receipt(FakeLane(), NoHead(), _q())
    assert block["available"] is False and "Unknown body" in block["reason"]
