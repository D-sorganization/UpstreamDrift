"""#12042 slice 7 follow-up: impact-windowed turn split terms."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import turn_split as ts
from src.shared.python.motion_matching.tour_capture_contract import MARKER_SEGMENTS

pytestmark = pytest.mark.unit
CLUB = MARKER_SEGMENTS["club"]
RATE = 360.0


def _swing(frames: int = 540, impact_s: float = 1.1) -> tuple[np.ndarray, ...]:
    """Club markers on a circle whose speed peaks (and height bottoms) at
    ``impact_s``; one body marker that never moves."""
    t = np.arange(frames) / RATE
    angle = 6.0 * np.exp(-(((t - impact_s) / 0.25) ** 2)).cumsum() / RATE
    labels = ("BackLeft", *CLUB)
    pts = np.zeros((frames, len(labels), 3))
    for j, radius in enumerate(np.linspace(0.6, 1.4, len(CLUB)), start=1):
        phase = angle - angle[np.argmin(np.abs(t - impact_s))] - np.pi / 2
        pts[:, j, 0] = radius * np.cos(phase)
        pts[:, j, 2] = 1.5 + radius * np.sin(phase)
    return t, pts, np.ones((frames, len(labels)), dtype=bool), labels


def test_impact_time_comes_from_the_club_markers() -> None:
    t, pts, valid, labels = _swing()
    valid[400:420, 1] = False  # a gap in one club marker is interpolated
    assert ts.capture_impact_time(t, pts, valid, labels) == pytest.approx(
        1.1, abs=2.5 / RATE
    )


def test_impact_time_needs_club_markers() -> None:
    t, pts, valid, labels = _swing()
    with pytest.raises(ValueError, match="club marker"):
        ts.capture_impact_time(t, pts[:, :1], valid[:, :1], labels[:1])
    valid[:, 1:] = False
    valid[:5, 1:] = True
    with pytest.raises(ValueError, match="club marker"):
        ts.capture_impact_time(t, pts, valid, labels)


def test_window_factors_hold_then_taper_to_zero() -> None:
    t = np.arange(0, 2.0, 1 / RATE)
    f = ts.window_factors(t, 1.0, 0.05)
    assert np.all(f[t <= 1.0] == 1.0) and np.all(f[t >= 1.05] == 0.0)
    mid = (t > 1.0) & (t < 1.05)
    assert np.all(np.diff(f[mid]) < 0) and np.all((f[mid] > 0) & (f[mid] < 1))
    np.testing.assert_array_equal(ts.window_factors(t, 1.0, 0.0), t <= 1.0)
    with pytest.raises(ValueError, match="taper"):
        ts.window_factors(t, 1.0, -0.1)


def test_windowed_thorax_targets_scale_and_drop() -> None:
    attach = {
        "BackLeft": ("Spine", (0.0, 0.09, 0.3)),
        "BackRight": ("Spine", (0.0, -0.09, 0.3)),
    }
    pts = np.zeros((3, 2, 3))
    pts[:, 0, 1], pts[:, 1, 1] = 0.09, -0.09
    valid = np.ones((3, 2), dtype=bool)
    out = ts.thorax_axis_targets(
        pts, valid, ("BackLeft", "BackRight"), attach, 0.4, factors=[1.0, 0.5, 0.0]
    )
    assert out is not None and out[2] is None
    assert out[0]["Spine"][2] == pytest.approx(0.4)
    assert out[1]["Spine"][2] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="factors"):
        ts.thorax_axis_targets(
            pts, valid, ("BackLeft", "BackRight"), attach, 0.4, factors=[1.0]
        )


def test_windowed_shoulder_weights_taper_to_one() -> None:
    labels = ("LShoulderBack", "RShoulderBack", "BackLeft")
    out = ts.shoulder_girdle_weights_per_frame(labels, 5.0, [1.0, 0.5, 0.0])
    assert out[0] == {"LShoulderBack": 5.0, "RShoulderBack": 5.0}
    assert out[1] == {"LShoulderBack": 3.0, "RShoulderBack": 3.0}
    assert out[2] == {}


def test_lane_window_uses_detected_impact() -> None:
    from src.shared.python.motion_matching.pipeline.lane import Lane

    t, pts, valid, labels = _swing()
    labels = ("BackLeft", "BackRight", "LShoulderBack", "RShoulderBack", *CLUB)
    pts = np.concatenate([pts[:, :1], np.zeros((len(t), 3, 3)), pts[:, 1:]], 1)
    pts[:, 0, 1], pts[:, 1, 1] = 0.09, -0.09
    valid = np.ones(pts.shape[:2], dtype=bool)
    lane = SimpleNamespace(points=pts, valid=valid, labels=labels, times=t)
    attach = {
        "BackLeft": ("Spine", (0.0, 0.09, 0.3)),
        "BackRight": ("Spine", (0.0, -0.09, 0.3)),
    }
    Lane.set_turn_split(
        lane, attach, 0.3, 5.0, window=ts.SplitWindow(thorax=True, girdle=True)
    )
    impact = lane.turn_split_impact_s
    assert impact == pytest.approx(1.1, abs=2.5 / RATE)
    late = int(np.searchsorted(t, impact + 0.06))
    assert lane.thorax_targets[late] is None and lane.thorax_targets[0] is not None
    assert lane.split_marker_weights_per_frame[late] == {}
    assert lane.split_marker_weights_per_frame[0]["LShoulderBack"] == 5.0
    assert ts.lane_split_weights_per_frame(lane) is lane.split_marker_weights_per_frame
    report = ts.turn_split_report(lane)
    assert report["window"]["impact_s"] == pytest.approx(impact)
    assert report["window"]["thorax"] and report["window"]["shoulder_girdle"]


def test_full_window_keeps_frame_constant_weights() -> None:
    from src.shared.python.motion_matching.pipeline.lane import Lane

    t, pts, valid, labels = _swing(60)
    lane = SimpleNamespace(points=pts, valid=valid, labels=labels, times=t)
    Lane.set_turn_split(lane, {}, 0.0, 5.0)
    assert lane.split_marker_weights_per_frame is None
    assert ts.lane_split_weights_per_frame(lane) is None


def test_cli_window_options() -> None:
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    args = build_parser().parse_args([])
    assert ts.split_window_from_args(args) == ts.SplitWindow()
    args = build_parser().parse_args(
        ["--thorax-window", "impact", "--shoulder-girdle-window", "impact"]
    )
    window = ts.split_window_from_args(args)
    assert window.thorax and window.girdle
    assert window.taper_s == ts.IMPACT_TAPER_S
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--split-taper-s", "-1"])


def test_solve_trajectory_honours_per_frame_marker_weights() -> None:
    from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK

    seen: list = []

    class Fake:
        coordinate_order = ("a",)
        _validate_trajectory_options = BaseFullBodyIK._validate_trajectory_options

        def solve_pose(self, targets, mask, start, **kwargs):
            seen.append(kwargs.get("marker_weights"))
            return SimpleNamespace(q=np.zeros(1), marker_rms_m=0.0)

    per_frame = [{"A": 5.0}, {}]
    BaseFullBodyIK.solve_trajectory(
        Fake(),
        np.zeros((2, 1, 3)),
        np.ones((2, 1), dtype=bool),
        np.zeros(1),
        ground=None,
        marker_weights={"A": 2.0},
        marker_weights_per_frame=per_frame,
    )
    assert seen == per_frame
    with pytest.raises(ValueError, match="marker_weights_per_frame"):
        BaseFullBodyIK.solve_trajectory(
            Fake(),
            np.zeros((2, 1, 3)),
            np.ones((2, 1), dtype=bool),
            np.zeros(1),
            ground=None,
            marker_weights_per_frame=per_frame[:1],
        )
