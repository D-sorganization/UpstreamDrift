"""A rendered frame on the exact contact instant, held for the viewer (#15150)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.impact_parameters import ClubheadSeries
from src.shared.python.motion_matching.same_input import InputBundle
from src.shared.python.video_timing.frame_schedule import FrameSchedule
from src.tools.native_viewer_export import contact_frame, core
from src.tools.native_viewer_export.ball import BallPlacement
from src.tools.native_viewer_export.core import ExportSettings, SwingInput

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

RATE_HZ, RADIUS_M, SPEED_MPS = 360.0, 0.021335, 40.0
CONTACT_S = 0.1 + 0.4 / RATE_HZ  # 0.4 of a sample after sample 36
T = np.arange(0.0, 0.3 + 1e-12, 0.001)


def _struck() -> tuple[ClubheadSeries, BallPlacement]:
    t = np.arange(0.0, 0.2, 1.0 / RATE_HZ)
    centre = np.column_stack([SPEED_MPS * (t - CONTACT_S), 0 * t, 0 * t])
    series = ClubheadSeries(
        times_s=t,
        face_center_m=centre,
        velocity_mps=np.gradient(centre, t, axis=0),
        face_normal=np.tile([1.0, 0.0, 0.0], (len(t), 1)),
        toe_axis=np.tile([0.0, -1.0, 0.0], (len(t), 1)),
        grip_axis=np.tile([0.0, 0.0, 1.0], (len(t), 1)),
    )
    return series, BallPlacement((RADIUS_M, 0.0, 0.0), RADIUS_M, "test", 0.05)


def _swing(ball: BallPlacement | None = None) -> SwingInput:
    names, q = ("a", "b"), np.column_stack([T, 2.0 * T])
    bundle = InputBundle(
        spec_bytes=json.dumps({"coordinate_order": list(names)}).encode(),
        coordinate_order=names,
        dt_s=0.001,
        q0=q[0],
        v0=np.zeros(2),
        efforts=np.zeros((len(q) - 1, 2)),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
        provenance={"impact_time_s": 0.1004},
    )
    return SwingInput(bundle, q, "t", "driver", "mujoco", ball=ball)


def test_schedule_includes_an_extra_instant_exactly() -> None:
    window = (0.09, 0.11)
    plain = FrameSchedule(T, 60.0, 0.1, window)
    extra = FrameSchedule(T, 60.0, 0.1, window, extra_times_s=(CONTACT_S, 0.2))
    times = extra.sample_times_s
    assert CONTACT_S in times and 0.2 not in times  # exact; outside is ignored
    assert len(times) == len(plain.sample_times_s) + 1 == extra.n_frames
    k = int(np.nonzero(times == CONTACT_S)[0][0])
    assert extra.interpolate(np.column_stack([T, T]))[k, 0] == pytest.approx(CONTACT_S)


def test_refine_finds_the_fractional_contact_and_keeps_the_ball_to_it(
    monkeypatch,
) -> None:
    series, ball = _struck()
    monkeypatch.setattr(contact_frame.impact_factors, "club_series", lambda s: series)
    swing, when = contact_frame.refine(_swing(ball), 0.1)
    assert when == pytest.approx(CONTACT_S, abs=2e-5) and when not in series.times_s
    assert swing.ball is not None and swing.ball.hide_after_s == when
    assert swing.ball.visible_at(when) and not swing.ball.visible_at(when + 1e-6)
    assert contact_frame.refine(_swing(None), 0.1)[1] is None  # no ball


def test_hold_defaults_differ_between_the_clips() -> None:
    assert contact_frame.hold_count(None, windowed=True) > 0
    assert contact_frame.hold_count(None, windowed=False) == 0
    assert contact_frame.hold_count(5, windowed=False) == 5
    with pytest.raises(ValueError, match="hold"):
        contact_frame.hold_count(-1, windowed=True)
    with pytest.raises(ValueError, match="hold"):
        ExportSettings(contact_hold_frames=-2)


def test_hold_order_repeats_the_contact_row() -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    assert list(contact_frame.hold_order(times, 2.0, 3)) == [0, 1, 2, 2, 2, 2, 3]
    assert list(contact_frame.hold_order(times, 2.0, 0)) == [0, 1, 2, 3]
    assert list(contact_frame.hold_order(times, None, 3)) == [0, 1, 2, 3]


def _clip(window: tuple[float, float] | None, hold: int | None):
    settings = ExportSettings(impact_time_s=CONTACT_S, contact_hold_frames=hold)
    plan = core.ClipPlan("_x", 0.1 if window else 1.0, window)
    return core._resampled(_swing(), None, settings, plan)[0]


def test_impact_window_clip_has_a_frame_on_contact_held_by_default() -> None:
    shown = _clip((CONTACT_S - 0.02, CONTACT_S + 0.02), None)
    times = np.array(shown.sample_times_s)
    hold = contact_frame.hold_count(None, windowed=True)
    assert int(np.count_nonzero(times == CONTACT_S)) == 1 + hold
    assert CONTACT_S - times[times < CONTACT_S].max() <= 1.0 / 600.0 + 1e-9
    k = int(np.nonzero(times == CONTACT_S)[0][0])
    assert np.allclose(shown.q[k : k + 1 + hold, 0], CONTACT_S)  # held pose


def test_full_speed_clip_has_the_frame_and_the_hold_setting_applies() -> None:
    plain = np.array(_clip(None, None).sample_times_s)
    assert int(np.count_nonzero(plain == CONTACT_S)) == 1
    assert (
        int(np.count_nonzero(np.array(_clip(None, 4).sample_times_s) == CONTACT_S)) == 5
    )
