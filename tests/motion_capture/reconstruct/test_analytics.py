"""Swing kinematics from 3-D joints agree with the synthetic motion's known angles."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct import RigidSkeleton, swing_trajectory
from src.motion_capture.reconstruct.analytics import (
    detect_events,
    line_turn_deg,
    summarize_swing,
    swing_series,
)
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES

pytestmark = pytest.mark.unit

FPS = 120.0


def _joints(n: int = 240, amplitude: float = 1.2) -> np.ndarray:
    skel = RigidSkeleton()
    roots, rotations = swing_trajectory(n, FPS, amplitude_rad=amplitude)
    return np.stack([skel.forward(roots[k], rotations[k]) for k in range(n)])


def test_line_turn_is_zero_at_start_and_unwraps() -> None:
    t = np.linspace(0, 2 * np.pi, 50)
    left = np.zeros((50, 3))
    right = np.column_stack([np.sin(t), np.zeros(50), np.cos(t)])
    turn = line_turn_deg(left, right)
    assert turn[0] == pytest.approx(0.0)
    assert turn[-1] == pytest.approx(360.0, abs=1e-6)  # unwrapped, not wrapped


def test_turn_angles_match_the_synthetic_rotations() -> None:
    joints = _joints()
    series = swing_series(joints, FPS)
    # swing_trajectory: pelvis turns 0.6 * torso, shoulders 0.6 + 0.4 = 1.0 * torso,
    # torso amplitude = 1.2 * 0.5 rad -> shoulders 34.4 deg, pelvis 20.6 deg.
    assert np.max(np.abs(series.shoulder_turn_deg)) == pytest.approx(34.4, abs=0.6)
    assert np.max(np.abs(series.pelvis_turn_deg)) == pytest.approx(20.6, abs=0.6)
    assert np.max(np.abs(series.x_factor_deg)) == pytest.approx(13.8, abs=0.6)
    assert (series.hand_speed_mps >= 0).all() and series.hand_speed_mps.max() > 0.5


def test_events_and_tempo_are_frame_indices_in_order() -> None:
    joints = _joints()
    series = swing_series(joints, FPS)
    events = detect_events(series, FPS)
    assert 0 <= events.address_frame <= events.top_frame < events.peak_speed_frame
    assert events.peak_speed_frame <= events.finish_frame < joints.shape[0]
    assert events.downswing_s > 0 and events.tempo_ratio is not None


def test_summary_is_robust_to_a_single_spiked_frame() -> None:
    joints = _joints()
    clean, _ = summarize_swing(joints, FPS)
    spiked = joints.copy()
    lw, rw = JOINT_NAMES.index("left_wrist"), JOINT_NAMES.index("right_wrist")
    spiked[100, lw] += [0.3, 0.0, 0.0]  # a 30 cm one-frame jump of one wrist
    spiked[100, rw] += [0.3, 0.0, 0.0]
    dirty, _ = summarize_swing(spiked, FPS)
    assert dirty.peak_hand_speed_mps == pytest.approx(
        clean.peak_hand_speed_mps, rel=0.05
    )
    assert dirty.peak_hand_speed_uncertainty_mps > 0
    assert dirty.max_shoulder_turn_deg == pytest.approx(clean.max_shoulder_turn_deg)


def test_contracts() -> None:
    with pytest.raises(Exception, match="T>=3"):
        swing_series(np.zeros((2, 15, 3)), FPS)
    with pytest.raises(Exception, match="joint missing"):
        swing_series(np.zeros((5, 3, 3)), FPS, joint_names=("a", "b", "c"))
