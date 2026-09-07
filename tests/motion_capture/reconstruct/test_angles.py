"""Joint-angle series from the 3-D fit and their 2-D analogues (#9682)."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.analytics import (
    SwingEvents,
    angle_stats,
    flexion_deg,
    joint_angles,
)
from src.motion_capture.reconstruct.analytics2d import joint_angles_2d
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES

pytestmark = pytest.mark.unit


def _pose(elbow_bend_deg: float, knee_bend_deg: float, tilt_deg: float) -> np.ndarray:
    """One frame of the 15-joint skeleton with prescribed bends (metres)."""
    j = np.zeros((len(JOINT_NAMES), 3))
    ix = {n: i for i, n in enumerate(JOINT_NAMES)}
    tilt = np.radians(tilt_deg)
    j[ix["mid_hip"]] = [0, 1.0, 0]
    j[ix["neck"]] = j[ix["mid_hip"]] + 0.5 * np.array([0, np.cos(tilt), np.sin(tilt)])
    j[ix["nose"]] = j[ix["neck"]] + [0, 0.15, 0]
    j[ix["left_hip"]], j[ix["right_hip"]] = [-0.15, 1.0, 0], [0.15, 1.0, 0]
    j[ix["left_shoulder"]] = j[ix["neck"]] + [-0.2, 0, 0]
    j[ix["right_shoulder"]] = j[ix["neck"]] + [0.2, 0, 0]
    bend = np.radians(elbow_bend_deg)
    for side in ("left", "right"):
        sh = j[ix[f"{side}_shoulder"]]
        el = sh + [0, -0.3, 0]
        j[ix[f"{side}_elbow"]] = el
        j[ix[f"{side}_wrist"]] = el + 0.3 * np.array([0, -np.cos(bend), np.sin(bend)])
        hip = j[ix[f"{side}_hip"]]
        kb = np.radians(knee_bend_deg)
        knee = hip + [0, -0.45, 0]
        j[ix[f"{side}_knee"]] = knee
        j[ix[f"{side}_ankle"]] = knee + 0.45 * np.array([0, -np.cos(kb), np.sin(kb)])
    return j


def test_flexion_and_trunk_angles_match_prescribed_pose() -> None:
    frames = np.stack([_pose(0, 0, 0), _pose(40, 25, 30), _pose(90, 0, -10)])
    angles = joint_angles(frames)
    assert set(angles) == {
        "left_elbow_flexion",
        "right_elbow_flexion",
        "left_knee_flexion",
        "right_knee_flexion",
        "trunk_forward_tilt",
        "trunk_side_bend",
        "lead_arm_shoulder_deg",
    }
    np.testing.assert_allclose(angles["left_elbow_flexion"], [0, 40, 90], atol=1e-6)
    np.testing.assert_allclose(angles["right_knee_flexion"], [0, 25, 0], atol=1e-6)
    np.testing.assert_allclose(angles["trunk_forward_tilt"], [0, 30, 10], atol=1e-6)
    np.testing.assert_allclose(angles["trunk_side_bend"], [0, 0, 0], atol=1e-6)
    assert np.all(
        (angles["lead_arm_shoulder_deg"] >= 0)
        & (angles["lead_arm_shoulder_deg"] <= 180)
    )
    straight = flexion_deg(
        np.array([[0, 1.0, 0]]), np.zeros((1, 3)), np.array([[0, -1.0, 0]])
    )
    assert straight[0] == pytest.approx(0.0)


def test_angle_stats_report_event_values_and_range() -> None:
    series = {"x": np.array([10.0, np.nan, 30.0, 40.0])}
    events = SwingEvents(
        address_frame=0,
        top_frame=1,
        peak_speed_frame=2,
        finish_frame=9,
        backswing_s=0.1,
        downswing_s=0.05,
        tempo_ratio=2.0,
    )
    stats = angle_stats(series, events)["x"]
    assert stats["address"] == 10.0 and np.isnan(stats["top"]) and stats["peak"] == 30.0
    assert np.isnan(stats["finish"])  # outside the series
    assert (stats["min"], stats["max"]) == (10.0, 40.0)


def test_image_plane_flexion_from_a_2d_payload() -> None:
    names = [
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "left_hip",
        "left_knee",
        "left_ankle",
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
    ]
    straight = [[0, 0], [0, 100], [0, 200], [50, 0], [50, 100], [50, 200]] * 2
    bent = [[0, 0], [0, 100], [100, 100], [50, 0], [50, 100], [50, 200]] * 2
    payload = {
        "fps": 10.0,
        "frames_total": 3,
        "detector_layout": {"name": "t", "keypoint_names": names},
        "frames": [
            {"time_s": 0.0, "keypoints_px": straight, "confidence": [0.9] * 12},
            {"time_s": 0.1, "keypoints_px": bent, "confidence": [0.9] * 12},
        ],
    }
    angles = joint_angles_2d(payload)
    np.testing.assert_allclose(
        angles["left_elbow_flexion_image"][:2], [0.0, 90.0], atol=1e-6
    )
    np.testing.assert_allclose(
        angles["left_knee_flexion_image"][:2], [0.0, 0.0], atol=1e-6
    )
    assert np.isnan(angles["left_elbow_flexion_image"][2])  # frame without a pose
