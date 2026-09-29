"""MediaPipe/BODY_25 payloads map onto the 15-joint reconstruct layout honestly."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.layouts import to_reconstruct_layout
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from src.shared.python.pose_estimation.mediapipe_estimator import MediaPipeEstimator

pytestmark = pytest.mark.unit


def _mediapipe_payload(n: int = 3):
    names = [
        MediaPipeEstimator.LANDMARK_MAP[i]
        for i in sorted(MediaPipeEstimator.LANDMARK_MAP)
    ]
    rng = np.random.default_rng(0)
    frames = []
    for t in range(n):
        px = rng.uniform(0, 1000, (len(names), 2))
        conf = np.full(len(names), 0.9)
        if t == 1:
            conf[names.index("right_hip")] = 0.0  # one hip unobserved
        frames.append(
            {
                "camera_id": "x",
                "time_s": t / 30,
                "keypoints_px": px.tolist(),
                "confidence": conf.tolist(),
            }
        )
    return {
        "view": "v",
        "fps": 30.0,
        "frames_total": n,
        "detector_layout": {"name": "mediapipe_pose_33", "keypoint_names": names},
        "frames": frames,
        "provenance": {"estimator": "mediapipe"},
    }


def test_mediapipe_maps_and_derives_midpoints_with_min_confidence() -> None:
    src = _mediapipe_payload()
    out = to_reconstruct_layout(src)
    assert out["detector_layout"]["keypoint_names"] == list(JOINT_NAMES)
    names = src["detector_layout"]["keypoint_names"]
    row0, src0 = out["frames"][0], src["frames"][0]
    lh, rh = names.index("left_hip"), names.index("right_hip")
    expected = 0.5 * (
        np.array(src0["keypoints_px"][lh]) + np.array(src0["keypoints_px"][rh])
    )
    assert np.allclose(row0["keypoints_px"][JOINT_NAMES.index("mid_hip")], expected)
    assert row0["confidence"][JOINT_NAMES.index("mid_hip")] == pytest.approx(0.9)
    nose = names.index("nose")
    assert row0["keypoints_px"][JOINT_NAMES.index("nose")] == src0["keypoints_px"][nose]
    # frame 1: one hip unobserved -> mid_hip unobserved, not half-invented
    row1 = out["frames"][1]
    assert row1["confidence"][JOINT_NAMES.index("mid_hip")] == 0.0
    assert row1["keypoints_px"][JOINT_NAMES.index("mid_hip")] == [0.0, 0.0]
    assert out["provenance"]["source_layout"] == "mediapipe_pose_33"
    assert len(src["frames"][0]["keypoints_px"]) == 33  # source untouched


def test_missing_source_joints_are_reported() -> None:
    src = _mediapipe_payload(1)
    src["detector_layout"]["keypoint_names"] = [
        n if n != "left_knee" else "knee_l"
        for n in src["detector_layout"]["keypoint_names"]
    ]
    with pytest.raises(Exception, match="lacks joints"):
        to_reconstruct_layout(src)


def test_body25_uses_its_own_mid_hip_and_neck() -> None:
    from src.shared.python.pose_estimation.openpose_dnn_estimator import (
        OpenPoseDnnEstimator,
    )

    names = [
        OpenPoseDnnEstimator.LANDMARK_MAP[i]
        for i in sorted(OpenPoseDnnEstimator.LANDMARK_MAP)
    ]
    rng = np.random.default_rng(1)
    px = rng.uniform(0, 1000, (len(names), 2))
    conf = np.full(len(names), 0.7)
    payload = {
        "view": "v",
        "fps": 30.0,
        "frames_total": 1,
        "detector_layout": {"name": "openpose_body25", "keypoint_names": names},
        "frames": [
            {
                "camera_id": "x",
                "time_s": 0.0,
                "keypoints_px": px.tolist(),
                "confidence": conf.tolist(),
            }
        ],
    }
    out = to_reconstruct_layout(payload)
    row = out["frames"][0]
    assert (
        row["keypoints_px"][JOINT_NAMES.index("mid_hip")]
        == px[names.index("mid_hip")].tolist()
    )
    assert (
        row["keypoints_px"][JOINT_NAMES.index("neck")]
        == px[names.index("neck")].tolist()
    )
    assert out["provenance"]["derived_joints"] == {}


def _payload(names: list[str], px: list[list[float]], conf: list[float]) -> dict:
    return {
        "view": "v",
        "fps": 30.0,
        "frames_total": 1,
        "detector_layout": {"name": "test_layout", "keypoint_names": names},
        "frames": [
            {
                "camera_id": "x",
                "time_s": 0.0,
                "keypoints_px": px,
                "confidence": conf,
            }
        ],
    }


def test_coco17_derives_mid_hip_and_neck_from_shoulders_and_hips() -> None:
    from src.shared.python.pose_estimation.rtmpose_onnx_estimator import (
        RtmposeOnnxEstimator,
    )

    names = [
        RtmposeOnnxEstimator.LANDMARK_MAP[i]
        for i in sorted(RtmposeOnnxEstimator.LANDMARK_MAP)
    ]
    px = [[10.0, 10.0]] * len(names)
    px[names.index("left_hip")] = [100.0, 200.0]
    px[names.index("right_hip")] = [300.0, 200.0]
    px[names.index("left_shoulder")] = [150.0, 100.0]
    px[names.index("right_shoulder")] = [250.0, 100.0]
    conf = [0.9] * len(names)
    payload = _payload(names, px, conf)

    out = to_reconstruct_layout(payload)
    row = out["frames"][0]
    # Hand-computed midpoints: hips -> (200, 200), shoulders -> (200, 100).
    assert row["keypoints_px"][JOINT_NAMES.index("mid_hip")] == [200.0, 200.0]
    assert row["keypoints_px"][JOINT_NAMES.index("neck")] == [200.0, 100.0]
    assert row["confidence"][JOINT_NAMES.index("mid_hip")] == pytest.approx(0.9)
    assert out["provenance"]["derived_joints"] == {
        "mid_hip": ["left_hip", "right_hip"],
        "neck": ["left_shoulder", "right_shoulder"],
    }


def test_coco17_unobserved_hip_leaves_mid_hip_unobserved() -> None:
    from src.shared.python.pose_estimation.rtmpose_onnx_estimator import (
        RtmposeOnnxEstimator,
    )

    names = [
        RtmposeOnnxEstimator.LANDMARK_MAP[i]
        for i in sorted(RtmposeOnnxEstimator.LANDMARK_MAP)
    ]
    px = [[10.0, 10.0]] * len(names)
    px[names.index("left_hip")] = [100.0, 200.0]
    px[names.index("right_hip")] = [300.0, 200.0]
    px[names.index("left_shoulder")] = [150.0, 100.0]
    px[names.index("right_shoulder")] = [250.0, 100.0]
    conf = [0.9] * len(names)
    conf[names.index("right_hip")] = 0.0  # one hip unobserved this frame
    payload = _payload(names, px, conf)

    row = to_reconstruct_layout(payload)["frames"][0]
    mid_hip = JOINT_NAMES.index("mid_hip")
    assert row["confidence"][mid_hip] == 0.0
    assert row["keypoints_px"][mid_hip] == [0.0, 0.0]  # never half-invented


def test_halpe26_uses_its_own_hip_and_neck_keypoints() -> None:
    from src.shared.python.pose_estimation.rtmpose_onnx_estimator import (
        RtmposeOnnxEstimator,
    )

    estimator = RtmposeOnnxEstimator(keypoint_set="halpe26")
    names = [estimator.LANDMARK_MAP[i] for i in sorted(estimator.LANDMARK_MAP)]
    px = [[10.0, 10.0]] * len(names)
    px[names.index("hip")] = [210.0, 205.0]
    px[names.index("neck")] = [205.0, 95.0]
    px[names.index("left_hip")] = [100.0, 200.0]
    px[names.index("right_hip")] = [300.0, 200.0]
    conf = [0.8] * len(names)
    payload = _payload(names, px, conf)

    out = to_reconstruct_layout(payload)
    row = out["frames"][0]
    # Halpe reports its own hip (mid-hip) and neck: no midpoint derivation.
    assert row["keypoints_px"][JOINT_NAMES.index("mid_hip")] == [210.0, 205.0]
    assert row["keypoints_px"][JOINT_NAMES.index("neck")] == [205.0, 95.0]
    assert out["provenance"]["derived_joints"] == {}
