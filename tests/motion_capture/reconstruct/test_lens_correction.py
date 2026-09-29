"""Distorted source pixels must reach the pinhole fitter in its ideal pixel space."""

import copy
import json
from dataclasses import replace

import cv2
import numpy as np
import pytest

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reconstruct.lens import LensCorrection
from src.shared.python.pose_estimation.observations import CameraIntrinsics

pytestmark = pytest.mark.unit


def test_distorted_pixels_recover_independent_ideal_projection_without_mutating_input():
    matrix = np.array([[900.0, 0, 640], [0, 880, 360], [0, 0, 1]])
    distortion = np.array([0.15, -0.04, 0.002, -0.001, 0.01])
    points = np.array([[-0.5, -0.3, 2.0], [0.6, 0.4, 2.5], [0.1, -0.5, 1.8]])
    projected, _ = cv2.projectPoints(
        points, np.zeros(3), np.zeros(3), matrix, distortion
    )
    payload = {
        "width": 1280,
        "height": 720,
        "frames": [
            {
                "time_s": 0.0,
                "keypoints_px": projected.reshape(-1, 2).tolist(),
                "confidence": [1.0, 1.0, 0.0],
            }
        ],
    }
    before = copy.deepcopy(payload)
    lens = LensCorrection(CameraIntrinsics(matrix, distortion), (1280, 720))
    corrected = lens.apply(payload)
    expected = (points / points[:, 2:]) @ matrix.T
    assert np.asarray(corrected["frames"][0]["keypoints_px"]) == pytest.approx(
        expected[:, :2], abs=1e-6
    )
    assert corrected["frames"][0]["confidence"] == [1.0, 1.0, 0.0]
    assert payload == before
    assert lens.apply(corrected) == corrected
    with pytest.raises(ValueError, match="different lens"):
        LensCorrection(CameraIntrinsics(matrix, distortion * 2), (1280, 720)).apply(
            corrected
        )


def test_camera_roundtrip_keeps_lens_metadata_but_projection_remains_ideal():
    matrix = np.array([[900.0, 0, 640], [0, 880, 360], [0, 0, 1]])
    distortion = np.array([0.15, -0.04, 0.002, -0.001, 0.01])
    camera = PinholeCamera(
        "front", matrix, np.eye(3), np.zeros(3), (1280, 720), distortion=distortion
    )
    restored = PinholeCamera.from_calibration(camera.to_calibration())
    assert restored.distortion == pytest.approx(distortion)
    pixels, _ = restored.project(np.array([[0.5, 0.3, 2.0]]))
    assert pixels[0] == pytest.approx([865, 492])


@pytest.mark.parametrize("count", [5, 8, 12, 14])
def test_original_video_and_expert_tracks_use_full_camera_distortion(count):
    from src.motion_capture.reconstruct.overlay3d import project_track
    from src.motion_capture.reference.registration import project_reference_to_camera

    matrix = np.array([[900.0, 0, 640], [0, 880, 360], [0, 0, 1]])
    coefficients = np.array(
        [
            0.15,
            -0.04,
            0.002,
            -0.001,
            0.01,
            0.003,
            -0.001,
            0.0005,
            0.002,
            -0.001,
            -0.003,
            0.001,
            0.004,
            -0.003,
        ]
    )[:count]
    camera = PinholeCamera(
        "front", matrix, np.eye(3), np.zeros(3), (1280, 720), distortion=coefficients
    )
    points = np.array([[[0.5, 0.3, 2.0]], [[0.2, -0.1, 1.0]]])
    expected, _ = cv2.projectPoints(
        points.reshape(-1, 3), np.zeros(3), np.zeros(3), matrix, coefficients
    )
    expected = expected.reshape(2, 1, 2)
    pixels, visible = project_track(points, camera)
    assert visible.all()
    assert pixels == pytest.approx(expected, abs=1e-6)
    for record in (camera, camera.to_calibration()):
        expert, mask = project_reference_to_camera(
            points, np.ones((2, 1), dtype=bool), record
        )
        assert mask.all()
        assert expert == pytest.approx(expected, abs=1e-6)


def test_distorted_capture_reconstructs_and_retains_raw_camera_calibration(tmp_path):
    from src.motion_capture.reconstruct.__main__ import lab_rig
    from src.motion_capture.reconstruct.pipeline import reconstruct_session
    from src.motion_capture.reconstruct.synthetic import (
        RenderOptions,
        SyntheticScene,
        write_synthetic_bundle,
    )

    distortion = np.array([0.15, -0.04, 0.002, -0.001, 0.01])
    cameras = [replace(camera, distortion=distortion) for camera in lab_rig()]
    scene = SyntheticScene(cameras, n_frames=24)
    views, truth = scene.render(
        RenderOptions(noise_px=0, occlusion_rate=0, outlier_rate=0)
    )
    joints = scene.joints_3d()
    ideal = copy.deepcopy(views)
    for camera in cameras:
        for index, row in enumerate(views[camera.camera_id]["frames"]):
            pixels, _ = cv2.projectPoints(
                camera.camera_from_world(joints[index]),
                np.zeros(3),
                np.zeros(3),
                camera.matrix,
                distortion,
            )
            row["keypoints_px"] = pixels.reshape(-1, 2).tolist()
    write_synthetic_bundle(tmp_path, views, truth)
    original_bytes = {
        path: path.read_bytes() for path in (tmp_path / "observations").glob("*.json")
    }
    summary = reconstruct_session(
        tmp_path, start_cameras=cameras, scale_anchor=("neck", 0.5)
    )
    assert summary.rms_px < 0.01
    assert all(path.read_bytes() == data for path, data in original_bytes.items())
    saved = json.loads((tmp_path / "reconstruct/reconstruction.json").read_text())
    for record in saved["cameras"]:
        assert record["intrinsics"]["distortion"] == pytest.approx(distortion)
        view = record["camera_id"]
        cleaned = json.loads(
            (tmp_path / f"reconstruct/observations/{view}.json").read_text()
        )
        assert cleaned["lens_correction"]["space"] == "ideal-pixels"
        for actual, expected in zip(
            cleaned["frames"], ideal[view]["frames"], strict=True
        ):
            valid = np.asarray(expected["confidence"]) > 0
            assert np.asarray(actual["keypoints_px"])[valid] == pytest.approx(
                np.asarray(expected["keypoints_px"])[valid], abs=1e-5
            )
