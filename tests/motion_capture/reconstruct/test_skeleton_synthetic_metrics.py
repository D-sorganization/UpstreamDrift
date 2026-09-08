"""Rigid skeleton invariants, synthetic rendering honesty, metric exactness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct import (
    PinholeCamera,
    RenderOptions,
    RigidSkeleton,
    SyntheticScene,
    bone_length_errors,
    camera_pose_error,
    joint_position_errors,
    look_at,
    outlier_flag_scores,
    swing_trajectory,
)
from src.motion_capture.reconstruct.cameras import intrinsics_from_fov
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES, PARENTS
from src.motion_capture.reconstruct.synthetic import (
    load_truth,
    write_synthetic_bundle,
)

pytestmark = pytest.mark.unit


def _rig() -> list[PinholeCamera]:
    k = intrinsics_from_fov(1280, 720, 70.0)
    target = np.array([0.0, 1.0, 0.0])
    positions = {
        "face_on": np.array([0.0, 1.2, 4.0]),
        "down_line": np.array([-4.0, 1.2, 0.0]),
        "high": np.array([2.5, 3.0, 3.0]),
    }
    return [
        PinholeCamera(name, k, look_at(p, target), p, (1280, 720))
        for name, p in positions.items()
    ]


def test_forward_kinematics_keeps_every_segment_length() -> None:
    skel = RigidSkeleton()
    roots, rotations = swing_trajectory(30, 60.0)
    for k in range(30):
        joints = skel.forward(roots[k], rotations[k])
        for i, name in enumerate(JOINT_NAMES[1:], start=1):
            parent = JOINT_NAMES.index(PARENTS[name])  # type: ignore[arg-type]
            assert np.isclose(
                np.linalg.norm(joints[i] - joints[parent]), skel.lengths_m[name]
            )


def test_skeleton_rejects_missing_or_nonpositive_lengths() -> None:
    with pytest.raises(Exception, match="missing"):
        RigidSkeleton({"neck": 0.5})
    bad = dict(RigidSkeleton().lengths_m)
    bad["left_knee"] = 0.0
    with pytest.raises(Exception, match="positive"):
        RigidSkeleton(bad)


def test_render_is_exact_without_corruption_and_lists_every_corruption(
    tmp_path: Path,
) -> None:
    scene = SyntheticScene(_rig(), fps=30.0, n_frames=20)
    clean, truth = scene.render(
        RenderOptions(noise_px=0.0, occlusion_rate=0.0, outlier_rate=0.0)
    )
    joints = np.array(truth.joints_3d_m)
    cam = scene.cameras[0]
    px, _ = cam.project(joints[3])
    row = clean[cam.camera_id]["frames"][3]
    assert np.allclose(np.array(row["keypoints_px"]), px)
    assert truth.outliers[cam.camera_id] == [] and truth.occluded[cam.camera_id] == []

    dirty, truth2 = scene.render(
        RenderOptions(noise_px=0.0, occlusion_rate=0.2, outlier_rate=0.2, seed=3)
    )
    out = truth2.outliers[cam.camera_id]
    occ = truth2.occluded[cam.camera_id]
    assert out and occ
    frame, joint = out[0]
    moved = np.array(dirty[cam.camera_id]["frames"][frame]["keypoints_px"][joint])
    assert np.linalg.norm(moved - px_of(cam, joints, frame, joint)) > 10.0
    assert (
        dirty[cam.camera_id]["frames"][frame]["confidence"][joint] > 0.9
    )  # outliers look confident
    f2, j2 = occ[0]
    assert dirty[cam.camera_id]["frames"][f2]["confidence"][j2] < 0.1

    write_synthetic_bundle(tmp_path, dirty, truth2)
    assert (tmp_path / "observations" / "face_on.json").is_file()
    back = load_truth(tmp_path / "truth.json")
    assert back == truth2
    payload = json.loads(
        (tmp_path / "observations" / "high.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "view-observations/1.0.0"
    assert payload["detector_layout"]["keypoint_names"] == list(JOINT_NAMES)


def px_of(cam: PinholeCamera, joints: np.ndarray, frame: int, joint: int) -> np.ndarray:
    px, _ = cam.project(joints[frame])
    return px[joint]


def test_render_options_contracts() -> None:
    with pytest.raises(Exception, match="occlusion_rate"):
        RenderOptions(occlusion_rate=1.0)
    with pytest.raises(Exception, match="noise_px"):
        RenderOptions(noise_px=-1)


def test_metrics_are_exact_on_truth_and_count_missing() -> None:
    cams = _rig()
    r, t = cams[0].rotation_world_from_camera, cams[0].position_m
    zero = camera_pose_error(r, t, r, t)
    assert zero.rotation_deg == pytest.approx(0.0) and zero.translation_m == 0.0
    tilted = look_at(t, np.array([0.0, 1.0, 0.0]) + np.array([0.1, 0.0, 0.0]))
    assert 0.0 < camera_pose_error(tilted, t + 0.05, r, t).rotation_deg < 5.0

    lengths = RigidSkeleton().bone_lengths()
    err = bone_length_errors({**lengths, "neck": lengths["neck"] * 1.1}, lengths)
    assert err["neck"] == pytest.approx(0.1) and err["left_knee"] == 0.0
    assert np.isnan(bone_length_errors({}, lengths)["neck"])

    truth = np.zeros((4, 15, 3))
    est = truth.copy()
    est[1, 2] = [0.0, 0.0, 0.5]
    est[2, 3] = np.nan
    pe = joint_position_errors(est, truth)
    assert pe.compared == 59 and pe.missing == 1
    assert pe.max_m == pytest.approx(0.5) and pe.mean_m == pytest.approx(0.5 / 59)

    scores = outlier_flag_scores([(0, 1), (0, 2)], [(0, 1), (5, 5)])
    assert (scores.true_positives, scores.false_positives, scores.false_negatives) == (
        1,
        1,
        1,
    )
    assert scores.precision == 0.5 and scores.recall == 0.5
