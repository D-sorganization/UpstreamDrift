"""Image-space fit of the golfer to 1..3 camera views (#9794)."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.__main__ import lab_rig
from src.motion_capture.reconstruct.model import FitOptions
from src.motion_capture.reconstruct.model.fit2d import (
    fit_trajectory_2d,
    initial_root_2d,
    map_landmarks_2d,
)
from src.motion_capture.reconstruct.model.golfer import (
    GOLFER_LANDMARK_MAP,
    golfer_model,
)
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES

pytestmark = pytest.mark.unit

FPS = 60.0
FRAMES = 30


def _truth() -> tuple[np.ndarray, np.ndarray]:
    """A smooth golfer motion at the lab rig's target; ``(q, landmarks (T,L,3))``."""
    model = golfer_model()
    t = np.arange(FRAMES) / FPS
    q = np.zeros((FRAMES, model.n_dof))
    q[:, 0] = 0.05 * np.sin(2 * np.pi * 0.5 * t)
    q[:, 1] = 1.0
    q[:, model.dof_slice("left_scapula").start] = 0.3 * np.sin(2 * np.pi * 0.5 * t)
    q[:, model.dof_slice("left_elbow")] = (
        0.6 + 0.3 * np.sin(2 * np.pi * 0.7 * t)[:, None]
    )
    q[:, model.dof_slice("right_elbow")] = 0.4
    q[:, model.dof_slice("torso")] = 0.2 * np.sin(2 * np.pi * 0.4 * t)[:, None]
    q[:, model.dof_slice("left_knee")] = 0.3
    q[:, model.dof_slice("right_knee")] = 0.3
    return q, model.landmarks(q)


def _observe(landmarks: np.ndarray, cameras, noise_px: float = 1.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    t, n_l = landmarks.shape[:2]
    kp = np.zeros((t, len(cameras), n_l, 2))
    conf = np.ones((t, len(cameras), n_l))
    for c, camera in enumerate(cameras):
        px, in_front = camera.project(landmarks.reshape(-1, 3))
        kp[:, c] = px.reshape(t, n_l, 2) + rng.normal(0, noise_px, (t, n_l, 2))
        conf[:, c] = in_front.reshape(t, n_l)
    return kp, conf


@pytest.mark.parametrize("views,limit_mm", [((0, 1, 2), 10.0), ((0, 1), 20.0)])
def test_multi_view_image_fit_recovers_the_landmarks(
    views: tuple[int, ...], limit_mm: float
) -> None:
    model = golfer_model()
    q_true, truth = _truth()
    cameras = [lab_rig()[i] for i in views]
    kp, conf = _observe(truth, cameras)
    assert conf.all(), "the rig must see every landmark"
    fit = fit_trajectory_2d(
        model, kp, conf, cameras, FPS, options=FitOptions(max_iterations=80)
    )
    assert fit.rms_px is not None and fit.rms_px < 2.5
    assert np.isnan(fit.rms_m) and fit.residual_px is not None
    err = np.linalg.norm(fit.landmarks_m - truth, axis=2)
    assert 1000 * np.sqrt(np.mean(err**2)) < limit_mm
    assert fit.velocity_violations == 0


@pytest.mark.timeout(180)
def test_single_view_image_fit_matches_the_image_and_reports_depth() -> None:
    model = golfer_model()
    q_true, truth = _truth()
    camera = lab_rig()[0]
    kp, conf = _observe(truth, [camera])
    fit = fit_trajectory_2d(
        model, kp, conf, [camera], FPS, options=FitOptions(max_iterations=80)
    )
    assert fit.rms_px is not None and fit.rms_px < 2.0
    # Depth is the weakly observed direction: report it, do not demand it.
    depth_dir = camera.rotation_world_from_camera[:, 2]
    depth_err = (fit.landmarks_m - truth) @ depth_dir
    in_plane = np.linalg.norm(
        (fit.landmarks_m - truth) - depth_err[..., None] * depth_dir, axis=2
    )
    assert np.sqrt(np.mean(in_plane**2)) < 0.03
    assert np.isfinite(depth_err).all()


def test_root_initialisation_and_landmark_mapping() -> None:
    model = golfer_model()
    _, truth = _truth()
    cameras = lab_rig()
    kp, conf = _observe(truth, cameras, noise_px=0.0)
    root = initial_root_2d(kp, conf, cameras)
    np.testing.assert_allclose(root, truth[:, 0], atol=1e-3)
    single = initial_root_2d(kp[:, :1], conf[:, :1], cameras[:1], depth_m=4.0)
    assert np.isfinite(single).all()
    # Frames without a confident root are interpolated from their neighbours.
    conf2 = conf.copy()
    conf2[5, :, 0] = 0.0
    root2 = initial_root_2d(kp, conf2, cameras)
    assert np.linalg.norm(root2[5] - truth[5, 0]) < 0.02
    # Reconstruct-joint pixels (mid_hip first) map onto the golfer landmarks.
    k = len(JOINT_NAMES)
    joints_px = np.arange(2 * 1 * k * 2, dtype=float).reshape(2, 1, k, 2) / 10
    joints_px = np.repeat(joints_px, 1, axis=1)
    joint_conf = np.ones((2, 1, k))
    joint_conf[0, 0, JOINT_NAMES.index("nose")] = 0.0
    lm, lm_conf = map_landmarks_2d(
        GOLFER_LANDMARK_MAP, model, joints_px, joint_conf, JOINT_NAMES
    )
    assert lm.shape == (2, 1, len(model.landmark_names), 2)
    nose = model.landmark_names.index("nose")
    assert lm_conf[0, 0, nose] == 0.0 and lm_conf[1, 0, nose] == 1.0
    pelvis = model.landmark_names.index("pelvis")
    np.testing.assert_allclose(
        lm[:, :, pelvis], joints_px[:, :, JOINT_NAMES.index("mid_hip")]
    )


def test_image_fit_preconditions() -> None:
    model = golfer_model()
    cameras = lab_rig()[:1]
    n_l = len(model.landmark_names)
    with pytest.raises(Exception, match="keypoints"):
        fit_trajectory_2d(
            model, np.zeros((3, 1, n_l, 3)), np.ones((3, 1, n_l)), cameras, 60.0
        )
    with pytest.raises(Exception, match="one camera per view"):
        fit_trajectory_2d(
            model, np.zeros((3, 2, n_l, 2)), np.ones((3, 2, n_l)), cameras, 60.0
        )
    with pytest.raises(Exception, match="confident"):
        fit_trajectory_2d(
            model, np.zeros((3, 1, n_l, 2)), np.zeros((3, 1, n_l)), cameras, 60.0
        )
