"""Camera placement from the golfer alone, then refined by the joint fit."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct import (
    PinholeCamera,
    RenderOptions,
    RigidSkeleton,
    SyntheticScene,
    bone_length_errors,
    look_at,
)
from src.motion_capture.reconstruct.bundle import (
    BundleOptions,
    bundle_adjust,
    observations_from_views,
)
from src.motion_capture.reconstruct.cameras import intrinsics_from_fov
from src.motion_capture.reconstruct.initialize import (
    initialize_cameras,
    relative_pose,
    subject_frame,
)
from src.motion_capture.reconstruct.skeleton import DEFAULT_LENGTHS_M

pytestmark = pytest.mark.unit


def _rig() -> list[PinholeCamera]:
    k = intrinsics_from_fov(1920, 1200, 70.0)
    target = np.array([0.0, 1.0, 0.0])
    spots = {
        "face_on": np.array([0.0, 1.2, 4.0]),
        "down_line": np.array([-4.0, 1.2, 0.0]),
        "high": np.array([2.5, 3.0, 3.0]),
    }
    return [
        PinholeCamera(n, k, look_at(p, target), p, (1920, 1200))
        for n, p in spots.items()
    ]


def _angles_and_distances(cams):
    axes = [c.rotation_world_from_camera[:, 2] for c in cams]
    ang = [np.degrees(np.arccos(np.clip(axes[0] @ a, -1, 1))) for a in axes[1:]]
    dist = [np.linalg.norm(c.position_m - cams[0].position_m) for c in cams[1:]]
    return np.array(ang), np.array(dist)


def test_relative_pose_recovers_rotation_and_direction() -> None:
    cams = _rig()
    rng = np.random.default_rng(0)
    pts = rng.uniform([-0.6, 0.2, -0.6], [0.6, 1.9, 0.6], (300, 3))
    px0, _ = cams[0].project(pts)
    px1, _ = cams[1].project(pts)
    px0 += rng.normal(0, 1.0, px0.shape)
    px1 += rng.normal(0, 1.0, px1.shape)
    r, t_unit, inliers = relative_pose(px0, px1, cams[0].matrix, cams[1].matrix)
    assert inliers > 250
    # truth: T_ref_from_other = R0^T R1, R0^T (p1 - p0)
    r_true = cams[0].rotation_world_from_camera.T @ cams[1].rotation_world_from_camera
    t_true = cams[0].rotation_world_from_camera.T @ (
        cams[1].position_m - cams[0].position_m
    )
    # five-point RANSAC at 1 px noise: about a degree; the joint fit refines it
    assert np.degrees(np.arccos((np.trace(r.T @ r_true) - 1) / 2)) < 1.5
    assert (
        np.degrees(
            np.arccos(np.clip(t_unit @ (t_true / np.linalg.norm(t_true)), -1, 1))
        )
        < 1.5
    )


def test_initialisation_from_joints_then_fit_matches_relative_geometry() -> None:
    cams = _rig()
    lengths = {k: v * 1.04 for k, v in DEFAULT_LENGTHS_M.items()}
    scene = SyntheticScene(cams, RigidSkeleton(lengths), fps=60.0, n_frames=60)
    views, truth = scene.render(
        RenderOptions(noise_px=1.0, occlusion_rate=0.02, outlier_rate=0.01, seed=4)
    )
    obs = observations_from_views(views, [c.camera_id for c in cams])
    init = initialize_cameras(
        obs,
        [c.matrix for c in cams],
        [c.image_size_px for c in cams],
        anchor=("neck", lengths["neck"]),
    )
    assert init.ok and all(p.inliers >= 40 for p in init.pairs)
    start = subject_frame(init.cameras, obs)
    ang_t, dist_t = _angles_and_distances(cams)
    ang_i, dist_i = _angles_and_distances(start)
    # a start, not an answer: joints cluster on a small body, so the pairwise
    # geometry is coarse; the joint fit below is what has to be accurate
    assert np.all(np.abs(ang_i - ang_t) < 12.0)
    assert np.all(np.abs(dist_i / dist_t - 1.0) < 0.3)
    result = bundle_adjust(
        start,
        obs,
        length_prior_m=DEFAULT_LENGTHS_M,
        options=BundleOptions(scale_anchor=("neck", lengths["neck"])),
    )
    ang_f, dist_f = _angles_and_distances(result.cameras)
    assert np.all(np.abs(ang_f - ang_t) < 0.5)
    assert np.all(np.abs(dist_f / dist_t - 1.0) < 0.02)
    assert result.rms_px < 2.0
    assert (
        max(bone_length_errors(result.bone_lengths_m, truth.bone_lengths_m).values())
        < 0.03
    )
    # the subject frame: hip-to-neck at address points up
    hip = result.joints_3d_m[0, 0]
    neck = result.joints_3d_m[0, 1]
    up = (neck - hip) / np.linalg.norm(neck - hip)
    assert up[1] > 0.99


def test_too_few_correspondences_is_refused() -> None:
    cams = _rig()
    with pytest.raises(Exception, match="eight"):
        relative_pose(
            np.zeros((3, 2)), np.zeros((3, 2)), cams[0].matrix, cams[1].matrix
        )
