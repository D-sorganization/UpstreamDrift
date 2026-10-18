"""Joint camera + skeleton fit on the synthetic harness (C3/C4 acceptance)."""

from __future__ import annotations

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
)
from src.motion_capture.reconstruct.bundle import (
    BundleOptions,
    Observations,
    bundle_adjust,
    observations_from_views,
)
from src.motion_capture.reconstruct.cameras import intrinsics_from_fov
from src.motion_capture.reconstruct.skeleton import DEFAULT_LENGTHS_M, JOINT_NAMES

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


def _perturbed(cams: list[PinholeCamera], deg: float, metres: float, seed: int = 0):
    """Every camera but the first moved by a known amount (the 'previous take')."""
    rng = np.random.default_rng(seed)
    out = [cams[0]]
    for cam in cams[1:]:
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        from src.motion_capture.reconstruct.skeleton import rotation_from_axis_angle

        r = (
            rotation_from_axis_angle(axis, np.radians(deg))
            @ cam.rotation_world_from_camera
        )
        t = cam.position_m + rng.normal(size=3) / np.sqrt(3) * metres
        out.append(PinholeCamera(cam.camera_id, cam.matrix, r, t, cam.image_size_px))
    return out


def _wrong_lengths() -> dict[str, float]:
    """A subject 6 % taller in every segment than the default prior."""
    return {k: v * 1.06 for k, v in DEFAULT_LENGTHS_M.items()}


def _scene(options: RenderOptions, n_frames: int = 40):
    cams = _rig()
    scene = SyntheticScene(
        cams, RigidSkeleton(_wrong_lengths()), fps=60.0, n_frames=n_frames
    )
    views, truth = scene.render(options)
    obs = observations_from_views(views, [c.camera_id for c in cams])
    return cams, obs, truth


def test_recovers_cameras_bone_lengths_and_joints_from_a_perturbed_start() -> None:
    cams, obs, truth = _scene(
        RenderOptions(noise_px=1.0, occlusion_rate=0.02, outlier_rate=0.0, seed=3)
    )
    start = _perturbed(cams, deg=3.0, metres=0.15)
    # One measured length on the subject fixes the scale gauge (the golfer's
    # hip-to-neck distance measured once with a tape); everything else is learned.
    opts = BundleOptions(scale_anchor=("neck", truth.bone_lengths_m["neck"]))
    result = bundle_adjust(start, obs, length_prior_m=DEFAULT_LENGTHS_M, options=opts)
    assert result.converged or result.iterations >= 10
    assert result.rms_px < 2.0 and result.initial_rms_px > result.rms_px
    for est, true in zip(result.cameras[1:], cams[1:], strict=True):
        err = camera_pose_error(
            est.rotation_world_from_camera,
            est.position_m,
            true.rotation_world_from_camera,
            true.position_m,
        )
        assert err.rotation_deg < 0.5 and err.translation_m < 0.03
    lengths = bone_length_errors(result.bone_lengths_m, truth.bone_lengths_m)
    assert max(lengths.values()) < 0.02  # learned the 6 % taller subject
    pe = joint_position_errors(result.joints_3d_m, np.array(truth.joints_3d_m))
    assert pe.mean_m is not None and pe.mean_m < 0.01 and pe.missing == 0
    assert result.cameras[0] is start[0]  # gauge camera untouched


def test_gross_outliers_are_rejected_after_convergence() -> None:
    cams, obs, truth = _scene(
        RenderOptions(noise_px=1.0, occlusion_rate=0.02, outlier_rate=0.03, seed=5)
    )
    start = _perturbed(cams, deg=2.0, metres=0.1, seed=1)
    opts = BundleOptions(scale_anchor=("neck", truth.bone_lengths_m["neck"]))
    result = bundle_adjust(start, obs, length_prior_m=DEFAULT_LENGTHS_M, options=opts)
    injected = set()
    for cid, pairs in truth.outliers.items():
        c = list(obs.camera_ids).index(cid)
        injected |= {(c * 10_000 + f, j) for f, j in pairs}
    flagged = {
        (
            list(obs.camera_ids).index(r.camera_id) * 10_000 + r.frame,
            JOINT_NAMES.index(r.joint),
        )
        for r in result.rejected
    }
    scores = outlier_flag_scores(flagged, injected)
    assert scores.recall is not None and scores.recall >= 0.9
    assert scores.precision is not None and scores.precision >= 0.85
    assert result.rms_px < 2.0
    pe = joint_position_errors(result.joints_3d_m, np.array(truth.joints_3d_m))
    # median: the few points every view got wrong are unobservable and are
    # placed by the segments alone; they are counted in `unobservable_points`
    errors = np.linalg.norm(result.joints_3d_m - np.array(truth.joints_3d_m), axis=2)
    assert float(np.median(errors)) < 0.02 and pe.p95_m is not None and pe.p95_m < 0.05
    assert result.unobservable_points >= 1


def test_contracts() -> None:
    cams = _rig()
    with pytest.raises(Exception, match="need >= 2"):
        Observations(("a",), np.zeros((1, 2, 3, 2)), np.zeros((1, 2, 3)))
    obs = Observations(
        tuple(c.camera_id for c in cams), np.zeros((3, 2, 15, 2)), np.ones((3, 2, 15))
    )
    with pytest.raises(Exception, match="missing segments"):
        bundle_adjust(cams, obs, length_prior_m={"neck": 0.5})
    with pytest.raises(Exception, match="sigma_px"):
        BundleOptions(sigma_px=0)
    with pytest.raises(Exception, match="anchor"):
        BundleOptions(scale_anchor=("neck", -1.0))
    with pytest.raises(Exception, match="name a segment"):
        bundle_adjust(
            cams,
            obs,
            length_prior_m=DEFAULT_LENGTHS_M,
            options=BundleOptions(scale_anchor=("tail", 0.3)),
        )


def test_compact_and_expand_frames_keep_only_observed_frames() -> None:
    """Frames no camera saw stay out of the bundle and come back unobservable (#9802)."""
    from src.motion_capture.reconstruct.bundle import (
        BundleResult,
        Observations,
        RejectedObservation,
        compact_frames,
        expand_result,
        observed_frames,
    )
    from src.motion_capture.reconstruct.cameras import PinholeCamera

    c, t, k = 2, 6, 3
    px = np.zeros((c, t, k, 2))
    conf = np.zeros((c, t, k))
    conf[0, 1, 0] = 0.9
    conf[1, 4] = 0.5
    obs = Observations(("a", "b"), px, conf)
    assert observed_frames(obs).tolist() == [1, 4]
    compact, keep = compact_frames(obs)
    assert compact.pixels.shape == (c, 2, k, 2) and keep.tolist() == [1, 4]
    cameras = (
        PinholeCamera("a", np.eye(3), np.eye(3), np.zeros(3), (10, 10)),
        PinholeCamera("b", np.eye(3), np.eye(3), np.ones(3), (10, 10)),
    )
    result = BundleResult(
        cameras=cameras,
        joints_3d_m=np.ones((2, k, 3)),
        bone_lengths_m={},
        residuals_px=np.ones((c, 2, k)),
        rejected=(RejectedObservation("a", 1, "j", 3.0),),
        rms_px=1.0,
        iterations=1,
        converged=True,
        unobservable_points=1,
    )
    full = expand_result(result, keep, t)
    assert full.joints_3d_m.shape == (t, k, 3)
    assert full.joints_3d_m[1].sum() == k * 3 and full.joints_3d_m[0].sum() == 0
    assert (
        np.isnan(full.residuals_px[:, 0]).all()
        and full.residuals_px[:, 4].sum() == c * k
    )
    assert full.rejected[0].frame == 4
    assert full.unobservable_points == 1 + 4 * k
    with pytest.raises(Exception, match="no observed frame"):
        compact_frames(Observations(("a", "b"), px, np.zeros((c, t, k))))


def test_behind_camera_observation_produces_large_residual_and_is_rejected() -> None:
    """An observation whose 3D point is behind a camera contributes a large penalty and is rejected (#13003)."""
    from src.motion_capture.reconstruct.bundle import _Problem, _residual_map

    cams, obs, truth = _scene(
        RenderOptions(noise_px=0.0, occlusion_rate=0.0, outlier_rate=0.0, seed=42),
        n_frames=2,
    )
    joints = np.array(truth.joints_3d_m)[:2].copy()
    # Camera 'face_on' is at (0, 1.2, 4) looking at (0, 1, 0) (so looking towards -z).
    # A point at z = +8 is at depth < 0 (behind camera 'face_on').
    neck_idx = JOINT_NAMES.index("neck")
    joints[0, neck_idx] = np.array([0.0, 1.2, 8.0])

    opts = BundleOptions(scale_anchor=("neck", truth.bone_lengths_m["neck"]))
    problem = _Problem(
        cams, obs, joints, DEFAULT_LENGTHS_M, DEFAULT_LENGTHS_M, opts, JOINT_NAMES
    )
    residuals = problem.residuals(problem.x0)
    face_on_idx = [c.camera_id for c in cams].index("face_on")
    res_map = _residual_map(cams, joints, obs, problem.mask)
    assert res_map[face_on_idx, 0, neck_idx] >= 1000.0
    assert np.all(np.isfinite(residuals))
    assert np.max(np.abs(residuals)) >= 1000.0

    # Add a camera facing away from the scene so the reconstructed scene is behind it.
    behind_cam = PinholeCamera(
        "behind_cam",
        cams[0].matrix,
        look_at(np.array([0.0, 1.2, -4.0]), np.array([0.0, 1.0, -8.0])),
        np.array([0.0, 1.2, -4.0]),
        (1920, 1200),
    )
    all_cams = [*cams, behind_cam]
    c, t, k, _ = obs.pixels.shape
    new_pixels = np.zeros((c + 1, t, k, 2))
    new_pixels[:c] = obs.pixels
    new_pixels[c, 0, neck_idx] = [960.0, 600.0]
    new_conf = np.zeros((c + 1, t, k))
    new_conf[:c] = obs.confidence
    new_conf[c, 0, neck_idx] = 1.0
    new_obs = Observations(
        tuple(cam.camera_id for cam in all_cams), new_pixels, new_conf
    )

    res = bundle_adjust(
        all_cams, new_obs, length_prior_m=DEFAULT_LENGTHS_M, options=opts
    )
    rejected_keys = {(r.camera_id, r.frame, r.joint) for r in res.rejected}
    assert ("behind_cam", 0, "neck") in rejected_keys


def test_bundle_converged_reflects_least_squares_status() -> None:
    """BundleResult.converged must be False when max_iterations is exceeded (#13003)."""
    cams, obs, truth = _scene(
        RenderOptions(noise_px=1.0, occlusion_rate=0.0, outlier_rate=0.0, seed=1),
        n_frames=2,
    )
    start = _perturbed(cams, deg=5.0, metres=0.2, seed=2)
    opts = BundleOptions(
        scale_anchor=("neck", truth.bone_lengths_m["neck"]),
        max_iterations=1,
    )
    res = bundle_adjust(start, obs, length_prior_m=DEFAULT_LENGTHS_M, options=opts)
    assert res.converged is False


def test_observations_from_views_mixed_fps_pairs_by_physical_time() -> None:
    """Views at different fps associate by physical time and drop no frames (#13002)."""
    layout = {"keypoint_names": ["nose", "neck"]}
    frames_x = [
        {
            "time_s": i / 30.0,
            "keypoints_px": [[100.0 * (i / 30.0), 200.0], [150.0, 250.0]],
            "confidence": [0.9, 0.9],
        }
        for i in range(15)
    ]
    frames_y = [
        {
            "time_s": i / 60.0,
            "keypoints_px": [[100.0 * (i / 60.0), 200.0], [150.0, 250.0]],
            "confidence": [0.9, 0.9],
        }
        for i in range(30)
    ]
    payloads = {
        "x": {
            "view": "x",
            "fps": 30.0,
            "frames_total": 15,
            "detector_layout": layout,
            "frames": frames_x,
        },
        "y": {
            "view": "y",
            "fps": 60.0,
            "frames_total": 30,
            "detector_layout": layout,
            "frames": frames_y,
        },
    }
    obs = observations_from_views(payloads, ["x", "y"])
    assert obs.pixels.shape[1] == 30
    assert getattr(obs, "dropped_frames", 0) == 0
    # In both views, slot t corresponds to the same physical time (t / 60.0)
    # At t = 2 (t_s = 2/60 = 1/30 s), both views should have observations with matching px
    np.testing.assert_allclose(obs.pixels[0, 2, 0], obs.pixels[1, 2, 0])
    # At t = 1 (t_s = 1/60 s), view x has no frame (must be NaN, conf 0.0)
    assert np.isnan(obs.pixels[0, 1]).all()
    assert obs.confidence[0, 1].sum() == 0.0
    # View y has a frame at t = 1
    assert np.isfinite(obs.pixels[1, 1]).all()
    # At t = 28 (t_s = 28/60 = 14/30 s), both views match
    np.testing.assert_allclose(obs.pixels[0, 28, 0], obs.pixels[1, 28, 0])
    # The last frame of y (t = 29) is preserved, not dropped
    assert np.isfinite(obs.pixels[1, 29]).all()


def test_observations_from_views_unequal_frame_counts_same_fps() -> None:
    """Views with unequal total frames at same fps preserve all frames (#13002)."""
    layout = {"keypoint_names": ["neck"]}
    frames_a = [
        {"time_s": i / 30.0, "keypoints_px": [[10.0, 20.0]], "confidence": [0.8]}
        for i in range(10)
    ]
    frames_b = [
        {"time_s": i / 30.0, "keypoints_px": [[10.0, 20.0]], "confidence": [0.8]}
        for i in range(25)
    ]
    payloads = {
        "a": {
            "view": "a",
            "fps": 30.0,
            "frames_total": 10,
            "detector_layout": layout,
            "frames": frames_a,
        },
        "b": {
            "view": "b",
            "fps": 30.0,
            "frames_total": 25,
            "detector_layout": layout,
            "frames": frames_b,
        },
    }
    obs = observations_from_views(payloads, ["a", "b"])
    assert obs.pixels.shape[1] == 25
    assert getattr(obs, "dropped_frames", 0) == 0
    assert np.isfinite(obs.pixels[1, 24]).all()
