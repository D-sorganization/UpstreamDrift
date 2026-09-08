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
