"""Tests for virtual camera fitting and 2D swing envelope (COV-4, #11272)."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.capture_registry import (
    CaptureDataUnavailable,
    CaptureRegistryError,
    UnknownCaptureError,
    resolve_capture,
)
from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.motion_capture.reference.registration import project_reference_to_camera
from src.motion_capture.reference.virtual_camera_fit import (
    CameraFitOutcome,
    SwingEnvelope2D,
    VirtualCameraResult,
    camera_projection_to_pinhole_camera,
    compute_2d_envelope,
    fit_virtual_camera,
    pinhole_camera_to_camera_projection,
)
from src.shared.python.motion_matching.historical_fit.contracts import CameraProjection
from src.shared.python.motion_matching.loaders._marker_clusters import y_up_to_z_up

pytestmark = pytest.mark.unit


def synthetic_skeleton_3d() -> dict[str, np.ndarray]:
    """14 realistic landmarks in ADR-0041 world frame (X forward, Y up, Z right)."""
    return {
        "Pelvis": np.array([0.0, 0.95, 0.0]),
        "Head": np.array([0.0, 1.70, 0.0]),
        "LShoulder": np.array([0.0, 1.45, -0.22]),
        "RShoulder": np.array([0.0, 1.45, 0.22]),
        "LElbow": np.array([0.15, 1.15, -0.20]),
        "RElbow": np.array([0.15, 1.15, 0.20]),
        "LWrist": np.array([0.30, 0.85, -0.05]),
        "RWrist": np.array([0.30, 0.85, 0.05]),
        "LHip": np.array([0.0, 0.90, -0.15]),
        "RHip": np.array([0.0, 0.90, 0.15]),
        "LKnee": np.array([0.05, 0.50, -0.16]),
        "RKnee": np.array([0.05, 0.50, 0.16]),
        "LAnkle": np.array([0.0, 0.10, -0.18]),
        "RAnkle": np.array([0.0, 0.10, 0.18]),
    }


def synthetic_camera() -> PinholeCamera:
    """Down-the-line camera looking toward the origin in ADR-0041 world frame."""
    position = np.array([0.0, 1.20, 3.50])
    target = np.array([0.0, 0.95, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    r_wc = look_at(position, target, up)
    k = intrinsics_from_fov(1920, 1080, 55.0)
    return PinholeCamera(
        camera_id="synthetic_dtl",
        matrix=k,
        rotation_world_from_camera=r_wc,
        translation_world_from_camera_m=position,
        image_size_px=(1920, 1080),
    )


def test_synthetic_camera_recovered_noiseless() -> None:
    """A known synthetic camera is recovered within stated tolerance from noiseless projections."""
    # Stated tolerances
    ROTATION_TOL_RAD = 1e-3
    TRANSLATION_TOL_M = 1e-3
    RMS_TOL_PX = 1e-2

    skeleton = synthetic_skeleton_3d()
    world_points = np.stack(list(skeleton.values()), axis=0)  # (14, 3)
    cam_true = synthetic_camera()

    valid_mask = np.ones(len(world_points), dtype=bool)
    image_points, visible = project_reference_to_camera(
        world_points, valid_mask, cam_true
    )
    assert visible.all(), (
        "All synthetic points must be in front of camera and inside image"
    )

    result = fit_virtual_camera(
        world_points=world_points,
        image_points=image_points,
        intrinsics_prior=cam_true.matrix,
        image_size_px=cam_true.image_size_px,
    )

    assert result.outcome == CameraFitOutcome.FITTED
    assert result.degraded_reason is None
    assert result.fit_rms_px < RMS_TOL_PX
    assert result.condition_number < 1e4
    assert result.camera is not None

    # Check recovered camera rotation and translation
    r_diff = (
        result.camera.rotation_world_from_camera.T @ cam_true.rotation_world_from_camera
    )
    angle_error = np.arccos(np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0))
    trans_error = float(np.linalg.norm(result.camera.position_m - cam_true.position_m))

    assert angle_error < ROTATION_TOL_RAD, (
        f"Rotation error {angle_error:.4e} exceeds {ROTATION_TOL_RAD}"
    )
    assert trans_error < TRANSLATION_TOL_M, (
        f"Translation error {trans_error:.4e} exceeds {TRANSLATION_TOL_M}"
    )


def test_synthetic_camera_with_noise_widens_uncertainty_and_stays_unbiased() -> None:
    """Adding 2 px Gaussian noise widens reported uncertainty; estimate stays unbiased within tolerance."""
    POSITION_TOL_M = 0.20
    ROTATION_TOL_RAD = 0.08
    NOISE_SIGMA_PX = 2.0

    skeleton = synthetic_skeleton_3d()
    world_points = np.stack(list(skeleton.values()), axis=0)
    cam_true = synthetic_camera()

    valid_mask = np.ones(len(world_points), dtype=bool)
    image_points_clean, _ = project_reference_to_camera(
        world_points, valid_mask, cam_true
    )

    result_noiseless = fit_virtual_camera(
        world_points=world_points,
        image_points=image_points_clean,
        intrinsics_prior=cam_true.matrix,
        image_size_px=cam_true.image_size_px,
    )

    # Add 2 px noise with fixed seed for determinism
    rng = np.random.default_rng(20261003)
    noise = rng.normal(0.0, NOISE_SIGMA_PX, size=image_points_clean.shape)
    image_points_noisy = image_points_clean + noise

    result_noisy = fit_virtual_camera(
        world_points=world_points,
        image_points=image_points_noisy,
        intrinsics_prior=cam_true.matrix,
        image_size_px=cam_true.image_size_px,
    )

    assert result_noisy.outcome == CameraFitOutcome.FITTED
    # RMS should reflect the ~2 px added noise
    assert 1.0 <= result_noisy.fit_rms_px <= 4.0
    # Uncertainty widens: parameter covariance trace is strictly larger
    unc_noiseless = float(np.trace(result_noiseless.covariance))
    unc_noisy = float(np.trace(result_noisy.covariance))
    assert unc_noisy > unc_noiseless, "Noise must widen reported covariance uncertainty"

    # Estimate stays unbiased within tolerance
    assert result_noisy.camera is not None
    trans_err = float(
        np.linalg.norm(result_noisy.camera.position_m - cam_true.position_m)
    )
    r_diff = (
        result_noisy.camera.rotation_world_from_camera.T
        @ cam_true.rotation_world_from_camera
    )
    angle_err = np.arccos(np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0))

    assert trans_err < POSITION_TOL_M, (
        f"Position error {trans_err:.3f} exceeds {POSITION_TOL_M}"
    )
    assert angle_err < ROTATION_TOL_RAD, (
        f"Angle error {angle_err:.3f} exceeds {ROTATION_TOL_RAD}"
    )


def test_synthetic_coplanar_or_degenerate_is_degraded() -> None:
    """Coplanar or degenerate correspondences -> degraded outcome with reason, not confident result."""
    skeleton = synthetic_skeleton_3d()
    world_points = np.stack(list(skeleton.values()), axis=0).copy()
    # Flatten 3D points to pure coplanar (X = 0)
    world_points[:, 0] = 0.0

    cam_true = synthetic_camera()
    image_points, _ = project_reference_to_camera(
        world_points, np.ones(len(world_points), dtype=bool), cam_true
    )

    result = fit_virtual_camera(
        world_points=world_points,
        image_points=image_points,
        intrinsics_prior=cam_true.matrix,
        image_size_px=cam_true.image_size_px,
    )

    assert result.outcome == CameraFitOutcome.DEGRADED
    assert result.degraded_reason is not None
    assert (
        "coplanar" in result.degraded_reason.lower()
        or "degenerate" in result.degraded_reason.lower()
    )


def test_synthetic_wrong_axis_convention_rejected_by_sign_test() -> None:
    """Wrong axis convention (Y-up input left unconverted) -> mirrored projection is detected by sign test and rejected."""
    skeleton = synthetic_skeleton_3d()
    # In raw C3D: Y is up, X is forward, Z is right/away.
    # Suppose raw Y-up points are:
    raw_y_up = np.array(
        [
            skeleton["Head"],
            skeleton["Pelvis"],
            skeleton["LShoulder"],
            skeleton["RShoulder"],
            skeleton["LWrist"],
            skeleton["RWrist"],
        ]
    )  # (6, 3) where Y is height

    # If unconverted Y-up points are erroneously treated as canonical Z-up (or if lateral axis is mirrored)
    # y_up_to_z_up maps (x, y, z) -> (x, -z, y).
    # Leaving it unconverted means we use raw_y_up directly, which flips chirality or inverts vertical/lateral.
    cam = synthetic_camera()
    # Correct projection from proper converted skeleton
    proper_z_up = y_up_to_z_up(raw_y_up)
    # Convert canonical z-up to ADR-0041: x_w = x_can, y_w = z_can, z_w = -y_can
    proper_adr = np.column_stack(
        [proper_z_up[:, 0], proper_z_up[:, 2], -proper_z_up[:, 1]]
    )
    img_points, _ = project_reference_to_camera(
        proper_adr, np.ones(len(proper_adr), dtype=bool), cam
    )

    # Erroneous unconverted input:
    unconverted_adr = np.column_stack([raw_y_up[:, 0], raw_y_up[:, 2], -raw_y_up[:, 1]])

    with pytest.raises(ValueError, match=r"(?i)mirrored|axis convention"):
        fit_virtual_camera(
            world_points=unconverted_adr,
            image_points=img_points,
            intrinsics_prior=cam.matrix,
            image_size_px=cam.image_size_px,
        )


def test_synthetic_envelope_bands_monotone_and_sample_counts() -> None:
    """Envelope bands are monotone (p5 <= p50 <= p95). 13-swing count asserted, missing marker frames reduce n explicitly."""
    n_swings = 13
    n_phases = 20
    n_landmarks = 14
    rng = np.random.default_rng(42)

    # Generate 13 swings with realistic variation around nominal
    base_traj = rng.uniform(200.0, 800.0, size=(n_phases, n_landmarks, 2))
    swings = np.empty((n_swings, n_phases, n_landmarks, 2), dtype=float)
    for s in range(n_swings):
        swings[s] = base_traj + rng.normal(0.0, 15.0, size=base_traj.shape)

    # Introduce missing marker frames in swings 2 and 5 for landmark 3
    swings[2, 5:10, 3, :] = np.nan
    swings[5, 8:12, 3, :] = np.nan

    phase_bins = np.linspace(0.0, 1.0, n_phases)
    envelope = compute_2d_envelope(
        projected_swings=swings,
        phase_bins=phase_bins,
        body_height_px=900.0,
        expected_swings=13,
    )

    assert isinstance(envelope, SwingEnvelope2D)
    assert envelope.p5.shape == (n_phases, n_landmarks, 2)
    assert envelope.p50.shape == (n_phases, n_landmarks, 2)
    assert envelope.p95.shape == (n_phases, n_landmarks, 2)

    # Monotonicity check: p5 <= p50 <= p95
    assert np.all(envelope.p5 <= envelope.p50)
    assert np.all(envelope.p50 <= envelope.p95)

    # Height-normalized monotonicity
    assert envelope.p5_height_normalized is not None
    assert envelope.p50_height_normalized is not None
    assert envelope.p95_height_normalized is not None
    assert np.all(envelope.p5_height_normalized <= envelope.p50_height_normalized)
    assert np.all(envelope.p50_height_normalized <= envelope.p95_height_normalized)

    # Assert 13-swing count
    assert envelope.sample_counts[0, 0] == 13
    # Frame 9 landmark 3 is missing in both swing 2 and swing 5 -> count is 11
    assert envelope.sample_counts[9, 3] == 11

    # Wrong swing count raises ValueError
    with pytest.raises(ValueError, match="13"):
        compute_2d_envelope(
            projected_swings=swings[:10],
            phase_bins=phase_bins,
            expected_swings=13,
        )


def test_synthetic_typed_errors_and_missing_registry_skip() -> None:
    """Nonfinite input, mismatched landmark set, or missing registry entry -> typed errors or a typed skip."""
    skeleton = synthetic_skeleton_3d()
    world_points = np.stack(list(skeleton.values()), axis=0)
    cam = synthetic_camera()
    image_points, _ = project_reference_to_camera(
        world_points, np.ones(len(world_points), dtype=bool), cam
    )

    # Nonfinite input
    bad_world = world_points.copy()
    bad_world[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_virtual_camera(bad_world, image_points, cam.matrix, cam.image_size_px)

    bad_img = image_points.copy()
    bad_img[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        fit_virtual_camera(world_points, bad_img, cam.matrix, cam.image_size_px)

    # Mismatched landmark count
    with pytest.raises(ValueError, match="length"):
        fit_virtual_camera(
            world_points[:10], image_points, cam.matrix, cam.image_size_px
        )

    # Adapter round-trip between PinholeCamera and CameraProjection
    proj = pinhole_camera_to_camera_projection(cam)
    assert isinstance(proj, CameraProjection)
    cam_back = camera_projection_to_pinhole_camera(
        proj, cam.image_size_px, camera_id=cam.camera_id
    )
    assert np.allclose(
        cam_back.rotation_world_from_camera, cam.rotation_world_from_camera
    )
    assert np.allclose(cam_back.position_m, cam.position_m)
    assert np.allclose(cam_back.matrix, cam.matrix)

    # Missing registry entry raises UnknownCaptureError or CaptureDataUnavailable
    with pytest.raises(CaptureRegistryError):
        resolve_capture("non-existent-capture-id-xyz")
