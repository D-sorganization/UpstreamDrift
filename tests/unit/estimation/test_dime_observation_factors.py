"""Behavioral test suite for robust marker and markerless observation factors (#11424).

Validates:
- RED tests:
  1. Irregular timestamps rejection / validation (non-monotonic, non-finite, mismatched).
  2. Camera transform inversion handling (SE(3) inversion consistency, SO(3) det=+1).
  3. One occluded marker / missing keypoint handling without zero-filling.
  4. Anisotropic noise whitening (decoupled axis scaling, full covariance Cholesky).
  5. Outlier rejection / downweighting via robust loss (Huber, Tukey, Cauchy, Pseudo-Huber).
  6. Mirrored coordinates / chirality validation (reflection det=-1, camera depth Z <= 0).
  7. Quaternion sign-equivalent poses produce identical residuals.
- GREEN tests:
  1. Finite-difference derivative comparisons away from singularities (analytical vs FD).
  2. Noiseless projection recovery (residuals identically zero at ground truth state).
  3. Masked residual dimension checks (held-out and occluded cleanly excluded from fit).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError, PreconditionError
from src.shared.python.estimation.dime_observation_factors import (
    ChiralityViolationError,
    CovarianceValidationError,
    DimeCameraParameters,
    DimeObservationFactor,
    HeldOutEvaluationReport,
    Marker3DObservationFactor,
    MarkerAttachment,
    Markerless2DObservationFactor,
    ObservationTiming,
    RobustLossKernel,
    TimingViolationError,
    invert_camera_extrinsics,
    quaternion_to_rotation_matrix,
)


# ==============================================================================
# Helper kinematic models for testing
# ==============================================================================


def _simple_planar_kinematics(q: np.ndarray) -> np.ndarray:
    """Predict 3 marker positions in world frame from 3-DOF state [x, y, theta]."""
    x, y, theta = float(q[0]), float(q[1]), float(q[2])
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    local_offsets = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.3],
        ]
    )
    world_xy = local_offsets @ rot.T + np.array([x, y])
    z = np.full((len(local_offsets), 1), 1.5)
    return np.hstack([world_xy, z])


def _quaternion_body_kinematics(q: np.ndarray) -> np.ndarray:
    """Predict 3 marker positions from SE(3) state [px, py, pz, qw, qx, qy, qz]."""
    pos = q[:3]
    quat = q[3:7]
    r_mat = quaternion_to_rotation_matrix(quat)
    local_pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.1],
            [-0.2, 0.3, -0.1],
        ]
    )
    return local_pts @ r_mat.T + pos


# ==============================================================================
# RED Test Suite
# ==============================================================================


class TestRedObservationFactors:
    """RED behavioral suite enforcing fail-closed constraints and robust semantics."""

    @pytest.mark.unit
    def test_red_irregular_timestamps_rejection(self) -> None:
        """Irregular, non-monotonic, non-finite or mismatched timestamps must be rejected."""
        # Non-monotonic timestamps
        with pytest.raises(
            (TimingViolationError, PreconditionError, ContractViolationError)
        ):
            ObservationTiming(timestamps=np.array([0.0, 0.05, 0.04, 0.10]))

        # Repeated / non-increasing timestamps (zero dt)
        with pytest.raises(
            (TimingViolationError, PreconditionError, ContractViolationError)
        ):
            ObservationTiming(timestamps=np.array([0.0, 0.05, 0.05, 0.10]))

        # Non-finite timestamps (NaN / Inf)
        with pytest.raises(
            (TimingViolationError, PreconditionError, ContractViolationError)
        ):
            ObservationTiming(timestamps=np.array([0.0, np.nan, 0.10]))

        with pytest.raises(
            (TimingViolationError, PreconditionError, ContractViolationError)
        ):
            ObservationTiming(timestamps=np.array([0.0, np.inf, 0.10]))

        # Non-positive dt declaration
        with pytest.raises(
            (TimingViolationError, PreconditionError, ContractViolationError)
        ):
            ObservationTiming(timestamps=np.array([0.0, 0.05, 0.10]), dt=-0.05)

        # Mismatched frame indices length
        with pytest.raises(
            (TimingViolationError, PreconditionError, ContractViolationError)
        ):
            ObservationTiming(
                timestamps=np.array([0.0, 0.05, 0.10]),
                frame_indices=np.array([0, 1]),
            )

    @pytest.mark.unit
    def test_red_camera_transform_inversion_handling(self) -> None:
        """Camera transform inversion must preserve SE(3) consistency and reject non-SO(3)."""
        # Proper camera orientation looking forward and down
        angle = np.deg2rad(30.0)
        c, s = np.cos(angle), np.sin(angle)
        # R_x(30)
        r_wc = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        t_wc = np.array([1.5, -2.0, 2.5])

        # Test pure function invert_camera_extrinsics
        r_cw, t_cw = invert_camera_extrinsics(r_wc, t_wc)

        # Inversion round-trip
        r_rec, t_rec = invert_camera_extrinsics(r_cw, t_cw)
        assert np.allclose(r_rec, r_wc, atol=1e-12)
        assert np.allclose(t_rec, t_wc, atol=1e-12)

        # Geometric consistency on arbitrary world point
        p_world = np.array([2.0, 1.0, 0.5])
        # In camera frame: P_c = (P_w - t_wc) @ R_wc = P_w @ R_cw.T + t_cw
        p_c1 = (p_world - t_wc) @ r_wc
        p_c2 = p_world @ r_cw.T + t_cw
        assert np.allclose(p_c1, p_c2, atol=1e-12)

        # DimeCameraParameters inversion
        k_mat = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        cam = DimeCameraParameters.from_world_from_camera(
            camera_id="cam_test",
            matrix=k_mat,
            rotation_world_from_camera=r_wc,
            translation_world_from_camera_m=t_wc,
        )
        cam_inv = cam.invert()
        cam_roundtrip = cam_inv.invert()
        assert np.allclose(
            cam.rotation_world_to_camera,
            cam_roundtrip.rotation_world_to_camera,
            atol=1e-12,
        )
        assert np.allclose(
            cam.translation_world_to_camera,
            cam_roundtrip.translation_world_to_camera,
            atol=1e-12,
        )

        # Skewed (non-orthonormal) matrix must be rejected
        r_bad = r_wc.copy()
        r_bad[0, 1] += 0.2
        with pytest.raises(
            (ChiralityViolationError, PreconditionError, ContractViolationError)
        ):
            invert_camera_extrinsics(r_bad, t_wc)

    @pytest.mark.unit
    def test_red_occluded_marker_missing_keypoint_no_zero_filling(self) -> None:
        """Occluded markers or missing keypoints must not be zero-filled."""
        q_true = np.array([0.1, 0.2, 0.05])
        predicted_3d = _simple_planar_kinematics(q_true)  # (3, 3)

        # Case A: 3D Marker with 1 occluded marker (marker 1 is occluded)
        valid_mask_3d = np.array([True, False, True])
        factor_3d = Marker3DObservationFactor(
            observations_3d_m=predicted_3d,
            kinematics_fn=_simple_planar_kinematics,
            covariance=0.01**2,
            valid_mask=valid_mask_3d,
        )

        # Residual dimension must strictly be 2 markers * 3 coords = 6, NOT 9!
        assert factor_3d.residual_dimension == 6
        assert factor_3d.num_active_observations == 2

        # Residual vector must only contain errors for markers 0 and 2
        residuals_3d = factor_3d.evaluate_residuals(q_true)
        assert residuals_3d.shape == (6,)
        assert np.allclose(residuals_3d, 0.0, atol=1e-12)

        # If someone had zero-filled marker 1 to [0, 0, 0], evaluating at q_true
        # would create a non-zero residual for marker 1. Verify our residual has zero length for marker 1.
        raw_res = factor_3d.evaluate_raw_residuals(q_true)
        assert raw_res.shape == (6,)
        assert np.allclose(raw_res, 0.0, atol=1e-12)

        # Case B: 2D Markerless with 1 missing keypoint (keypoint 2 is missing)
        k_mat = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        cam = DimeCameraParameters(
            camera_id="cam0",
            matrix=k_mat,
            rotation_world_to_camera=np.eye(3),
            translation_world_to_camera=np.zeros(3),
        )
        obs_2d = cam.project(predicted_3d)  # (3, 2)
        valid_mask_2d = np.array([True, True, False])

        factor_2d = Markerless2DObservationFactor(
            observations_2d_px=obs_2d,
            kinematics_fn=_simple_planar_kinematics,
            camera=cam,
            covariance=1.0**2,
            valid_mask=valid_mask_2d,
        )

        # Residual dimension must strictly be 2 keypoints * 2 coords = 4, NOT 6!
        assert factor_2d.residual_dimension == 4
        assert factor_2d.num_active_observations == 2
        res_2d = factor_2d.evaluate_residuals(q_true)
        assert res_2d.shape == (4,)
        assert np.allclose(res_2d, 0.0, atol=1e-12)

    @pytest.mark.unit
    def test_red_anisotropic_noise_whitening(self) -> None:
        """Anisotropic noise whitening must correctly scale independent axes."""
        q_eval = np.array([0.0, 0.0, 0.0])
        pred_pts = _simple_planar_kinematics(q_eval)  # 3 markers

        # Anisotropic standard deviations in meters: [sigma_x=0.01, sigma_y=0.02, sigma_z=0.05]
        sigmas = np.array([0.01, 0.02, 0.05])
        variances = sigmas**2

        # Inject error: [dx=0.02, dy=0.04, dz=0.10] on every marker
        # Expected whitened error on each marker: [0.02/0.01, 0.04/0.02, 0.10/0.05] = [2.0, 2.0, 2.0]
        obs_perturbed = pred_pts.copy()
        obs_perturbed[:, 0] -= 0.02
        obs_perturbed[:, 1] -= 0.04
        obs_perturbed[:, 2] -= 0.10

        factor = Marker3DObservationFactor(
            observations_3d_m=obs_perturbed,
            kinematics_fn=_simple_planar_kinematics,
            covariance=variances,
            robust_loss=RobustLossKernel(
                "linear"
            ),  # unweighted robust kernel to test pure whitening
        )

        res = factor.evaluate_residuals(q_eval)
        # 3 markers * 3 coordinates = 9
        assert res.shape == (9,)
        expected_whitened = np.tile([2.0, 2.0, 2.0], 3)
        assert np.allclose(res, expected_whitened, atol=1e-12)

        # Full covariance matrix whitening test (3x3 with cross-correlation)
        cov_3x3 = np.array(
            [
                [0.04, 0.01, 0.0],
                [0.01, 0.02, 0.0],
                [0.0, 0.0, 0.09],
            ]
        )
        factor_full = Marker3DObservationFactor(
            observations_3d_m=obs_perturbed,
            kinematics_fn=_simple_planar_kinematics,
            covariance=cov_3x3,
            robust_loss=RobustLossKernel("linear"),
        )
        res_full = factor_full.evaluate_residuals(q_eval)
        # For each marker error e, z^T z must equal e^T cov_inv e
        raw_e = np.array([0.02, 0.04, 0.10])
        expected_mahalanobis_sq = float(raw_e @ np.linalg.inv(cov_3x3) @ raw_e)
        marker0_z = res_full[:3]
        assert np.isclose(
            float(marker0_z @ marker0_z), expected_mahalanobis_sq, atol=1e-10
        )

        # Non-positive variance must be rejected
        with pytest.raises(
            (CovarianceValidationError, PreconditionError, ContractViolationError)
        ):
            Marker3DObservationFactor(
                observations_3d_m=pred_pts,
                kinematics_fn=_simple_planar_kinematics,
                covariance=np.array([0.01, -0.02, 0.05]),
            )

    @pytest.mark.unit
    def test_red_outlier_downweighting_via_robust_loss(self) -> None:
        """Gross outliers must receive sub-quadratic penalties under robust kernels."""
        q_eval = np.array([0.0, 0.0, 0.0])
        pred_pts = _simple_planar_kinematics(q_eval)

        # Marker 0: inlier (error 1 sigma)
        # Marker 1: mild outlier (error 3 sigma)
        # Marker 2: gross outlier (error 15 sigma)
        sigma = 0.01
        obs_pts = pred_pts.copy()
        obs_pts[0, 0] -= 1.0 * sigma
        obs_pts[1, 0] -= 3.0 * sigma
        obs_pts[2, 0] -= 15.0 * sigma

        # Linear loss
        factor_linear = Marker3DObservationFactor(
            observations_3d_m=obs_pts,
            kinematics_fn=_simple_planar_kinematics,
            covariance=sigma**2,
            robust_loss=RobustLossKernel("linear"),
        )
        res_lin = factor_linear.evaluate_residuals(q_eval)
        pt_norms_lin = [
            np.linalg.norm(res_lin[0:3]),
            np.linalg.norm(res_lin[3:6]),
            np.linalg.norm(res_lin[6:9]),
        ]
        assert np.isclose(pt_norms_lin[0], 1.0, atol=1e-8)
        assert np.isclose(pt_norms_lin[1], 3.0, atol=1e-8)
        assert np.isclose(pt_norms_lin[2], 15.0, atol=1e-8)

        # Huber loss with delta = 1.345
        factor_huber = Marker3DObservationFactor(
            observations_3d_m=obs_pts,
            kinematics_fn=_simple_planar_kinematics,
            covariance=sigma**2,
            robust_loss=RobustLossKernel("huber", tuning_constant=1.345),
        )
        res_huber = factor_huber.evaluate_residuals(q_eval)
        pt_norms_huber = [
            np.linalg.norm(res_huber[0:3]),
            np.linalg.norm(res_huber[3:6]),
            np.linalg.norm(res_huber[6:9]),
        ]
        # Inlier <= delta is unmodified
        assert np.isclose(pt_norms_huber[0], 1.0, atol=1e-8)
        # Outliers have reduced effective norms (sqrt(delta * r))
        assert pt_norms_huber[2] < pt_norms_lin[2]
        assert np.isclose(pt_norms_huber[2], np.sqrt(1.345 * 15.0), atol=1e-6)

        # Tukey biweight loss with cutoff c = 4.685
        factor_tukey = Marker3DObservationFactor(
            observations_3d_m=obs_pts,
            kinematics_fn=_simple_planar_kinematics,
            covariance=sigma**2,
            robust_loss=RobustLossKernel("tukey", tuning_constant=4.685),
        )
        res_tukey = factor_tukey.evaluate_residuals(q_eval)
        pt_norms_tukey = [
            np.linalg.norm(res_tukey[0:3]),
            np.linalg.norm(res_tukey[3:6]),
            np.linalg.norm(res_tukey[6:9]),
        ]
        # Inlier has non-zero weight
        assert pt_norms_tukey[0] > 0.0
        # Gross outlier (> c) has exactly 0 weight, completely rejected!
        assert np.isclose(pt_norms_tukey[2], 0.0, atol=1e-12)

        # Cauchy & Pseudo-Huber downweight gross outliers
        factor_cauchy = Marker3DObservationFactor(
            observations_3d_m=obs_pts,
            kinematics_fn=_simple_planar_kinematics,
            covariance=sigma**2,
            robust_loss=RobustLossKernel("cauchy", tuning_constant=2.0),
        )
        res_cauchy = factor_cauchy.evaluate_residuals(q_eval)
        assert np.linalg.norm(res_cauchy[6:9]) < pt_norms_lin[2]

        factor_ph = Marker3DObservationFactor(
            observations_3d_m=obs_pts,
            kinematics_fn=_simple_planar_kinematics,
            covariance=sigma**2,
            robust_loss=RobustLossKernel("pseudo_huber", tuning_constant=1.345),
        )
        res_ph = factor_ph.evaluate_residuals(q_eval)
        assert np.linalg.norm(res_ph[6:9]) < pt_norms_lin[2]

    @pytest.mark.unit
    def test_red_mirrored_coordinates_chirality_validation(self) -> None:
        """Reflection matrices (det=-1) and negative camera depth (Z<=0) must be rejected."""
        # Reflection matrix with det=-1
        r_mirror = np.diag([1.0, 1.0, -1.0])
        t_cam = np.array([0.0, 0.0, 0.0])
        k_mat = np.eye(3) * 500.0
        k_mat[2, 2] = 1.0

        with pytest.raises(
            (ChiralityViolationError, PreconditionError, ContractViolationError)
        ):
            DimeCameraParameters(
                camera_id="cam_mirror",
                matrix=k_mat,
                rotation_world_to_camera=r_mirror,
                translation_world_to_camera=t_cam,
            )

        # Chirality depth check: point behind camera (Z_c = -1.0)
        cam = DimeCameraParameters(
            camera_id="cam_normal",
            matrix=k_mat,
            rotation_world_to_camera=np.eye(3),
            translation_world_to_camera=np.zeros(3),
        )
        points_behind = np.array([[1.0, 1.0, -1.5]])  # Z < 0

        with pytest.raises(
            (ChiralityViolationError, PreconditionError, ContractViolationError)
        ):
            cam.validate_chirality(points_behind)

    @pytest.mark.unit
    def test_red_quaternion_sign_equivalent_poses_produce_identical_residuals(
        self,
    ) -> None:
        """Quaternion sign-equivalent poses q and -q must produce identical residuals."""
        # Base state with quaternion [w, x, y, z]
        # Orientation rotated 45 degrees about [1, 1, 1]/sqrt(3)
        axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        theta = np.deg2rad(45.0)
        qw = np.cos(theta / 2.0)
        qxyz = np.sin(theta / 2.0) * axis
        q_pos = np.array([0.5, 0.2, 1.0, qw, qxyz[0], qxyz[1], qxyz[2]])
        q_neg = np.array([0.5, 0.2, 1.0, -qw, -qxyz[0], -qxyz[1], -qxyz[2]])

        # 3D Marker observations
        obs_3d = _quaternion_body_kinematics(q_pos)
        # Perturb observations with noise
        obs_noisy_3d = obs_3d + 0.005

        factor_3d = Marker3DObservationFactor(
            observations_3d_m=obs_noisy_3d,
            kinematics_fn=_quaternion_body_kinematics,
            covariance=0.01**2,
            robust_loss=RobustLossKernel("huber"),
        )
        res_pos_3d = factor_3d.evaluate_residuals(q_pos)
        res_neg_3d = factor_3d.evaluate_residuals(q_neg)
        assert np.allclose(res_pos_3d, res_neg_3d, atol=1e-12)

        # 2D Keypoint observations
        k_mat = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        cam = DimeCameraParameters(
            camera_id="cam0",
            matrix=k_mat,
            rotation_world_to_camera=np.eye(3),
            translation_world_to_camera=np.array([0.0, 0.0, 1.0]),
        )
        obs_2d = cam.project(obs_3d) + 1.5

        factor_2d = Markerless2DObservationFactor(
            observations_2d_px=obs_2d,
            kinematics_fn=_quaternion_body_kinematics,
            camera=cam,
            covariance=2.0**2,
            robust_loss=RobustLossKernel("cauchy"),
        )
        res_pos_2d = factor_2d.evaluate_residuals(q_pos)
        res_neg_2d = factor_2d.evaluate_residuals(q_neg)
        assert np.allclose(res_pos_2d, res_neg_2d, atol=1e-12)


# ==============================================================================
# GREEN Test Suite
# ==============================================================================


class TestGreenObservationFactors:
    """GREEN behavioral suite validating numerical derivatives and partition semantics."""

    @pytest.mark.unit
    def test_green_finite_difference_derivatives_match_analytical(self) -> None:
        """Factor Jacobians must match central finite-difference derivatives within tolerance."""
        q_test = np.array([0.2, 0.3, 0.15])
        predicted_3d = _simple_planar_kinematics(q_test)
        # Small offset to make residuals nonzero away from Huber corner
        obs = predicted_3d + 0.003

        factor = Marker3DObservationFactor(
            observations_3d_m=obs,
            kinematics_fn=_simple_planar_kinematics,
            covariance=0.01**2,
            robust_loss=RobustLossKernel("pseudo_huber", tuning_constant=2.0),
        )

        jac_auto = factor.evaluate_jacobian(q_test, method="auto")
        jac_fd = factor.evaluate_jacobian(q_test, method="finite", step=1e-6)

        assert jac_auto.shape == (9, 3)
        assert np.allclose(jac_auto, jac_fd, atol=1e-5, rtol=1e-4)

        # Check 2D factor Jacobians as well
        k_mat = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]])
        cam = DimeCameraParameters(
            camera_id="cam0",
            matrix=k_mat,
            rotation_world_to_camera=np.eye(3),
            translation_world_to_camera=np.zeros(3),
        )
        obs_2d = cam.project(predicted_3d) + 2.0

        factor_2d = Markerless2DObservationFactor(
            observations_2d_px=obs_2d,
            kinematics_fn=_simple_planar_kinematics,
            camera=cam,
            covariance=1.5**2,
            robust_loss=RobustLossKernel("pseudo_huber", tuning_constant=2.0),
        )
        jac_2d_auto = factor_2d.evaluate_jacobian(q_test, method="auto")
        jac_2d_fd = factor_2d.evaluate_jacobian(q_test, method="finite", step=1e-6)

        assert jac_2d_auto.shape == (6, 3)
        assert np.allclose(jac_2d_auto, jac_2d_fd, atol=1e-5, rtol=1e-4)

    @pytest.mark.unit
    def test_green_noiseless_projection_recovery(self) -> None:
        """Residuals must be identically zero when candidate state equals ground truth."""
        q_true = np.array([0.15, -0.25, 0.40])
        pts_3d = _simple_planar_kinematics(q_true)

        # 3D factor
        factor_3d = Marker3DObservationFactor(
            observations_3d_m=pts_3d,
            kinematics_fn=_simple_planar_kinematics,
            covariance=0.01**2,
        )
        res_3d = factor_3d.evaluate_residuals(q_true)
        assert np.allclose(res_3d, 0.0, atol=1e-12)
        raw_res_3d = factor_3d.evaluate_raw_residuals(q_true)
        assert np.allclose(raw_res_3d, 0.0, atol=1e-12)

        # 2D factor
        k_mat = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        cam = DimeCameraParameters(
            camera_id="cam_ground_truth",
            matrix=k_mat,
            rotation_world_to_camera=np.eye(3),
            translation_world_to_camera=np.array([0.0, 0.0, 0.5]),
        )
        pts_2d = cam.project(pts_3d)

        factor_2d = Markerless2DObservationFactor(
            observations_2d_px=pts_2d,
            kinematics_fn=_simple_planar_kinematics,
            camera=cam,
            covariance=1.0**2,
        )
        res_2d = factor_2d.evaluate_residuals(q_true)
        assert np.allclose(res_2d, 0.0, atol=1e-12)
        raw_res_2d = factor_2d.evaluate_raw_residuals(q_true)
        assert np.allclose(raw_res_2d, 0.0, atol=1e-12)

    @pytest.mark.unit
    def test_green_masked_residual_dimension_checks(self) -> None:
        """Held-out observations must be cleanly excluded from the fitting residual vector."""
        q_eval = np.array([0.1, 0.2, 0.0])
        pts = _simple_planar_kinematics(q_eval)  # 3 markers

        # Setup: marker 0 is active fit, marker 1 is occluded, marker 2 is held-out
        valid_mask = np.array([True, False, True])
        held_out_mask = np.array([False, False, True])

        # Active fit markers: valid & ~held_out => only marker 0!
        # Active held-out markers: valid & held_out => only marker 2!

        # Introduce intentional errors:
        # marker 0 has 0.01m error
        # marker 2 (held-out) has 0.05m error
        obs = pts.copy()
        obs[0, 0] -= 0.01
        obs[2, 1] -= 0.05

        factor = Marker3DObservationFactor(
            observations_3d_m=obs,
            kinematics_fn=_simple_planar_kinematics,
            covariance=0.01**2,
            valid_mask=valid_mask,
            held_out_mask=held_out_mask,
            robust_loss=RobustLossKernel("linear"),
        )

        # Residual dimension must strictly be 1 marker * 3 coords = 3!
        assert factor.residual_dimension == 3
        assert factor.num_active_observations == 1

        fit_res = factor.evaluate_residuals(q_eval)
        assert fit_res.shape == (3,)
        # Error on marker 0 is 0.01m / 0.01m = 1.0
        assert np.isclose(fit_res[0], 1.0, atol=1e-12)

        # Evaluate held-out partition
        report = factor.evaluate_held_out(q_eval)
        assert isinstance(report, HeldOutEvaluationReport)
        assert report.num_held_out == 1
        assert report.canonical_unit == "m"
        assert np.isclose(report.rms_error, 0.05, atol=1e-12)
        assert np.isclose(report.max_error, 0.05, atol=1e-12)
        assert report.raw_residuals.shape == (1, 3)

        # Jacobian dimension check
        jac = factor.evaluate_jacobian(q_eval)
        assert jac.shape == (3, 3)
