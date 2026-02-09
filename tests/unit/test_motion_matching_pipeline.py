"""Tests for the complete motion matching pipeline (Issue #759).

Covers:
- Joint angle computation (shared utility)
- MediaPipe and OpenPose joint angle integration
- Validation metrics per S3 tolerances
- End-to-end pipeline validation
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.pose_estimation.joint_angle_utils import (
    OPENPOSE_TO_CANONICAL,
    _angle_between,
    _compute_flexion,
    compute_joint_angles,
)
from src.shared.python.pose_estimation.validation_metrics import (
    S3_TOLERANCES,
    ValidationReport,
    compute_fit_quality,
    compute_joint_angle_rmse,
    compute_marker_rmse,
    compute_temporal_jitter,
    validate_pipeline_output,
)

# ===================================================================
# Fixtures: synthetic keypoint sets
# ===================================================================


@pytest.fixture
def mediapipe_keypoints() -> dict[str, np.ndarray]:
    """A plausible standing pose using MediaPipe landmark names."""
    return {
        "left_shoulder": np.array([0.4, 0.6, 0.0]),
        "right_shoulder": np.array([0.6, 0.6, 0.0]),
        "left_elbow": np.array([0.3, 0.4, 0.0]),
        "right_elbow": np.array([0.7, 0.4, 0.0]),
        "left_wrist": np.array([0.25, 0.2, 0.0]),
        "right_wrist": np.array([0.75, 0.2, 0.0]),
        "left_hip": np.array([0.45, 0.9, 0.0]),
        "right_hip": np.array([0.55, 0.9, 0.0]),
        "left_knee": np.array([0.45, 1.2, 0.0]),
        "right_knee": np.array([0.55, 1.2, 0.0]),
        "left_ankle": np.array([0.45, 1.5, 0.0]),
        "right_ankle": np.array([0.55, 1.5, 0.0]),
    }


@pytest.fixture
def openpose_keypoints() -> dict[str, np.ndarray]:
    """Same pose but with OpenPose BODY_25 names."""
    return {
        "LShoulder": np.array([0.4, 0.6, 0.0]),
        "RShoulder": np.array([0.6, 0.6, 0.0]),
        "LElbow": np.array([0.3, 0.4, 0.0]),
        "RElbow": np.array([0.7, 0.4, 0.0]),
        "LWrist": np.array([0.25, 0.2, 0.0]),
        "RWrist": np.array([0.75, 0.2, 0.0]),
        "LHip": np.array([0.45, 0.9, 0.0]),
        "RHip": np.array([0.55, 0.9, 0.0]),
        "LKnee": np.array([0.45, 1.2, 0.0]),
        "RKnee": np.array([0.55, 1.2, 0.0]),
        "LAnkle": np.array([0.45, 1.5, 0.0]),
        "RAnkle": np.array([0.55, 1.5, 0.0]),
    }


# ===================================================================
# Test: _angle_between
# ===================================================================


class TestAngleBetween:
    """Tests for the low-level angle-between-vectors helper."""

    def test_perpendicular_vectors(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        assert pytest.approx(_angle_between(v1, v2), abs=1e-6) == np.pi / 2

    def test_parallel_vectors(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])
        assert pytest.approx(_angle_between(v1, v2), abs=1e-6) == 0.0

    def test_antiparallel_vectors(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([-1.0, 0.0, 0.0])
        assert pytest.approx(_angle_between(v1, v2), abs=1e-6) == np.pi

    def test_zero_vector_returns_nan(self) -> None:
        v1 = np.array([0.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        assert np.isnan(_angle_between(v1, v2))

    def test_45_degree_angle(self) -> None:
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 1.0, 0.0])
        assert pytest.approx(_angle_between(v1, v2), abs=1e-6) == np.pi / 4


# ===================================================================
# Test: _compute_flexion
# ===================================================================


class TestComputeFlexion:
    """Tests for the three-point flexion angle helper."""

    def test_straight_limb(self) -> None:
        """A straight limb should give pi radians."""
        proximal = np.array([0.0, 1.0, 0.0])
        joint = np.array([0.0, 0.0, 0.0])
        distal = np.array([0.0, -1.0, 0.0])
        assert (
            pytest.approx(_compute_flexion(proximal, joint, distal), abs=1e-6) == np.pi
        )

    def test_right_angle(self) -> None:
        """A 90-degree bend."""
        proximal = np.array([0.0, 1.0, 0.0])
        joint = np.array([0.0, 0.0, 0.0])
        distal = np.array([1.0, 0.0, 0.0])
        assert (
            pytest.approx(_compute_flexion(proximal, joint, distal), abs=1e-6)
            == np.pi / 2
        )


# ===================================================================
# Test: compute_joint_angles (MediaPipe naming)
# ===================================================================


class TestComputeJointAnglesMediaPipe:
    """Tests for joint angle computation with MediaPipe keypoints."""

    def test_all_angles_computed(self, mediapipe_keypoints: dict) -> None:
        angles = compute_joint_angles(mediapipe_keypoints)
        expected_keys = [
            "right_elbow_flexion",
            "left_elbow_flexion",
            "right_shoulder_flexion",
            "left_shoulder_flexion",
            "right_hip_flexion",
            "left_hip_flexion",
            "right_knee_flexion",
            "left_knee_flexion",
            "trunk_rotation",
        ]
        for key in expected_keys:
            assert key in angles, f"Missing angle: {key}"

    def test_angles_are_positive(self, mediapipe_keypoints: dict) -> None:
        angles = compute_joint_angles(mediapipe_keypoints)
        for name, value in angles.items():
            assert value >= 0, f"{name} is negative: {value}"

    def test_angles_within_range(self, mediapipe_keypoints: dict) -> None:
        """All angles should be between 0 and pi."""
        angles = compute_joint_angles(mediapipe_keypoints)
        for name, value in angles.items():
            assert 0 <= value <= np.pi + 1e-6, f"{name} out of range: {value}"

    def test_missing_keypoints_skipped(self) -> None:
        """Only elbow angles should appear if hip/knee are missing."""
        partial = {
            "left_shoulder": np.array([0.0, 1.0, 0.0]),
            "left_elbow": np.array([0.0, 0.5, 0.0]),
            "left_wrist": np.array([0.0, 0.0, 0.0]),
        }
        angles = compute_joint_angles(partial)
        assert "left_elbow_flexion" in angles
        assert "left_knee_flexion" not in angles
        assert "trunk_rotation" not in angles

    def test_empty_keypoints(self) -> None:
        angles = compute_joint_angles({})
        assert angles == {}


# ===================================================================
# Test: compute_joint_angles (OpenPose naming)
# ===================================================================


class TestComputeJointAnglesOpenPose:
    """Tests for joint angle computation with OpenPose keypoints."""

    def test_all_angles_with_mapping(self, openpose_keypoints: dict) -> None:
        angles = compute_joint_angles(
            openpose_keypoints, keypoint_mapping=OPENPOSE_TO_CANONICAL
        )
        assert "right_elbow_flexion" in angles
        assert "left_shoulder_flexion" in angles
        assert "trunk_rotation" in angles

    def test_consistent_with_mediapipe(
        self,
        mediapipe_keypoints: dict,
        openpose_keypoints: dict,
    ) -> None:
        """Same pose should give same angles regardless of naming."""
        mp_angles = compute_joint_angles(mediapipe_keypoints)
        op_angles = compute_joint_angles(
            openpose_keypoints, keypoint_mapping=OPENPOSE_TO_CANONICAL
        )

        for key in mp_angles:
            assert key in op_angles, f"OpenPose missing {key}"
            np.testing.assert_allclose(
                mp_angles[key], op_angles[key], atol=1e-10, err_msg=key
            )


# ===================================================================
# Test: Validation metrics
# ===================================================================


class TestJointAngleRMSE:
    """Tests for joint angle RMSE computation."""

    def test_perfect_match(self) -> None:
        angles = np.linspace(0, np.pi, 100)
        predicted = {"elbow": angles}
        reference = {"elbow": angles}
        result = compute_joint_angle_rmse(predicted, reference)
        assert result["aggregate_rmse_rad"] == pytest.approx(0.0, abs=1e-10)
        assert result["aggregate_grade"] == "excellent"

    def test_small_error_excellent(self) -> None:
        rng = np.random.default_rng(42)
        angles = np.linspace(0, np.pi, 100)
        noise = rng.normal(0, np.radians(1.0), 100)
        predicted = {"elbow": angles + noise}
        reference = {"elbow": angles}
        result = compute_joint_angle_rmse(predicted, reference)
        assert result["per_joint"]["elbow"]["grade"] == "excellent"

    def test_large_error_poor(self) -> None:
        angles = np.linspace(0, np.pi, 100)
        predicted = {"elbow": angles + np.radians(20.0)}
        reference = {"elbow": angles}
        result = compute_joint_angle_rmse(predicted, reference)
        assert result["per_joint"]["elbow"]["grade"] == "poor"

    def test_missing_reference_joint_ignored(self) -> None:
        predicted = {"elbow": np.zeros(10), "wrist": np.ones(10)}
        reference = {"elbow": np.zeros(10)}
        result = compute_joint_angle_rmse(predicted, reference)
        assert "wrist" not in result["per_joint"]


class TestMarkerRMSE:
    """Tests for marker position RMSE."""

    def test_perfect_match(self) -> None:
        markers = np.random.randn(50, 5, 3)
        result = compute_marker_rmse(markers, markers)
        assert result["aggregate_rmse_m"] == pytest.approx(0.0, abs=1e-10)
        assert result["aggregate_grade"] == "excellent"

    def test_small_error(self) -> None:
        rng = np.random.default_rng(42)
        markers = rng.standard_normal((50, 5, 3))
        noisy = markers + rng.normal(0, 0.005, markers.shape)
        result = compute_marker_rmse(noisy, markers)
        assert result["aggregate_rmse_m"] < 0.010
        assert result["aggregate_grade"] == "excellent"

    def test_empty_arrays(self) -> None:
        result = compute_marker_rmse(np.zeros((0, 5, 3)), np.zeros((0, 5, 3)))
        assert result["aggregate_grade"] == "poor"


class TestTemporalJitter:
    """Tests for temporal jitter computation."""

    def test_smooth_signal_low_jitter(self) -> None:
        """A slowly-varying signal should have lower jitter than noisy one."""
        t = np.linspace(0, 1, 100)
        # Very gentle motion: 0.1 radian amplitude
        angles = 0.1 * np.sin(2 * np.pi * 0.5 * t)
        result = compute_temporal_jitter({"joint": angles}, dt=1.0 / 100)
        # The jitter should be finite and smaller than a noisy signal
        assert result["aggregate_jitter_rad_s"] < 1.0
        assert result["aggregate_grade"] in ("excellent", "acceptable")

    def test_noisy_signal_high_jitter(self) -> None:
        """A random noise signal should have high jitter."""
        rng = np.random.default_rng(42)
        angles = rng.standard_normal(100)
        result = compute_temporal_jitter({"joint": angles}, dt=1.0 / 30)
        assert result["aggregate_jitter_rad_s"] > S3_TOLERANCES["jitter_excellent"]

    def test_too_few_frames(self) -> None:
        result = compute_temporal_jitter({"joint": np.array([1.0, 2.0])}, dt=0.01)
        assert result["aggregate_jitter_rad_s"] == 0.0


class TestFitQuality:
    """Tests for fit quality grading."""

    def test_excellent_fit(self) -> None:
        result = compute_fit_quality(
            r_squared=0.995, condition_number=50.0, rms_error=0.001
        )
        assert result["r_squared_grade"] == "excellent"
        assert result["condition_well_posed"] is True

    def test_poor_fit(self) -> None:
        result = compute_fit_quality(
            r_squared=0.80, condition_number=1e8, rms_error=0.5
        )
        assert result["r_squared_grade"] == "poor"
        assert result["condition_well_posed"] is False


class TestValidatePipelineOutput:
    """Tests for the full validation function."""

    def test_full_validation_excellent(self) -> None:
        """Perfect data should get excellent overall grade."""
        angles = np.linspace(0, np.pi, 100)
        report = validate_pipeline_output(
            joint_angles_series={"elbow": angles},
            reference_angles={"elbow": angles},
            dt=1.0 / 30,
            r_squared=0.999,
            condition_number=10.0,
            rms_error=0.001,
        )
        assert isinstance(report, ValidationReport)
        assert report.overall_grade == "excellent"

    def test_full_validation_poor(self) -> None:
        """Noisy data with bad fit should get poor grade."""
        rng = np.random.default_rng(0)
        angles = rng.standard_normal(100)
        ref = np.zeros(100)
        report = validate_pipeline_output(
            joint_angles_series={"elbow": angles},
            reference_angles={"elbow": ref},
            dt=1.0 / 30,
            r_squared=0.5,
            condition_number=1e9,
            rms_error=1.0,
        )
        assert report.overall_grade == "poor"

    def test_no_data_returns_poor_or_unknown(self) -> None:
        """With no pose data, fit defaults produce poor; grade reflects that."""
        report = validate_pipeline_output()
        # Default r_squared=0.0 and rms_error=inf yield poor fit quality
        assert report.overall_grade in ("poor", "unknown")

    def test_marker_validation_included(self) -> None:
        rng = np.random.default_rng(42)
        markers = rng.standard_normal((50, 5, 3))
        report = validate_pipeline_output(
            predicted_markers=markers,
            reference_markers=markers,
        )
        assert report.marker_metrics["aggregate_grade"] == "excellent"

    def test_s3_tolerance_constants_exist(self) -> None:
        """Verify all expected S3 tolerance constants are defined."""
        expected = [
            "joint_angle_rmse_excellent",
            "joint_angle_rmse_acceptable",
            "marker_rmse_excellent",
            "marker_rmse_acceptable",
            "jitter_excellent",
            "jitter_acceptable",
            "r_squared_excellent",
            "r_squared_acceptable",
        ]
        for key in expected:
            assert key in S3_TOLERANCES, f"Missing tolerance: {key}"
