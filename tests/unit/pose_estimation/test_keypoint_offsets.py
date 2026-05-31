"""Tests for calibratable detector-keypoint offsets."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.pose_estimation.keypoint_offsets import (
    KeypointOffsetSite,
    estimate_keypoint_offset,
    estimate_keypoint_offset_model,
)


def _rot_z(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_estimate_keypoint_offset_recovers_segment_frame_offset() -> None:
    offset_segment = np.array([0.04, -0.015, 0.02])
    rotations = np.stack([_rot_z(0.0), _rot_z(math.pi / 2.0), _rot_z(math.pi)])
    centers = np.array(
        [[0.5, 0.0, 1.0], [0.2, -0.2, 1.1], [-0.4, 0.1, 0.9]],
        dtype=float,
    )
    observed = centers + np.einsum("nij,j->ni", rotations, offset_segment)

    estimate = estimate_keypoint_offset(
        keypoint_name="right_hip",
        canonical_site="right_hip_center",
        segment_name="pelvis",
        joint_centers_world_m=centers,
        keypoints_world_m=observed,
        segment_rotations_world_from_segment=rotations,
        confidences=[0.9, 0.8, 0.7],
    )

    np.testing.assert_allclose(estimate.offset_m, offset_segment, atol=1e-12)
    assert estimate.sample_count == 3
    assert estimate.mean_confidence == pytest.approx(0.8)
    assert estimate.rms_residual_m < 1e-12
    np.testing.assert_allclose(
        estimate.predict_keypoint_m(centers[1], rotations[1]),
        observed[1],
        atol=1e-12,
    )


def test_estimate_keypoint_offset_filters_low_confidence_and_reports_uncertainty() -> (
    None
):
    true_offset = np.array([0.02, 0.01, -0.03])
    center = np.zeros((4, 3))
    rotations = np.repeat(np.eye(3)[None, :, :], 4, axis=0)
    noise = np.array(
        [[0.0, 0.0, 0.0], [0.002, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.002, 0.0, 0.0]]
    )
    observed = center + true_offset + noise

    estimate = estimate_keypoint_offset(
        keypoint_name="left_knee",
        canonical_site="left_knee_center",
        segment_name="left_shank",
        joint_centers_world_m=center,
        keypoints_world_m=observed,
        segment_rotations_world_from_segment=rotations,
        confidences=[1.0, 1.0, 0.05, 1.0],
        min_confidence=0.5,
    )

    np.testing.assert_allclose(estimate.offset_m, true_offset, atol=1e-12)
    assert estimate.sample_count == 3
    assert estimate.standard_error_m[0] > 0.0
    assert estimate.covariance_m2[0][0] > 0.0


def test_offset_model_predicts_and_residuals_by_site() -> None:
    site = KeypointOffsetSite(
        keypoint_name="right_shoulder",
        canonical_site="right_glenohumeral_center",
        segment_name="torso",
    )
    offset = np.array([0.03, 0.0, 0.01])
    centers = {"right_shoulder": np.array([[0.0, 0.0, 1.4], [0.1, 0.0, 1.4]])}
    rotations = {"torso": np.repeat(np.eye(3)[None, :, :], 2, axis=0)}
    observed = {"right_shoulder": centers["right_shoulder"] + offset}

    model = estimate_keypoint_offset_model(
        sites=[site],
        joint_centers_world_m=centers,
        keypoints_world_m=observed,
        segment_rotations_world_from_segment=rotations,
    )

    residuals = model.residuals_for_clip(
        joint_centers_world_m=centers,
        keypoints_world_m=observed,
        segment_rotations_world_from_segment=rotations,
    )
    np.testing.assert_allclose(residuals["right_shoulder"], np.zeros((2, 3)))
    assert model.offset_for("right_shoulder").canonical_site == site.canonical_site
    assert model.to_documentation_rows()[0]["segment_name"] == "torso"


def test_estimation_rejects_non_rotation_matrix() -> None:
    centers = np.zeros((1, 3))
    observed = np.zeros((1, 3))
    bad_rotation = np.array([[[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

    with pytest.raises(ValueError, match="proper rotation"):
        estimate_keypoint_offset(
            keypoint_name="nose",
            canonical_site="head_center",
            segment_name="head",
            joint_centers_world_m=centers,
            keypoints_world_m=observed,
            segment_rotations_world_from_segment=bad_rotation,
        )
