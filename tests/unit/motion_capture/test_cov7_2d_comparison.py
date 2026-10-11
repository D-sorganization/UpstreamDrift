"""Unit tests for COV-7 2D comparison receipts and uncertainty propagation (#11275)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.comparison_2d import (
    Comparison2DReceipt,
    ComparisonLevel,
    L1ComparisonResult,
    L2ComparisonResult,
    MetricSpread,
    build_2d_comparison_receipt,
    compute_l1_envelope_comparison,
    compute_l2_paired_comparison,
    propagate_camera_uncertainty,
)
from src.motion_capture.reference.swing_pairing import (
    PairingConfidence,
    PairingDecisionStatus,
    SwingPairingResult,
)
from src.motion_capture.reference.virtual_camera_fit import SwingEnvelope2D


@pytest.fixture
def synthetic_phase_bins() -> np.ndarray:
    """Standard 50 phase-normalized bins in [0, 1]."""
    return np.linspace(0.0, 1.0, 50, dtype=float)


@pytest.fixture
def synthetic_landmark_names() -> tuple[str, ...]:
    """Four canonical test landmark names."""
    return ("wrist_lead", "wrist_trail", "shoulder_lead", "hip_lead")


@pytest.fixture
def synthetic_camera() -> PinholeCamera:
    """Synthetic calibrated pinhole camera."""
    matrix = np.array(
        [
            [1000.0, 0.0, 960.0],
            [0.0, 1000.0, 540.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    r_wc = np.eye(3, dtype=float)
    t_wc = np.array([0.0, 1.2, -3.0], dtype=float)
    return PinholeCamera(
        camera_id="cam_synthetic",
        matrix=matrix,
        rotation_world_from_camera=r_wc,
        translation_world_from_camera_m=t_wc,
        image_size_px=(1920, 1080),
    )


@pytest.fixture
def synthetic_envelope(
    synthetic_phase_bins: np.ndarray,
    synthetic_landmark_names: tuple[str, ...],
) -> SwingEnvelope2D:
    """Synthetic 2D swing envelope with p5/p50/p95 bands."""
    t_len = len(synthetic_phase_bins)
    k_len = len(synthetic_landmark_names)
    p50 = np.zeros((t_len, k_len, 2), dtype=float)
    for k in range(k_len):
        p50[:, k, 0] = 500.0 + 100.0 * np.sin(
            2 * np.pi * synthetic_phase_bins + k * 0.5
        )
        p50[:, k, 1] = 400.0 + 80.0 * np.cos(2 * np.pi * synthetic_phase_bins + k * 0.5)

    p5 = p50 - 15.0
    p95 = p50 + 15.0
    counts = np.full((t_len, k_len), 13, dtype=int)
    return SwingEnvelope2D(
        phase_bins=synthetic_phase_bins,
        p5=p5,
        p50=p50,
        p95=p95,
        sample_counts=counts,
        landmark_names=synthetic_landmark_names,
        p5_height_normalized=p5 / 1000.0,
        p50_height_normalized=p50 / 1000.0,
        p95_height_normalized=p95 / 1000.0,
    )


@pytest.fixture
def synthetic_paired_result() -> SwingPairingResult:
    """Pairing result indicating confidence >= tau_pair with capture-O-s01."""
    return SwingPairingResult(
        video_swing_id="cov-01-s01",
        status=PairingDecisionStatus.PAIRED,
        paired_capture_swing_id="capture-O-s01",
        confidence=PairingConfidence(
            margin=0.35,
            best_normalized_distance=0.25,
            second_best_normalized_distance=0.60,
            envelope_median_distance=1.0,
            tau_pair=0.20,
            epsilon=0.05,
        ),
        normalized_distances={"capture-O-s01": 0.25, "capture-O-s02": 0.60},
        raw_distances={"capture-O-s01": 0.25, "capture-O-s02": 0.60},
        receipt_hashes={"cov-01-s01": "hash_pair_01"},
    )


@pytest.fixture
def synthetic_unpaired_result() -> SwingPairingResult:
    """Pairing result indicating unpaired swing."""
    return SwingPairingResult(
        video_swing_id="cov-02-s01",
        status=PairingDecisionStatus.UNPAIRED,
        paired_capture_swing_id=None,
        confidence=PairingConfidence(
            margin=0.01,
            best_normalized_distance=1.25,
            second_best_normalized_distance=1.26,
            envelope_median_distance=1.0,
            tau_pair=0.20,
            epsilon=0.05,
        ),
        normalized_distances={"capture-O-s01": 1.25},
        raw_distances={"capture-O-s01": 1.25},
        receipt_hashes={"cov-02-s01": "hash_unpaired_02"},
    )


def test_synthetic_known_bias_recovered_as_median_residual(
    synthetic_phase_bins: np.ndarray,
    synthetic_landmark_names: tuple[str, ...],
    synthetic_paired_result: SwingPairingResult,
) -> None:
    """A known per-landmark bias of +5 px is recovered as a median residual of +5 +- tolerance."""
    t_len = len(synthetic_phase_bins)
    k_len = len(synthetic_landmark_names)

    # Reference trajectory
    ref_proj = np.zeros((t_len, k_len, 2), dtype=float)
    for k in range(k_len):
        ref_proj[:, k, 0] = 600.0 + 50.0 * np.sin(np.pi * synthetic_phase_bins)
        ref_proj[:, k, 1] = 450.0 + 30.0 * np.cos(np.pi * synthetic_phase_bins)

    # Injected known bias of +5.0 px on X axis for all landmarks
    observed_video = np.array(ref_proj, copy=True)
    observed_video[:, :, 0] += 5.0

    result = compute_l2_paired_comparison(
        video_landmarks=observed_video,
        projected_reference=ref_proj,
        pairing_result=synthetic_paired_result,
        landmark_names=synthetic_landmark_names,
        body_height_px=1000.0,
    )

    assert isinstance(result, L2ComparisonResult)
    assert result.residuals_px["p50"] == pytest.approx(5.0, abs=1e-2)
    for name in synthetic_landmark_names:
        lm_res = result.landmark_residuals_px[name]
        assert lm_res["p50"] == pytest.approx(5.0, abs=1e-2)
        # Normalized residual check (5 px / 1000 px = 0.005)
        assert result.residuals_norm["p50"] == pytest.approx(0.005, abs=1e-4)


def test_synthetic_unpaired_swing_raises_for_l2(
    synthetic_phase_bins: np.ndarray,
    synthetic_landmark_names: tuple[str, ...],
    synthetic_unpaired_result: SwingPairingResult,
) -> None:
    """An unpaired swing never produces L2 metrics (type-level or contract guard raises ValueError)."""
    t_len = len(synthetic_phase_bins)
    k_len = len(synthetic_landmark_names)
    pts = np.ones((t_len, k_len, 2), dtype=float) * 500.0

    with pytest.raises(ValueError, match="[Uu]npaired"):
        compute_l2_paired_comparison(
            video_landmarks=pts,
            projected_reference=pts,
            pairing_result=synthetic_unpaired_result,
            landmark_names=synthetic_landmark_names,
        )


def test_synthetic_leakage_guard_excludes_offset_calibration_frames(
    synthetic_phase_bins: np.ndarray,
    synthetic_landmark_names: tuple[str, ...],
    synthetic_paired_result: SwingPairingResult,
) -> None:
    """Frames used for offset fitting are absent from the evaluated set (leakage guard)."""
    t_len = len(synthetic_phase_bins)
    k_len = len(synthetic_landmark_names)
    ref_proj = np.full((t_len, k_len, 2), 500.0, dtype=float)
    observed = np.full((t_len, k_len, 2), 502.0, dtype=float)

    calibration_frames = (0, 1, 2, 3, 4)
    # Inject an enormous 500 px error specifically on calibration frames
    observed[0:5, :, :] += 500.0

    result = compute_l2_paired_comparison(
        video_landmarks=observed,
        projected_reference=ref_proj,
        pairing_result=synthetic_paired_result,
        calibration_frames=calibration_frames,
        landmark_names=synthetic_landmark_names,
    )

    # Calibration frames must be strictly excluded from evaluation
    assert result.calibration_frames_excluded == calibration_frames
    assert result.evaluated_frames_count == t_len - len(calibration_frames)
    # Worst residual in evaluated frames is 2.0 * sqrt(2) ~ 2.828 px, NOT > 500 px
    assert result.residuals_px["worst"] < 10.0

    # Leakage guard: explicitly passing calibration frame in evaluated frame list raises ValueError
    with pytest.raises(ValueError, match="[Ll]eakage guard"):
        compute_l2_paired_comparison(
            video_landmarks=observed,
            projected_reference=ref_proj,
            pairing_result=synthetic_paired_result,
            calibration_frames=calibration_frames,
            evaluated_frames=(2, 10, 20),  # Frame 2 is a calibration frame!
        )


def test_synthetic_occluded_landmark_excluded_from_visibility_weighted_metric(
    synthetic_phase_bins: np.ndarray,
    synthetic_landmark_names: tuple[str, ...],
    synthetic_paired_result: SwingPairingResult,
) -> None:
    """An occluded or not-visible landmark is excluded from visibility-weighted metric and counted in missingness, never zero-filled."""
    t_len = len(synthetic_phase_bins)
    k_len = len(synthetic_landmark_names)
    ref_proj = np.full((t_len, k_len, 2), 300.0, dtype=float)
    observed = np.full((t_len, k_len, 2), 304.0, dtype=float)

    # Landmark 0 is occluded (visibility 0.0) for the first 10 frames
    visibilities = np.ones((t_len, k_len), dtype=float)
    visibilities[0:10, 0] = 0.0
    # Set coordinates to NaN on occluded frames
    observed[0:10, 0, :] = np.nan

    result = compute_l2_paired_comparison(
        video_landmarks=observed,
        projected_reference=ref_proj,
        pairing_result=synthetic_paired_result,
        visibilities=visibilities,
        landmark_names=synthetic_landmark_names,
    )

    # Missingness report must count exactly 10 missing frames for landmark 0
    assert result.missingness_report["wrist_lead"] == 10
    assert result.missingness_report["wrist_trail"] == 0

    # Visibility-weighted RMSE must be computed only over valid, visible landmarks
    assert np.isfinite(result.visibility_weighted_rmse_px)
    assert np.isfinite(result.unweighted_rmse_px)
    # The occluded frames must not be filled with (0,0) (which would yield residuals ~ 300 px)
    assert result.visibility_weighted_rmse_px < 10.0


def test_synthetic_camera_uncertainty_propagation_wider_covariance_gives_wider_spread(
    synthetic_camera: PinholeCamera,
) -> None:
    """Camera-uncertainty propagation: a wider covariance gives a wider metric spread; difference smaller than spread is 'not resolvable'."""
    world_pts = np.array(
        [
            [-0.2, 1.2, 0.1],
            [0.2, 1.2, 0.1],
            [-0.1, 0.9, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=float,
    )

    cov_narrow = np.eye(6, dtype=float) * 1e-6
    cov_wide = np.eye(6, dtype=float) * 1e-2

    spread_narrow = propagate_camera_uncertainty(
        reference_points_3d=world_pts,
        camera=synthetic_camera,
        covariance=cov_narrow,
        sample_count=80,
        seed=42,
    )
    spread_wide = propagate_camera_uncertainty(
        reference_points_3d=world_pts,
        camera=synthetic_camera,
        covariance=cov_wide,
        sample_count=80,
        seed=42,
    )

    assert isinstance(spread_narrow, MetricSpread)
    assert isinstance(spread_wide, MetricSpread)
    assert spread_wide.spread > spread_narrow.spread

    # Test difference smaller than spread is marked 'not resolvable'
    small_diff = spread_wide.spread * 0.4
    res_unresolvable = propagate_camera_uncertainty(
        reference_points_3d=world_pts,
        camera=synthetic_camera,
        covariance=cov_wide,
        sample_count=80,
        difference=small_diff,
        seed=42,
    )
    assert not res_unresolvable.is_resolvable
    assert res_unresolvable.status == "not resolvable"

    # Test difference larger than spread is resolvable
    large_diff = spread_wide.spread * 2.5
    res_resolvable = propagate_camera_uncertainty(
        reference_points_3d=world_pts,
        camera=synthetic_camera,
        covariance=cov_wide,
        sample_count=80,
        difference=large_diff,
        seed=42,
    )
    assert res_resolvable.is_resolvable
    assert res_resolvable.status == "resolvable"


def test_synthetic_receipt_stale_input_hash_raises_refusal(
    synthetic_phase_bins: np.ndarray,
    synthetic_envelope: SwingEnvelope2D,
    synthetic_landmark_names: tuple[str, ...],
) -> None:
    """The receipt carries input hashes for observations, camera, pairing and profile. Stale hash raises refusal."""
    l1_res = compute_l1_envelope_comparison(
        video_landmarks=synthetic_envelope.p50,
        envelope=synthetic_envelope,
        video_swing_id="cov-01-s01",
        backend="mediapipe",
        landmark_names=synthetic_landmark_names,
    )

    valid_hashes = {
        "observations": "a" * 64,
        "camera": "b" * 64,
        "pairing": "c" * 64,
        "profile": "d" * 64,
    }

    # Successful receipt building
    receipt = build_2d_comparison_receipt(
        video_swing_id="cov-01-s01",
        backend="mediapipe",
        level=ComparisonLevel.L1,
        l1_result=l1_res,
        input_hashes=valid_hashes,
        expected_hashes=valid_hashes,
    )
    assert isinstance(receipt, Comparison2DReceipt)
    assert receipt.observations_hash == valid_hashes["observations"]

    # Stale/mismatched pairing hash raises refusal
    stale_expected = dict(valid_hashes)
    stale_expected["pairing"] = "f" * 64
    with pytest.raises(ValueError, match="[Ss]tale|mismatch"):
        build_2d_comparison_receipt(
            video_swing_id="cov-01-s01",
            backend="mediapipe",
            level=ComparisonLevel.L1,
            l1_result=l1_res,
            input_hashes=valid_hashes,
            expected_hashes=stale_expected,
        )


def test_synthetic_l1_envelope_inside_fraction_and_signed_distance(
    synthetic_envelope: SwingEnvelope2D,
    synthetic_landmark_names: tuple[str, ...],
) -> None:
    """L1 envelope comparison computes fraction inside envelope and signed distance to median."""
    # Observations exactly at median -> 100% inside, 0 signed distance
    res_center = compute_l1_envelope_comparison(
        video_landmarks=synthetic_envelope.p50,
        envelope=synthetic_envelope,
        landmark_names=synthetic_landmark_names,
        body_height_px=1000.0,
    )
    assert isinstance(res_center, L1ComparisonResult)
    assert res_center.fraction_inside_envelope == pytest.approx(1.0)
    assert res_center.signed_distance_to_median_px == pytest.approx(0.0, abs=1e-3)

    # Observations shifted by +5 px
    shifted = synthetic_envelope.p50 + 5.0
    res_shifted = compute_l1_envelope_comparison(
        video_landmarks=shifted,
        envelope=synthetic_envelope,
        landmark_names=synthetic_landmark_names,
        body_height_px=1000.0,
    )
    # Inside envelope since width is +-15 px
    assert res_shifted.fraction_inside_envelope == pytest.approx(1.0)
    assert res_shifted.signed_distance_to_median_px == pytest.approx(
        5.0 * np.sqrt(2), abs=1e-2
    )
    assert res_shifted.signed_distance_to_median_norm == pytest.approx(
        (5.0 * np.sqrt(2)) / 1000.0, abs=1e-4
    )
