"""Unit tests for COV-6 swing pairing, similarity matrix, confidence and abstention (#11274)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.motion_capture.reference.swing_pairing import (
    PairingConfidence,
    PairingDecisionStatus,
    PairingMatrix,
    PairingOptions,
    SideEvidence,
    SwingPairingFeatures,
    SwingPairingResult,
    build_pairing_matrix,
    pair_video_swing,
)
from src.motion_capture.reference.synchronization import TimeMapping


@pytest.fixture
def synthetic_phase_bins() -> np.ndarray:
    """Standard 50 phase-normalized bins in [0, 1]."""
    return np.linspace(0.0, 1.0, 50, dtype=float)


@pytest.fixture
def synthetic_capture_swings(
    synthetic_phase_bins: np.ndarray,
) -> dict[str, SwingPairingFeatures]:
    """Five distinct synthetic capture swings with well-separated trajectories."""
    t = synthetic_phase_bins
    swings: dict[str, SwingPairingFeatures] = {}
    for k in range(5):
        swing_id = f"capture-O-s{k + 1:02d}"
        # Distinct phase offset and amplitude for each capture swing
        feat = np.column_stack(
            [
                np.sin(2 * np.pi * t + k * (np.pi / 4)),
                np.cos(2 * np.pi * t + k * (np.pi / 4)),
            ]
        )
        swings[swing_id] = SwingPairingFeatures(
            swing_id=swing_id,
            phase_bins=synthetic_phase_bins,
            features=feat,
            backend="reference",
            receipt_hash=f"ref_receipt_hash_{k + 1:02d}",
        )
    return swings


def test_synthetic_projected_swing_with_noise_pairs_with_high_confidence(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Projecting swing #3 with small noise pairs to #3 with confidence >= tau_pair."""
    rng = np.random.default_rng(42)
    s3_features = synthetic_capture_swings["capture-O-s03"].features
    noise = rng.normal(0.0, 0.01, size=s3_features.shape)
    video_features = s3_features + noise

    video_swing = SwingPairingFeatures(
        swing_id="cov-03-s01",
        phase_bins=synthetic_phase_bins,
        features=video_features,
        backend="reference",
        receipt_hash="ref_receipt_hash_03",
    )

    result = pair_video_swing(video_swing, synthetic_capture_swings)

    assert result.status == PairingDecisionStatus.PAIRED
    assert result.paired_capture_swing_id == "capture-O-s03"
    assert result.confidence.margin >= 0.20
    assert float(result.confidence) >= 0.20
    assert result.confidence.best_normalized_distance < 1.0


def test_synthetic_average_swing_is_ambiguous(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """A video synthesized as the exact average of #2 and #3 yields ambiguous abstention."""
    s2_features = synthetic_capture_swings["capture-O-s02"].features
    s3_features = synthetic_capture_swings["capture-O-s03"].features
    video_features = 0.5 * (s2_features + s3_features)

    video_swing = SwingPairingFeatures(
        swing_id="cov-02-s01",
        phase_bins=synthetic_phase_bins,
        features=video_features,
        backend="reference",
        receipt_hash="ref_receipt_hash_video",
    )

    result = pair_video_swing(video_swing, synthetic_capture_swings)

    assert result.status == PairingDecisionStatus.AMBIGUOUS
    assert result.paired_capture_swing_id is None
    # Top two normalized distances must be within epsilon (default 0.05) or margin < tau_pair
    assert result.confidence.margin < 0.20


def test_synthetic_unrelated_motion_is_unpaired(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """A video from an unrelated motion (far outside envelope median) yields unpaired."""
    t = synthetic_phase_bins
    # Unrelated high-amplitude, high-frequency motion completely outside inter-swing envelope
    unrelated_features = np.column_stack(
        [
            10.0 + 5.0 * np.sin(20 * np.pi * t),
            10.0 + 5.0 * np.cos(20 * np.pi * t),
        ]
    )

    video_swing = SwingPairingFeatures(
        swing_id="cov-unrelated",
        phase_bins=synthetic_phase_bins,
        features=unrelated_features,
        backend="reference",
        receipt_hash="ref_receipt_hash_video",
    )

    result = pair_video_swing(video_swing, synthetic_capture_swings)

    assert result.status == PairingDecisionStatus.UNPAIRED
    assert result.paired_capture_swing_id is None
    # No capture swing closer than envelope median (best normalized distance >= 1.0)
    assert result.confidence.best_normalized_distance >= 1.0


def test_permuting_capture_swings_preserves_pairing_order_invariance(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Permuting the capture-swing dictionary order does not change pairing decision."""
    rng = np.random.default_rng(123)
    s3_features = synthetic_capture_swings["capture-O-s03"].features
    noise = rng.normal(0.0, 0.01, size=s3_features.shape)
    video_swing = SwingPairingFeatures(
        swing_id="cov-perm-test",
        phase_bins=synthetic_phase_bins,
        features=s3_features + noise,
        backend="reference",
        receipt_hash="ref_receipt_hash_video",
    )

    keys = list(synthetic_capture_swings.keys())
    reversed_swings = {k: synthetic_capture_swings[k] for k in reversed(keys)}
    shuffled_keys = [keys[2], keys[0], keys[4], keys[1], keys[3]]
    shuffled_swings = {k: synthetic_capture_swings[k] for k in shuffled_keys}

    res_original = pair_video_swing(video_swing, synthetic_capture_swings)
    res_reversed = pair_video_swing(video_swing, reversed_swings)
    res_shuffled = pair_video_swing(video_swing, shuffled_swings)

    assert res_original.status == res_reversed.status == res_shuffled.status
    assert (
        res_original.paired_capture_swing_id
        == res_reversed.paired_capture_swing_id
        == res_shuffled.paired_capture_swing_id
    )
    assert np.isclose(res_original.confidence.margin, res_reversed.confidence.margin)
    assert np.isclose(res_original.confidence.margin, res_shuffled.confidence.margin)


def test_leakage_guard_rejects_evaluation_backend_observations(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Pairing never reads evaluation backend observations unless reference backend."""
    video_swing_eval = SwingPairingFeatures(
        swing_id="cov-eval-backend",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s01"].features,
        backend="rtmpose",  # Evaluation backend under test, not reference backend
        receipt_hash="eval_receipt_hash_99",
    )

    with pytest.raises(ValueError, match="leakage guard"):
        pair_video_swing(
            video_swing_eval,
            synthetic_capture_swings,
            options=PairingOptions(reference_backend="reference"),
        )


def test_leakage_guard_rejects_mismatched_reference_receipt_hash(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Pairing rejects observations if reference receipt hash doesn't match expected."""
    video_swing = SwingPairingFeatures(
        swing_id="cov-hash-mismatch",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s01"].features,
        backend="reference",
        receipt_hash="unexpected_unverified_hash",
    )

    with pytest.raises(ValueError, match="Receipt hash mismatch"):
        pair_video_swing(
            video_swing,
            synthetic_capture_swings,
            options=PairingOptions(
                reference_backend="reference",
                expected_receipt_hashes={"cov-hash-mismatch": "expected_valid_hash"},
            ),
        )


def test_nonfinite_features_raise_value_error(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Nonfinite values (NaN or Inf) in features raise ValueError."""
    bad_features = np.zeros((len(synthetic_phase_bins), 2), dtype=float)
    bad_features[10, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        SwingPairingFeatures(
            swing_id="cov-nan",
            phase_bins=synthetic_phase_bins,
            features=bad_features,
            backend="reference",
            receipt_hash="ref_receipt_hash",
        )

    bad_features_inf = np.zeros((len(synthetic_phase_bins), 2), dtype=float)
    bad_features_inf[5, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        SwingPairingFeatures(
            swing_id="cov-inf",
            phase_bins=synthetic_phase_bins,
            features=bad_features_inf,
            backend="reference",
            receipt_hash="ref_receipt_hash",
        )


def test_mismatched_phase_bins_raise_value_error(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Mismatched phase bins between video and capture swings raise ValueError."""
    diff_phase_bins = np.linspace(0.0, 1.0, 60, dtype=float)
    features_60 = np.zeros((60, 2), dtype=float)

    video_swing_mismatched = SwingPairingFeatures(
        swing_id="cov-diff-bins",
        phase_bins=diff_phase_bins,
        features=features_60,
        backend="reference",
        receipt_hash="ref_receipt_hash",
    )

    with pytest.raises(ValueError, match="phase"):
        pair_video_swing(video_swing_mismatched, synthetic_capture_swings)


def test_time_mapping_built_for_paired_swing_with_event_anchors(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Paired swing constructs TimeMapping from event anchors when provided."""
    s1_features = synthetic_capture_swings["capture-O-s01"].features
    video_swing = SwingPairingFeatures(
        swing_id="cov-01-s01",
        phase_bins=synthetic_phase_bins,
        features=s1_features,
        backend="reference",
        receipt_hash="ref_receipt_hash_01",
        event_anchors={"address": 0.5, "top": 1.2, "impact": 1.5, "finish": 2.2},
    )

    capture_anchors = {
        "capture-O-s01": {"address": 10.0, "top": 10.8, "impact": 11.1, "finish": 11.9}
    }

    result = pair_video_swing(
        video_swing,
        synthetic_capture_swings,
        capture_event_anchors=capture_anchors,
    )

    assert result.status == PairingDecisionStatus.PAIRED
    assert result.time_mapping is not None
    assert isinstance(result.time_mapping, TimeMapping)
    # Check that reference address maps close to scene address
    t_scene_address = result.time_mapping.reference_to_scene(10.0)
    assert np.isclose(t_scene_address, 0.5)


def test_build_pairing_matrix_summary_and_serialization(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """BuildPairingMatrix creates a complete matrix with counts and JSON serialization."""
    s1_features = synthetic_capture_swings["capture-O-s01"].features
    s2_features = synthetic_capture_swings["capture-O-s02"].features
    s3_features = synthetic_capture_swings["capture-O-s03"].features

    video1 = SwingPairingFeatures(
        swing_id="cov-01-s01",
        phase_bins=synthetic_phase_bins,
        features=s1_features,
        backend="reference",
        receipt_hash="ref_hash_01",
    )
    video2_avg = SwingPairingFeatures(
        swing_id="cov-02-s01",
        phase_bins=synthetic_phase_bins,
        features=0.5 * (s2_features + s3_features),
        backend="reference",
        receipt_hash="ref_hash_02",
    )
    video3_unrel = SwingPairingFeatures(
        swing_id="cov-03-s01",
        phase_bins=synthetic_phase_bins,
        features=100.0 + s1_features,
        backend="reference",
        receipt_hash="ref_hash_03",
    )

    side_evidence = {
        "cov-01-s01": SideEvidence(
            owner_recollection="likely swing 1", club_speed_rank=1
        )
    }

    matrix = build_pairing_matrix(
        video_swings=[video1, video2_avg, video3_unrel],
        capture_swings=synthetic_capture_swings,
        side_evidence=side_evidence,
    )

    assert isinstance(matrix, PairingMatrix)
    counts = matrix.summary_counts()
    assert counts["paired"] == 1
    assert counts["ambiguous"] == 1
    assert counts["unpaired"] == 1

    dumped = matrix.model_dump()
    assert "results" in dumped
    assert "cov-01-s01" in dumped["results"]
    assert dumped["results"]["cov-01-s01"]["status"] == "paired"


def test_fewer_than_two_capture_swings_raises_value_error(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Pairing with fewer than two capture swings fails because envelope cannot be estimated."""
    single_capture = {"capture-O-s01": synthetic_capture_swings["capture-O-s01"]}
    video_swing = SwingPairingFeatures(
        swing_id="cov-single",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s01"].features,
    )
    with pytest.raises(ValueError, match="At least two capture swings"):
        pair_video_swing(video_swing, single_capture)


def test_pairing_matrix_missing_key_raises_key_error(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Looking up nonexistent swing ID in PairingMatrix raises KeyError."""
    matrix = build_pairing_matrix(
        video_swings=[],
        capture_swings=synthetic_capture_swings,
    )
    with pytest.raises(KeyError, match="not found"):
        matrix.get_result("nonexistent-swing")


def test_leakage_guard_rejects_empty_receipt_hash_when_expected_map_provided(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """When expected receipt hashes are provided, empty hash on any participant raises ValueError."""
    # Video swing has empty hash
    v_empty = SwingPairingFeatures(
        swing_id="cov-v-empty",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s01"].features,
        backend="reference",
        receipt_hash="",
    )
    expected_map = {
        "cov-v-empty": "hash_v",
        "capture-O-s01": "ref_receipt_hash_01",
        "capture-O-s02": "ref_receipt_hash_02",
        "capture-O-s03": "ref_receipt_hash_03",
        "capture-O-s04": "ref_receipt_hash_04",
        "capture-O-s05": "ref_receipt_hash_05",
    }
    with pytest.raises(
        ValueError, match="Receipt hash for 'cov-v-empty' must be non-empty"
    ):
        pair_video_swing(
            v_empty,
            synthetic_capture_swings,
            options=PairingOptions(expected_receipt_hashes=expected_map),
        )

    # Capture swing has empty hash
    v_valid = SwingPairingFeatures(
        swing_id="cov-v-valid",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s01"].features,
        backend="reference",
        receipt_hash="hash_v",
    )
    tampered_captures = dict(synthetic_capture_swings)
    tampered_captures["capture-O-s02"] = SwingPairingFeatures(
        swing_id="capture-O-s02",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s02"].features,
        backend="reference",
        receipt_hash="",
    )
    expected_map_valid = dict(expected_map)
    expected_map_valid["cov-v-valid"] = "hash_v"
    with pytest.raises(
        ValueError, match="Receipt hash for 'capture-O-s02' must be non-empty"
    ):
        pair_video_swing(
            v_valid,
            tampered_captures,
            options=PairingOptions(expected_receipt_hashes=expected_map_valid),
        )


def test_leakage_guard_rejects_missing_swing_id_in_expected_hashes(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """When expected receipt hashes map is provided, any participant absent from the map raises ValueError."""
    v_valid = SwingPairingFeatures(
        swing_id="cov-v-valid",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s01"].features,
        backend="reference",
        receipt_hash="hash_v",
    )
    # Expected map only has video swing, missing capture swings
    partial_map = {"cov-v-valid": "hash_v"}
    with pytest.raises(ValueError, match="missing from expected_receipt_hashes"):
        pair_video_swing(
            v_valid,
            synthetic_capture_swings,
            options=PairingOptions(expected_receipt_hashes=partial_map),
        )


def test_leakage_guard_rejects_tampered_capture_swing_receipt_hash(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Tampered hash on a capture swing raises ValueError even when video swing matches."""
    v_valid = SwingPairingFeatures(
        swing_id="cov-v-valid",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s01"].features,
        backend="reference",
        receipt_hash="hash_v",
    )
    tampered_captures = dict(synthetic_capture_swings)
    tampered_captures["capture-O-s03"] = SwingPairingFeatures(
        swing_id="capture-O-s03",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s03"].features,
        backend="reference",
        receipt_hash="TAMPERED_HASH",
    )
    expected_map = {
        "cov-v-valid": "hash_v",
        "capture-O-s01": "ref_receipt_hash_01",
        "capture-O-s02": "ref_receipt_hash_02",
        "capture-O-s03": "ref_receipt_hash_03",
        "capture-O-s04": "ref_receipt_hash_04",
        "capture-O-s05": "ref_receipt_hash_05",
    }
    with pytest.raises(ValueError, match="Receipt hash mismatch for 'capture-O-s03'"):
        pair_video_swing(
            v_valid,
            tampered_captures,
            options=PairingOptions(expected_receipt_hashes=expected_map),
        )


def test_pairing_matrix_receipt_hashes_preserves_all_participants_including_empty(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Result receipt_hashes contains all participants, including capture swings with empty hashes."""
    v_swing = SwingPairingFeatures(
        swing_id="cov-v",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s01"].features,
        backend="reference",
        receipt_hash="",
    )
    captures = dict(synthetic_capture_swings)
    captures["capture-O-s02"] = SwingPairingFeatures(
        swing_id="capture-O-s02",
        phase_bins=synthetic_phase_bins,
        features=synthetic_capture_swings["capture-O-s02"].features,
        backend="reference",
        receipt_hash="",
    )

    result = pair_video_swing(v_swing, captures)
    assert "cov-v" in result.receipt_hashes
    assert result.receipt_hashes["cov-v"] == ""
    assert "capture-O-s02" in result.receipt_hashes
    assert result.receipt_hashes["capture-O-s02"] == ""
    for k in synthetic_capture_swings:
        assert k in result.receipt_hashes


def test_single_candidate_capture_swing_abstains_without_explicit_opt_in(
    synthetic_phase_bins: np.ndarray,
    synthetic_capture_swings: dict[str, SwingPairingFeatures],
) -> None:
    """Single candidate capture swing abstains by default, and only pairs when allow_single_candidate=True."""
    c1 = synthetic_capture_swings["capture-O-s01"]
    v_matching = SwingPairingFeatures(
        swing_id="cov-v-match",
        phase_bins=synthetic_phase_bins,
        features=c1.features,
        backend="reference",
        receipt_hash="hash_v",
    )
    single_capture = {"capture-O-s01": c1}

    # Default options: allow_single_candidate is False. Even though distance is 0, must abstain.
    result_default = pair_video_swing(
        v_matching,
        single_capture,
        inter_capture_median=1.0,
    )
    assert result_default.status == PairingDecisionStatus.AMBIGUOUS
    assert result_default.paired_capture_swing_id is None
    assert result_default.confidence.margin == 0.0

    # With allow_single_candidate=True: pairs with candidate
    opt_in = PairingOptions(allow_single_candidate=True)
    result_opt_in = pair_video_swing(
        v_matching,
        single_capture,
        options=opt_in,
        inter_capture_median=1.0,
    )
    assert result_opt_in.status == PairingDecisionStatus.PAIRED
    assert result_opt_in.paired_capture_swing_id == "capture-O-s01"
    assert result_opt_in.confidence.margin >= opt_in.tau_pair
