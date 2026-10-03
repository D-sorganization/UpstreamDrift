"""COV-6 Swing pairing with capture-O swings: similarity matrix, confidence and abstention (#11274).

Decides with evidence which (if any) capture-O swing each video swing matches,
evaluating phase-normalized DTW distance normalized by inter-capture variation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.motion_capture.reference.synchronization import EventAnchors, TimeMapping
from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.signal_toolkit.signal_processing import compute_dtw_distance

logger = get_logger(__name__)

DEFAULT_TAU_PAIR = 0.20
DEFAULT_EPSILON = 0.05
DEFAULT_REFERENCE_BACKEND = "reference"


class PairingDecisionStatus(str, Enum):
    """Decision status for video-to-capture swing pairing."""

    PAIRED = "paired"
    AMBIGUOUS = "ambiguous"
    UNPAIRED = "unpaired"


class PairingOptions(BaseModel):
    """Configurable options and frozen thresholds for swing pairing."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    tau_pair: float = Field(default=DEFAULT_TAU_PAIR, ge=0.0)
    epsilon: float = Field(default=DEFAULT_EPSILON, ge=0.0)
    reference_backend: str = DEFAULT_REFERENCE_BACKEND
    expected_receipt_hashes: Mapping[str, str] | None = None
    dtw_window: int | None = None


class SideEvidence(BaseModel):
    """Recorded side evidence that never overrides the distance rule."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    owner_recollection: str | None = None
    file_creation_time: str | None = None
    capture_swing_order_time: str | None = None
    club_speed_rank: int | None = None
    notes: str | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)


class PairingConfidence(BaseModel):
    """Pairing confidence metrics based on normalized distance margins."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    margin: float
    best_normalized_distance: float
    second_best_normalized_distance: float | None = None
    envelope_median_distance: float
    tau_pair: float = DEFAULT_TAU_PAIR
    epsilon: float = DEFAULT_EPSILON

    def __float__(self) -> float:
        return float(self.margin)

    def __ge__(self, other: Any) -> bool:
        return self.margin >= float(other)

    def __gt__(self, other: Any) -> bool:
        return self.margin > float(other)

    def __le__(self, other: Any) -> bool:
        return self.margin <= float(other)

    def __lt__(self, other: Any) -> bool:
        return self.margin < float(other)


@dataclass(frozen=True)
class SwingPairingFeatures:
    """Phase-normalized features and metadata for a swing."""

    swing_id: str
    phase_bins: np.ndarray
    features: np.ndarray
    backend: str = DEFAULT_REFERENCE_BACKEND
    receipt_hash: str = ""
    event_anchors: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.swing_id, str) or not self.swing_id.strip():
            require(False, "swing_id must be non-empty")
            raise ValueError("swing_id must be non-empty")
        if not isinstance(self.features, np.ndarray):
            require(False, "features must be a numpy ndarray")
            raise ValueError("features must be a numpy ndarray")
        if not np.isfinite(self.features).all():
            require(False, "Features must be finite (no NaN or Inf)")
            raise ValueError("Features must be finite (no NaN or Inf)")
        if not isinstance(self.phase_bins, np.ndarray):
            require(False, "phase_bins must be a numpy ndarray")
            raise ValueError("phase_bins must be a numpy ndarray")
        if not np.isfinite(self.phase_bins).all():
            require(False, "phase_bins must be finite")
            raise ValueError("phase_bins must be finite")
        if len(self.phase_bins) != len(self.features):
            require(False, "phase_bins length must match features length")
            raise ValueError("phase_bins length must match features length")


class SwingPairingResult(BaseModel):
    """Pairing decision, confidence, and time mapping for a single video swing."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    video_swing_id: str
    status: PairingDecisionStatus
    paired_capture_swing_id: str | None = None
    confidence: PairingConfidence
    normalized_distances: Mapping[str, float]
    raw_distances: Mapping[str, float]
    side_evidence: SideEvidence | None = None
    time_mapping: TimeMapping | None = None
    receipt_hashes: Mapping[str, str] = Field(default_factory=dict)


class PairingMatrix(BaseModel):
    """Full pairing matrix of video swings against capture-O swings."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    video_swing_ids: tuple[str, ...]
    capture_swing_ids: tuple[str, ...]
    envelope_median_distance: float
    tau_pair: float
    epsilon: float
    results: Mapping[str, SwingPairingResult]
    receipt_hashes: Mapping[str, str] = Field(default_factory=dict)

    def summary_counts(self) -> dict[str, int]:
        """Return counts categorized by decision status."""
        counts = {"paired": 0, "ambiguous": 0, "unpaired": 0}
        for res in self.results.values():
            counts[res.status.value] += 1
        return counts

    def get_result(self, video_swing_id: str) -> SwingPairingResult:
        """Lookup pairing result by video swing id."""
        if video_swing_id not in self.results:
            raise KeyError(f"Swing '{video_swing_id}' not found in pairing matrix")
        return self.results[video_swing_id]


def _compute_trajectory_dtw_distance(
    feat1: np.ndarray,
    feat2: np.ndarray,
    window: int | None = None,
) -> float:
    """Compute channel-averaged DTW distance between two feature trajectories."""
    f1 = feat1.reshape((len(feat1), -1))
    f2 = feat2.reshape((len(feat2), -1))
    if f1.shape[1] != f2.shape[1]:
        require(False, "Feature channel counts must match")
        raise ValueError("Feature channel counts must match")
    n_channels = f1.shape[1]
    total_dist = 0.0
    for d in range(n_channels):
        total_dist += compute_dtw_distance(f1[:, d], f2[:, d], window=window)
    return float(total_dist / n_channels)


def _validate_phase_alignment(
    swing_a: SwingPairingFeatures,
    swing_b: SwingPairingFeatures,
) -> None:
    """Assert identical phase discretization between two swings."""
    if len(swing_a.phase_bins) != len(swing_b.phase_bins) or not np.allclose(
        swing_a.phase_bins, swing_b.phase_bins
    ):
        require(False, "Mismatched phase bins between swings")
        raise ValueError("Mismatched phase bins between swings")


def _check_leakage(
    video_swing: SwingPairingFeatures,
    options: PairingOptions,
) -> None:
    """Enforce leakage guard: observations must come from reference backend."""
    if video_swing.backend != options.reference_backend:
        require(
            False,
            f"leakage guard: evaluation backend '{video_swing.backend}' cannot be used; "
            f"must use reference backend '{options.reference_backend}'",
        )
        raise ValueError(
            f"leakage guard: evaluation backend '{video_swing.backend}' cannot be used; "
            f"must use reference backend '{options.reference_backend}'"
        )
    if options.expected_receipt_hashes is not None:
        expected = options.expected_receipt_hashes.get(video_swing.swing_id)
        if expected is not None and video_swing.receipt_hash != expected:
            require(
                False,
                f"Receipt hash mismatch for '{video_swing.swing_id}': "
                f"expected '{expected}', got '{video_swing.receipt_hash}'",
            )
            raise ValueError(
                f"Receipt hash mismatch for '{video_swing.swing_id}': "
                f"expected '{expected}', got '{video_swing.receipt_hash}'"
            )


def compute_inter_capture_envelope_median(
    capture_swings: Mapping[str, SwingPairingFeatures],
    *,
    window: int | None = None,
) -> float:
    """Compute median DTW distance across all distinct pairs of capture swings."""
    keys = sorted(capture_swings.keys())
    if len(keys) < 2:
        require(False, "At least two capture swings required for inter-swing envelope")
        raise ValueError(
            "At least two capture swings required for inter-swing envelope"
        )

    distances: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sw_i = capture_swings[keys[i]]
            sw_j = capture_swings[keys[j]]
            _validate_phase_alignment(sw_i, sw_j)
            d = _compute_trajectory_dtw_distance(sw_i.features, sw_j.features, window)
            distances.append(d)

    med = float(np.median(distances))
    return max(med, 1e-6)


def _build_time_mapping(
    paired_capture_id: str,
    video_anchors: Mapping[str, float] | None,
    capture_anchors_map: Mapping[str, Mapping[str, float]] | None,
    capture_swings: Mapping[str, SwingPairingFeatures],
) -> TimeMapping | None:
    """Construct TimeMapping from paired event anchors if available."""
    if not video_anchors:
        return None
    ref_anchors: Mapping[str, float] | None = None
    if capture_anchors_map and paired_capture_id in capture_anchors_map:
        ref_anchors = capture_anchors_map[paired_capture_id]
    elif (
        paired_capture_id in capture_swings
        and capture_swings[paired_capture_id].event_anchors
    ):
        ref_anchors = capture_swings[paired_capture_id].event_anchors

    if not ref_anchors:
        return None

    try:
        anchors = EventAnchors(reference=ref_anchors, scene=video_anchors)
        return TimeMapping(event_anchors=anchors)
    except (ValueError, TypeError) as err:
        logger.warning(
            "Failed to construct EventAnchors for paired swing %s: %s",
            paired_capture_id,
            err,
        )
        return None


def pair_video_swing(
    video_swing: SwingPairingFeatures,
    capture_swings: Mapping[str, SwingPairingFeatures],
    *,
    options: PairingOptions | None = None,
    side_evidence: SideEvidence | None = None,
    capture_event_anchors: Mapping[str, Mapping[str, float]] | None = None,
    inter_capture_median: float | None = None,
) -> SwingPairingResult:
    """Pair a single video swing against capture swings with confidence and abstention."""
    opts = options or PairingOptions()
    _check_leakage(video_swing, opts)

    envelope_median = (
        inter_capture_median
        if inter_capture_median is not None and inter_capture_median > 0
        else compute_inter_capture_envelope_median(
            capture_swings, window=opts.dtw_window
        )
    )

    raw_distances: dict[str, float] = {}
    norm_distances: dict[str, float] = {}

    for cap_id, cap_swing in capture_swings.items():
        _validate_phase_alignment(video_swing, cap_swing)
        raw_d = _compute_trajectory_dtw_distance(
            video_swing.features, cap_swing.features, window=opts.dtw_window
        )
        raw_distances[cap_id] = raw_d
        norm_distances[cap_id] = raw_d / envelope_median

    sorted_candidates = sorted(
        norm_distances.items(), key=lambda item: (item[1], item[0])
    )
    best_id, best_norm = sorted_candidates[0]
    second_norm = sorted_candidates[1][1] if len(sorted_candidates) > 1 else None

    margin = float(second_norm - best_norm) if second_norm is not None else float("inf")
    if math.isinf(margin):
        margin = 1.0

    confidence = PairingConfidence(
        margin=margin,
        best_normalized_distance=best_norm,
        second_best_normalized_distance=second_norm,
        envelope_median_distance=envelope_median,
        tau_pair=opts.tau_pair,
        epsilon=opts.epsilon,
    )

    if best_norm >= 1.0:
        status = PairingDecisionStatus.UNPAIRED
        paired_id: str | None = None
    elif margin < opts.epsilon or margin < opts.tau_pair:
        status = PairingDecisionStatus.AMBIGUOUS
        paired_id = None
    else:
        status = PairingDecisionStatus.PAIRED
        paired_id = best_id

    time_mapping = (
        _build_time_mapping(
            paired_id,
            video_swing.event_anchors,
            capture_event_anchors,
            capture_swings,
        )
        if status == PairingDecisionStatus.PAIRED and paired_id is not None
        else None
    )

    hashes = {video_swing.swing_id: video_swing.receipt_hash}
    for k, v in capture_swings.items():
        if v.receipt_hash:
            hashes[k] = v.receipt_hash

    return SwingPairingResult(
        video_swing_id=video_swing.swing_id,
        status=status,
        paired_capture_swing_id=paired_id,
        confidence=confidence,
        normalized_distances=norm_distances,
        raw_distances=raw_distances,
        side_evidence=side_evidence,
        time_mapping=time_mapping,
        receipt_hashes=hashes,
    )


def build_pairing_matrix(
    video_swings: Sequence[SwingPairingFeatures],
    capture_swings: Mapping[str, SwingPairingFeatures],
    *,
    options: PairingOptions | None = None,
    side_evidence: Mapping[str, SideEvidence] | None = None,
    capture_event_anchors: Mapping[str, Mapping[str, float]] | None = None,
    receipt_hashes: Mapping[str, str] | None = None,
) -> PairingMatrix:
    """Build full pairing matrix of multiple video swings against capture swings."""
    opts = options or PairingOptions()
    median_dist = compute_inter_capture_envelope_median(
        capture_swings, window=opts.dtw_window
    )

    results: dict[str, SwingPairingResult] = {}
    all_receipt_hashes: dict[str, str] = dict(receipt_hashes or {})

    for video_swing in video_swings:
        evidence = side_evidence.get(video_swing.swing_id) if side_evidence else None
        res = pair_video_swing(
            video_swing,
            capture_swings,
            options=opts,
            side_evidence=evidence,
            capture_event_anchors=capture_event_anchors,
            inter_capture_median=median_dist,
        )
        results[video_swing.swing_id] = res
        all_receipt_hashes.update(res.receipt_hashes)

    return PairingMatrix(
        video_swing_ids=tuple(v.swing_id for v in video_swings),
        capture_swing_ids=tuple(sorted(capture_swings.keys())),
        envelope_median_distance=median_dist,
        tau_pair=opts.tau_pair,
        epsilon=opts.epsilon,
        results=results,
        receipt_hashes=all_receipt_hashes,
    )
