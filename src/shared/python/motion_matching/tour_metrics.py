"""Standardized shared evaluation metrics for tour motion matching.

Computes:
1. whole_marker_rmse_m: Full trial marker tracking RMS across all valid tracked markers
2. early_marker_rmse_m: Early prefix tracking RMS (t <= 0.60 s, address / backswing)
3. terminal_marker_rmse_m: Full-marker terminal window RMS (never silently drops head)
4. club_marker_rmse_m: Marker tracking RMS specifically for the club cluster (Marker_2 and Marker_3)
5. pelvis_yaw_rmse_rad: Orientation RMSE in the horizontal transverse plane (pelvis heading)

MS-61 (#10348) also discloses dual terminal diagnostics:
- terminal_full_marker_rmse_m (alias of terminal_marker_rmse_m)
- terminal_body_excluding_head_rmse_m (reduced-model diagnostic only)
- terminal_head_cluster_rmse_m
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    TourCapture,
)

HEAD_MARKER_LABELS: tuple[str, ...] = tuple(MARKER_SEGMENTS["head"])


@dataclass(frozen=True)
class SharedMetrics:
    """Shared kinematic metrics; terminal fields always disclose the head cluster."""

    whole_marker_rmse_m: float
    early_marker_rmse_m: float
    terminal_marker_rmse_m: float
    club_marker_rmse_m: float
    pelvis_yaw_rmse_rad: float
    terminal_body_excluding_head_rmse_m: float = 0.0
    terminal_head_cluster_rmse_m: float = 0.0

    @property
    def terminal_full_marker_rmse_m(self) -> float:
        """Full-marker terminal RMS (authoritative; identical to terminal_marker_rmse_m)."""
        return self.terminal_marker_rmse_m

    def as_dict(self) -> dict[str, float]:
        return {
            "whole_marker_rmse_m": self.whole_marker_rmse_m,
            "early_marker_rmse_m": self.early_marker_rmse_m,
            "terminal_marker_rmse_m": self.terminal_marker_rmse_m,
            "terminal_full_marker_rmse_m": self.terminal_full_marker_rmse_m,
            "terminal_body_excluding_head_rmse_m": (
                self.terminal_body_excluding_head_rmse_m
            ),
            "terminal_head_cluster_rmse_m": self.terminal_head_cluster_rmse_m,
            "club_marker_rmse_m": self.club_marker_rmse_m,
            "pelvis_yaw_rmse_rad": self.pelvis_yaw_rmse_rad,
        }


def _pelvis_yaw(points: np.ndarray, wl_idx: int, wr_idx: int) -> np.ndarray:
    """Compute pelvis yaw angle (radians) in the transverse plane from WaistLeft and WaistRight."""
    delta = points[:, wl_idx, :] - points[:, wr_idx, :]
    return np.arctan2(delta[:, 0], delta[:, 2])


def _rms_from_mask(dist_sq: np.ndarray, mask: np.ndarray) -> float:
    errs = dist_sq[mask]
    return float(np.sqrt(np.mean(errs))) if len(errs) > 0 else 0.0


def _terminal_partition_rmse(
    dist_sq: np.ndarray,
    valid: np.ndarray,
    labels: Sequence[str],
    term_start: int,
) -> tuple[float, float, float]:
    """Return (full, head-cluster, body-excluding-head) terminal-window RMSE."""
    term_mask = np.zeros_like(valid)
    term_mask[term_start:, :] = valid[term_start:, :]
    head_set = set(HEAD_MARKER_LABELS)
    head_indices = [i for i, label in enumerate(labels) if label in head_set]
    body_indices = [i for i, label in enumerate(labels) if label not in head_set]
    head_term_mask = np.zeros_like(valid)
    body_term_mask = np.zeros_like(valid)
    if head_indices:
        head_term_mask[term_start:, :][:, head_indices] = valid[term_start:, :][
            :, head_indices
        ]
    if body_indices:
        body_term_mask[term_start:, :][:, body_indices] = valid[term_start:, :][
            :, body_indices
        ]
    return (
        _rms_from_mask(dist_sq, term_mask),
        _rms_from_mask(dist_sq, head_term_mask),
        _rms_from_mask(dist_sq, body_term_mask),
    )


def _club_rmse(dist_sq: np.ndarray, valid: np.ndarray, labels: Sequence[str]) -> float:
    club_labels = set(MARKER_SEGMENTS["club"])
    club_indices = [i for i, label in enumerate(labels) if label in club_labels]
    if not club_indices:
        return 0.0
    club_mask = np.zeros_like(valid)
    club_mask[:, club_indices] = valid[:, club_indices]
    return _rms_from_mask(dist_sq, club_mask)


def _pelvis_yaw_rmse(
    obs: np.ndarray,
    pred: np.ndarray,
    valid: np.ndarray,
    labels: Sequence[str],
) -> float:
    if "WaistLeft" not in labels or "WaistRight" not in labels:
        return 0.0
    wl_idx = labels.index("WaistLeft")
    wr_idx = labels.index("WaistRight")
    valid_both = valid[:, wl_idx] & valid[:, wr_idx]
    if not np.any(valid_both):
        return 0.0
    obs_yaw = _pelvis_yaw(obs[valid_both], wl_idx, wr_idx)
    pred_yaw = _pelvis_yaw(pred[valid_both], wl_idx, wr_idx)
    angle_diff = np.remainder(pred_yaw - obs_yaw + np.pi, 2 * np.pi) - np.pi
    return float(np.sqrt(np.mean(angle_diff**2)))


def compute_shared_metrics(
    capture: TourCapture,
    predicted_points_m: np.ndarray,
    *,
    tracked_labels: Sequence[str] | None = None,
    early_cutoff_s: float = 0.60,
    terminal_ratio: float = 0.10,
) -> SharedMetrics:
    """Compute the five standardized kinematic metrics on predicted vs observed markers.

    Preconditions:
    - predicted_points_m must have shape (frames, markers, 3) matching capture
    - predicted_points_m must be finite where capture is valid

    Returns:
    - SharedMetrics instance with the 5 scalar metrics
    """
    pred = np.asarray(predicted_points_m, dtype=np.float64)
    expected_shape = (capture.frames, len(capture.labels), 3)
    if pred.shape != expected_shape:
        raise ValueError(f"Shape mismatch: expected {expected_shape}, got {pred.shape}")

    obs = capture.points_m
    if tracked_labels is not None:
        tracked_set = set(tracked_labels)
        label_mask = np.array(
            [lbl in tracked_set for lbl in capture.labels], dtype=bool
        )
        valid = capture.valid & label_mask[None, :]
    else:
        valid = capture.valid & np.isfinite(pred).all(axis=-1)

    if not np.isfinite(pred[valid]).all():
        raise ValueError("Predicted points must be finite at all valid capture points")

    dist_sq = np.sum((pred - obs) ** 2, axis=-1)
    term_start = int(capture.frames * (1.0 - terminal_ratio))
    term_rmse, head_term_rmse, body_term_rmse = _terminal_partition_rmse(
        dist_sq, valid, capture.labels, term_start
    )
    early_mask = (capture.time_s <= early_cutoff_s)[:, None] & valid
    return SharedMetrics(
        whole_marker_rmse_m=_rms_from_mask(dist_sq, valid),
        early_marker_rmse_m=_rms_from_mask(dist_sq, early_mask),
        terminal_marker_rmse_m=term_rmse,
        club_marker_rmse_m=_club_rmse(dist_sq, valid, capture.labels),
        pelvis_yaw_rmse_rad=_pelvis_yaw_rmse(obs, pred, valid, capture.labels),
        terminal_body_excluding_head_rmse_m=body_term_rmse,
        terminal_head_cluster_rmse_m=head_term_rmse,
    )
