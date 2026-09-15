"""Standardized five shared evaluation metrics for tour motion matching.

Computes:
1. whole_marker_rmse_m: Full trial marker tracking RMS across all valid tracked markers
2. early_marker_rmse_m: Early prefix tracking RMS (t <= 0.60 s, address / backswing)
3. terminal_marker_rmse_m: Terminal window tracking RMS (final 10% of frames / follow-through)
4. club_marker_rmse_m: Marker tracking RMS specifically for the club cluster (Marker_2 and Marker_3)
5. pelvis_yaw_rmse_rad: Orientation RMSE in the horizontal transverse plane (pelvis heading)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    TourCapture,
)


@dataclass(frozen=True)
class SharedMetrics:
    """The five standardized metrics reported across all physical engines."""

    whole_marker_rmse_m: float
    early_marker_rmse_m: float
    terminal_marker_rmse_m: float
    club_marker_rmse_m: float
    pelvis_yaw_rmse_rad: float

    def as_dict(self) -> dict[str, float]:
        return {
            "whole_marker_rmse_m": self.whole_marker_rmse_m,
            "early_marker_rmse_m": self.early_marker_rmse_m,
            "terminal_marker_rmse_m": self.terminal_marker_rmse_m,
            "club_marker_rmse_m": self.club_marker_rmse_m,
            "pelvis_yaw_rmse_rad": self.pelvis_yaw_rmse_rad,
        }


def _pelvis_yaw(points: np.ndarray, wl_idx: int, wr_idx: int) -> np.ndarray:
    """Compute pelvis yaw angle (radians) in the transverse plane from WaistLeft and WaistRight."""
    delta = points[:, wl_idx, :] - points[:, wr_idx, :]
    return np.arctan2(delta[:, 0], delta[:, 2])


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
    cap_time = capture.time_s
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

    diff = pred - obs
    dist_sq = np.sum(diff**2, axis=-1)

    # 1. Whole marker RMS
    whole_errs = dist_sq[valid]
    whole_rmse = float(np.sqrt(np.mean(whole_errs))) if len(whole_errs) > 0 else 0.0

    # 2. Early marker RMS (t <= early_cutoff_s)
    early_mask = (cap_time <= early_cutoff_s)[:, None] & valid
    early_errs = dist_sq[early_mask]
    early_rmse = float(np.sqrt(np.mean(early_errs))) if len(early_errs) > 0 else 0.0

    # 3. Terminal marker RMS (final terminal_ratio portion)
    term_start = int(capture.frames * (1.0 - terminal_ratio))
    term_mask = np.zeros_like(valid)
    term_mask[term_start:, :] = valid[term_start:, :]
    term_errs = dist_sq[term_mask]
    term_rmse = float(np.sqrt(np.mean(term_errs))) if len(term_errs) > 0 else 0.0

    # 4. Club marker RMS
    club_labels = set(MARKER_SEGMENTS["club"])
    club_indices = [i for i, label in enumerate(capture.labels) if label in club_labels]
    if club_indices:
        club_mask = np.zeros_like(valid)
        club_mask[:, club_indices] = valid[:, club_indices]
        club_errs = dist_sq[club_mask]
        club_rmse = float(np.sqrt(np.mean(club_errs))) if len(club_errs) > 0 else 0.0
    else:
        club_rmse = 0.0

    # 5. Pelvis yaw RMSE
    if "WaistLeft" in capture.labels and "WaistRight" in capture.labels:
        wl_idx = capture.index("WaistLeft")
        wr_idx = capture.index("WaistRight")
        valid_both = valid[:, wl_idx] & valid[:, wr_idx]
        if np.any(valid_both):
            obs_yaw = _pelvis_yaw(obs[valid_both], wl_idx, wr_idx)
            pred_yaw = _pelvis_yaw(pred[valid_both], wl_idx, wr_idx)
            angle_diff = np.remainder(pred_yaw - obs_yaw + np.pi, 2 * np.pi) - np.pi
            pelvis_yaw_rmse = float(np.sqrt(np.mean(angle_diff**2)))
        else:
            pelvis_yaw_rmse = 0.0
    else:
        pelvis_yaw_rmse = 0.0

    return SharedMetrics(
        whole_marker_rmse_m=whole_rmse,
        early_marker_rmse_m=early_rmse,
        terminal_marker_rmse_m=term_rmse,
        club_marker_rmse_m=club_rmse,
        pelvis_yaw_rmse_rad=pelvis_yaw_rmse,
    )
