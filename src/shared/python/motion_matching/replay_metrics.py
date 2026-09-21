"""Shared replay NPZ reader/writer and 5 uninterrupted metrics evaluator.

Standardized contract across Pinocchio, MuJoCo, and Drake for Step 4 of the Visuals Handoff.

Evaluates:
1. whole_rms_m: root mean square marker error over all valid (t, m) samples.
2. early_rms_m: root mean square marker error over valid samples with time_s <= 0.60 s.
3. terminal_rms_m: root mean square marker error on the final frame across valid markers.
4. club_cluster_rms_m: root mean square error on the final frame for clubhead/shaft markers.
5. pelvis_yaw_error_pct: relative pelvis heading error percentage at terminal frame from WaistLeft -> WaistRight.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


@dataclass(frozen=True)
class ReplayFiveMetrics:
    """The 5 uninterrupted metrics required by the tour matching benchmark."""

    whole_rms_m: float
    early_rms_m: float
    terminal_rms_m: float
    club_cluster_rms_m: float
    pelvis_yaw_error_pct: float

    def as_dict(self) -> dict[str, float]:
        return {
            "whole_rms_m": float(self.whole_rms_m),
            "early_rms_m": float(self.early_rms_m),
            "terminal_rms_m": float(self.terminal_rms_m),
            "club_cluster_rms_m": float(self.club_cluster_rms_m),
            "pelvis_yaw_error_pct": float(self.pelvis_yaw_error_pct),
        }


def save_native_replay_npz(
    path: Path | str,
    *,
    time_s: Array | Sequence[float],
    native_state: Array,
    markers_m: Array,
    target_m: Array,
    valid: BoolArray | Array,
) -> None:
    """Save a standardized replay archive with strict DbC contract validation.

    Layout:
    - time_s: 1D array of shape (N,)
    - native_state: 2D array of shape (N, 2*nq)
    - markers_m: 3D array of shape (N, M, 3)
    - target_m: 3D array of shape (N, M, 3)
    - valid: 2D array of shape (N, M) as bool
    """
    t_arr = np.asarray(time_s, dtype=np.float64)
    state_arr = np.asarray(native_state, dtype=np.float64)
    markers_arr = np.asarray(markers_m, dtype=np.float64)
    target_arr = np.asarray(target_m, dtype=np.float64)
    valid_arr = np.asarray(valid, dtype=bool)

    if t_arr.ndim != 1:
        raise ValueError(f"time_s must be 1D, got ndim={t_arr.ndim}")
    n_frames = len(t_arr)
    if n_frames < 2:
        raise ValueError(f"Replay must contain at least 2 frames, got {n_frames}")
    if np.any(np.diff(t_arr) <= 0):
        raise ValueError("time_s must be strictly monotonically increasing")

    if state_arr.ndim != 2 or state_arr.shape[0] != n_frames:
        raise ValueError(
            f"shape mismatch: native_state shape {state_arr.shape} does not match (N={n_frames}, 2*nq)"
        )
    if state_arr.shape[1] % 2 != 0:
        raise ValueError(
            f"native_state dimension {state_arr.shape[1]} must be even (2*nq)"
        )

    if (
        markers_arr.ndim != 3
        or markers_arr.shape[0] != n_frames
        or markers_arr.shape[2] != 3
    ):
        raise ValueError(
            f"shape mismatch: markers_m shape {markers_arr.shape} does not match (N={n_frames}, M, 3)"
        )
    n_markers = markers_arr.shape[1]

    if target_arr.ndim != 3 or target_arr.shape != (n_frames, n_markers, 3):
        raise ValueError(
            f"shape mismatch: target_m shape {target_arr.shape} does not match (N={n_frames}, M={n_markers}, 3)"
        )

    if valid_arr.shape != (n_frames, n_markers):
        raise ValueError(
            f"shape mismatch: valid shape {valid_arr.shape} does not match (N={n_frames}, M={n_markers})"
        )

    if not (np.isfinite(t_arr).all() and np.isfinite(state_arr).all()):
        raise ValueError("time_s and native_state must be finite")

    # In target markers, invalid samples may be NaN, but valid samples must be finite
    if not np.isfinite(target_arr[valid_arr]).all():
        raise ValueError("target_m must be finite for all valid samples")
    if not np.isfinite(markers_arr).all():
        raise ValueError("predicted markers_m must be entirely finite")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        time_s=t_arr,
        native_state=state_arr,
        markers_m=markers_arr,
        target_m=target_arr,
        valid=valid_arr,
    )


def load_native_replay_npz(path: Path | str) -> dict[str, Any]:
    """Load and validate a standardized replay archive.

    Returns dict containing 'time_s', 'native_state', 'markers_m', 'target_m', 'valid'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Replay file not found: {p}")

    with np.load(p) as data:
        required_keys = ("time_s", "native_state", "markers_m", "target_m", "valid")
        for k in required_keys:
            if k not in data:
                raise KeyError(f"Missing required key '{k}' in {p}")

        t_arr = np.asarray(data["time_s"], dtype=np.float64)
        state_arr = np.asarray(data["native_state"], dtype=np.float64)
        markers_arr = np.asarray(data["markers_m"], dtype=np.float64)
        target_arr = np.asarray(data["target_m"], dtype=np.float64)
        valid_arr = np.asarray(data["valid"], dtype=bool)

    # Perform structural validation
    n_frames = len(t_arr)
    if state_arr.shape[0] != n_frames or markers_arr.shape[0] != n_frames:
        raise ValueError(f"Inconsistent frame counts in {p}")
    if valid_arr.shape != markers_arr.shape[:2]:
        raise ValueError(
            f"Valid mask shape {valid_arr.shape} does not match {markers_arr.shape[:2]}"
        )

    return {
        "time_s": t_arr,
        "native_state": state_arr,
        "markers_m": markers_arr,
        "target_m": target_arr,
        "valid": valid_arr,
    }


def compute_replay_five_metrics(
    *,
    time_s: Array | Sequence[float],
    pred_markers_m: Array,
    target_markers_m: Array,
    valid: BoolArray | Array,
    marker_labels: Sequence[str],
    early_cutoff_s: float = 0.60,
) -> ReplayFiveMetrics:
    """Compute the standard 5 metrics from predicted and target marker trajectories.

    Preconditions:
    - time_s: shape (N,)
    - pred_markers_m: shape (N, M, 3)
    - target_markers_m: shape (N, M, 3)
    - valid: shape (N, M)
    - marker_labels: length M
    """
    t_arr = np.asarray(time_s, dtype=np.float64)
    pred = np.asarray(pred_markers_m, dtype=np.float64)
    target = np.asarray(target_markers_m, dtype=np.float64)
    val = np.asarray(valid, dtype=bool)

    n_frames, n_markers, dims = pred.shape
    if dims != 3 or target.shape != (n_frames, n_markers, 3):
        raise ValueError("Marker arrays must have shape (N, M, 3)")
    if len(marker_labels) != n_markers:
        raise ValueError(
            f"marker_labels count ({len(marker_labels)}) does not match markers shape ({n_markers})"
        )
    if val.shape != (n_frames, n_markers):
        raise ValueError(
            f"valid mask shape {val.shape} must match ({n_frames}, {n_markers})"
        )

    diff = pred - target
    sq_err = np.sum(diff**2, axis=-1)  # (N, M)

    # 1. Whole RMS
    if not np.any(val):
        raise ValueError("No valid marker samples found in replay")
    whole_rms_m = float(np.sqrt(np.mean(sq_err[val])))

    # 2. Early RMS (t <= early_cutoff_s)
    early_mask = (t_arr <= early_cutoff_s)[:, None] & val
    if np.any(early_mask):
        early_rms_m = float(np.sqrt(np.mean(sq_err[early_mask])))
    else:
        early_rms_m = 0.0

    # 3. Terminal RMS (last frame)
    term_val = val[-1]
    if np.any(term_val):
        terminal_rms_m = float(np.sqrt(np.mean(sq_err[-1, term_val])))
    else:
        terminal_rms_m = 0.0

    # 4. Club cluster RMS (last frame, clubhead and shaft markers)
    club_indices = [
        i
        for i, lbl in enumerate(marker_labels)
        if any(tag in lbl.lower() for tag in ("marker_2", "marker_3", "club"))
    ]
    if club_indices:
        club_term_mask = term_val[club_indices]
        if np.any(club_term_mask):
            term_club_idx = np.array(club_indices)[club_term_mask]
            club_cluster_rms_m = float(np.sqrt(np.mean(sq_err[-1, term_club_idx])))
        else:
            club_cluster_rms_m = 0.0
    else:
        club_cluster_rms_m = 0.0

    # 5. Pelvis yaw error pct
    if "WaistLeft" in marker_labels and "WaistRight" in marker_labels:
        wl_i = marker_labels.index("WaistLeft")
        wr_i = marker_labels.index("WaistRight")
        vp = pred[-1, wr_i, :2] - pred[-1, wl_i, :2]
        vt = target[-1, wr_i, :2] - target[-1, wl_i, :2]
        yaw_t = float(np.degrees(np.arctan2(vt[1], vt[0])))
        yaw_p = float(np.degrees(np.arctan2(vp[1], vp[0])))
        diff_deg = float((yaw_p - yaw_t + 180.0) % 360.0 - 180.0)
        pelvis_yaw_error_pct = float(abs(diff_deg) / max(abs(yaw_t), 1.0) * 100.0)
    else:
        pelvis_yaw_error_pct = 0.0

    return ReplayFiveMetrics(
        whole_rms_m=whole_rms_m,
        early_rms_m=early_rms_m,
        terminal_rms_m=terminal_rms_m,
        club_cluster_rms_m=club_cluster_rms_m,
        pelvis_yaw_error_pct=pelvis_yaw_error_pct,
    )
