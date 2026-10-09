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
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.evidence_integrity import is_real_sha256

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]


class PositionInterpolation(str, Enum):
    """Declared interpolation for marker positions, never generalized state."""

    LINEAR_POSITION = "linear_position/v1"


@dataclass(frozen=True)
class NativeMarkerPositionOutput:
    """A native replay's ordered marker positions and output clock."""

    time_s: Array | Sequence[float]
    positions_m: Array
    marker_labels: Sequence[str]
    frame_id: str
    timebase_id: str


@dataclass(frozen=True)
class ObservedMarkerPositions:
    """Measured marker positions, exact observation clock, and validity mask."""

    time_s: Array | Sequence[float]
    positions_m: Array
    valid: BoolArray | Array
    marker_labels: Sequence[str]
    frame_id: str
    timebase_id: str


@dataclass(frozen=True)
class ReplayObservationAlignment:
    """Native marker positions sampled on the exact measured observation clock.

    Input-policy and native integration clocks remain separate from the
    observation clock. This record contains positions only; it does not
    interpolate generalized coordinates, velocities, quaternions, or engine
    state.
    """

    native_output_time_s: Array
    observation_time_s: Array
    predicted_positions_m: Array
    observation_positions_m: Array
    observation_valid: BoolArray
    marker_labels: tuple[str, ...]
    frame_id: str
    timebase_id: str
    interpolation: PositionInterpolation
    source_identity_sha256: str
    output_identity_sha256: str
    observation_identity_sha256: str
    native_output_time_grid_sha256: str
    observation_time_grid_sha256: str
    alignment_identity_sha256: str

    def compute_replay_five_metrics(
        self, *, early_cutoff_s: float = 0.60
    ) -> ReplayFiveMetrics:
        """Reuse canonical replay metrics without changing the observation clock."""
        return compute_replay_five_metrics(
            time_s=self.observation_time_s,
            pred_markers_m=self.predicted_positions_m,
            target_markers_m=self.observation_positions_m,
            valid=self.observation_valid,
            marker_labels=self.marker_labels,
            early_cutoff_s=early_cutoff_s,
        )


def _canonical_array_sha256(values: np.ndarray) -> str:
    """Hash a contiguous little-endian numeric array with shape and dtype."""
    array = np.asarray(values)
    canonical = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<")))
    digest = hashlib.sha256()
    metadata = json.dumps(
        {"dtype": canonical.dtype.str, "shape": canonical.shape},
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _update_digest_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _position_payload_sha256(
    *,
    prefix: str,
    identity_sha256: str,
    times_s: Array,
    positions_m: Array,
    marker_labels: tuple[str, ...],
    frame_id: str,
    timebase_id: str,
    valid: BoolArray | None = None,
) -> str:
    digest = hashlib.sha256()
    _update_digest_text(digest, prefix)
    _update_digest_text(digest, identity_sha256)
    _update_digest_text(digest, frame_id)
    _update_digest_text(digest, timebase_id)
    _update_digest_text(
        digest, json.dumps(marker_labels, separators=(",", ":"), ensure_ascii=True)
    )
    digest.update(bytes.fromhex(_canonical_array_sha256(times_s)))
    canonical_positions = np.array(positions_m, dtype=np.float64, copy=True)
    if valid is not None:
        canonical_positions[~valid] = 0.0
        digest.update(bytes.fromhex(_canonical_array_sha256(valid)))
    digest.update(bytes.fromhex(_canonical_array_sha256(canonical_positions)))
    return digest.hexdigest()


def _validated_times(values: Array | Sequence[float], field: str) -> Array:
    times = np.asarray(values, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2:
        raise ValueError(f"{field} must be a one-dimensional clock with two samples")
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
        raise ValueError(f"{field} must be finite and strictly increasing")
    return np.array(times, dtype=np.float64, copy=True)


def _readonly_copy(values: np.ndarray, *, dtype: Any | None = None) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class _ValidatedPositionAlignment:
    method: PositionInterpolation
    source_identity_sha256: str
    native_times: Array
    observation_times: Array
    native_positions: Array
    observation_positions: Array
    observation_valid: BoolArray
    marker_labels: tuple[str, ...]
    frame_id: str
    timebase_id: str


def _validate_position_alignment(
    native_output: NativeMarkerPositionOutput,
    observations: ObservedMarkerPositions,
    interpolation: PositionInterpolation,
    source_identity_sha256: str,
) -> _ValidatedPositionAlignment:
    try:
        method = PositionInterpolation(interpolation)
    except (TypeError, ValueError) as exc:
        raise ValueError("unsupported marker-position interpolation") from exc
    if method is not PositionInterpolation.LINEAR_POSITION:
        raise ValueError("unsupported marker-position interpolation")
    if (
        not is_real_sha256(source_identity_sha256)
        or source_identity_sha256 != str(source_identity_sha256).lower()
    ):
        raise ValueError("source_identity_sha256 must be a lowercase SHA-256 digest")

    for value, name in (
        (native_output.frame_id, "native_output.frame_id"),
        (observations.frame_id, "observations.frame_id"),
        (native_output.timebase_id, "native_output.timebase_id"),
        (observations.timebase_id, "observations.timebase_id"),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty")
    if native_output.frame_id != observations.frame_id:
        raise ValueError("native output and observations must share a frame")
    if native_output.timebase_id != observations.timebase_id:
        raise ValueError("native output and observations must share a timebase")

    native_times = _validated_times(native_output.time_s, "native_output.time_s")
    observation_times = _validated_times(observations.time_s, "observations.time_s")
    if (
        observation_times[0] < native_times[0]
        or observation_times[-1] > native_times[-1]
    ):
        raise ValueError(
            "observation clock requires extrapolation outside native output"
        )

    labels = tuple(native_output.marker_labels)
    observed_labels = tuple(observations.marker_labels)
    if (
        not labels
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(labels) != len(set(labels))
    ):
        raise ValueError("native_marker_labels must be non-empty unique identities")
    if labels != observed_labels:
        raise ValueError("native and observation marker identities or order differ")
    native_positions = np.asarray(native_output.positions_m, dtype=np.float64)
    observed_positions = np.asarray(observations.positions_m, dtype=np.float64)
    valid = np.asarray(observations.valid, dtype=bool)
    marker_count = len(labels)
    if native_positions.shape != (len(native_times), marker_count, 3):
        raise ValueError(
            "native output must contain three-dimensional marker positions"
        )
    if observed_positions.shape != (len(observation_times), marker_count, 3):
        raise ValueError("observations must contain three-dimensional marker positions")
    if valid.shape != (len(observation_times), marker_count):
        raise ValueError("observation_valid shape must match observation markers")
    if not np.isfinite(native_positions).all():
        raise ValueError("native output marker positions must be finite")
    if not np.isfinite(observed_positions[valid]).all():
        raise ValueError("valid observation marker positions must be finite")

    return _ValidatedPositionAlignment(
        method=method,
        source_identity_sha256=source_identity_sha256,
        native_times=native_times,
        observation_times=observation_times,
        native_positions=native_positions,
        observation_positions=observed_positions,
        observation_valid=valid,
        marker_labels=labels,
        frame_id=native_output.frame_id,
        timebase_id=native_output.timebase_id,
    )


def _sample_positions_on_clock(
    native_times: Array, native_positions: Array, observation_times: Array
) -> Array:
    right = np.searchsorted(native_times, observation_times, side="left")
    right = np.minimum(right, len(native_times) - 1)
    exact = native_times[right] == observation_times
    left = np.where(exact, right, np.maximum(right - 1, 0))
    interval = native_times[right] - native_times[left]
    fraction = np.zeros(len(observation_times), dtype=np.float64)
    interpolated = ~exact
    fraction[interpolated] = (
        observation_times[interpolated] - native_times[left[interpolated]]
    ) / interval[interpolated]
    aligned_positions = native_positions[left] + fraction[:, None, None] * (
        native_positions[right] - native_positions[left]
    )
    return aligned_positions


def align_native_positions_to_observations(
    native_output: NativeMarkerPositionOutput,
    observations: ObservedMarkerPositions,
    *,
    interpolation: PositionInterpolation,
    source_identity_sha256: str,
) -> ReplayObservationAlignment:
    """Sample native 3-D marker positions at observations without extrapolation.

    The exact observation clock is retained for metric evaluation. Output and
    observation arrays must already share an explicit marker ordering, frame,
    and timebase. Missing observations follow the existing ``valid`` mask
    contract; native output positions and valid observations must be finite.
    """
    inputs = _validate_position_alignment(
        native_output, observations, interpolation, source_identity_sha256
    )
    aligned_positions = _sample_positions_on_clock(
        inputs.native_times, inputs.native_positions, inputs.observation_times
    )

    native_clock_sha256 = _canonical_array_sha256(inputs.native_times)
    observation_clock_sha256 = _canonical_array_sha256(inputs.observation_times)
    output_identity = _position_payload_sha256(
        prefix="native-position-output/1.0.0",
        identity_sha256=inputs.source_identity_sha256,
        times_s=inputs.native_times,
        positions_m=inputs.native_positions,
        marker_labels=inputs.marker_labels,
        frame_id=inputs.frame_id,
        timebase_id=inputs.timebase_id,
    )
    observation_identity = _position_payload_sha256(
        prefix="observation-position-set/1.0.0",
        identity_sha256=inputs.source_identity_sha256,
        times_s=inputs.observation_times,
        positions_m=inputs.observation_positions,
        marker_labels=inputs.marker_labels,
        frame_id=inputs.frame_id,
        timebase_id=inputs.timebase_id,
        valid=inputs.observation_valid,
    )
    alignment_digest = hashlib.sha256()
    _update_digest_text(alignment_digest, "replay-observation-alignment/1.0.0")
    _update_digest_text(alignment_digest, inputs.method.value)
    _update_digest_text(alignment_digest, output_identity)
    _update_digest_text(alignment_digest, observation_identity)
    _update_digest_text(alignment_digest, native_clock_sha256)
    _update_digest_text(alignment_digest, observation_clock_sha256)
    alignment_digest.update(bytes.fromhex(_canonical_array_sha256(aligned_positions)))
    return ReplayObservationAlignment(
        native_output_time_s=_readonly_copy(inputs.native_times),
        observation_time_s=_readonly_copy(inputs.observation_times),
        predicted_positions_m=_readonly_copy(aligned_positions),
        observation_positions_m=_readonly_copy(inputs.observation_positions),
        observation_valid=_readonly_copy(inputs.observation_valid, dtype=np.bool_),
        marker_labels=inputs.marker_labels,
        frame_id=inputs.frame_id,
        timebase_id=inputs.timebase_id,
        interpolation=inputs.method,
        source_identity_sha256=inputs.source_identity_sha256,
        output_identity_sha256=output_identity,
        observation_identity_sha256=observation_identity,
        native_output_time_grid_sha256=native_clock_sha256,
        observation_time_grid_sha256=observation_clock_sha256,
        alignment_identity_sha256=alignment_digest.hexdigest(),
    )


# Canonical club-cluster label tags (Marker_2 / Marker_3 clubhead-shaft pairs,
# plus any explicit 'club' naming). Single source of truth for every replay
# consumer; do not re-implement label matching locally.
CLUB_MARKER_TAGS: tuple[str, ...] = ("marker_2", "marker_3", "club")


def is_club_marker_label(label: str) -> bool:
    """Return True when a marker label belongs to the canonical club cluster."""
    lowered = str(label).lower()
    return any(tag in lowered for tag in CLUB_MARKER_TAGS)


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
        early_rms_m = float("nan")

    # 3. Terminal RMS (last frame)
    term_val = val[-1]
    if np.any(term_val):
        terminal_rms_m = float(np.sqrt(np.mean(sq_err[-1, term_val])))
    else:
        terminal_rms_m = float("nan")

    # 4. Club cluster RMS (last frame, clubhead and shaft markers)
    club_indices = [
        i for i, lbl in enumerate(marker_labels) if is_club_marker_label(lbl)
    ]
    if club_indices:
        club_term_mask = term_val[club_indices]
        if np.any(club_term_mask):
            term_club_idx = np.array(club_indices)[club_term_mask]
            club_cluster_rms_m = float(np.sqrt(np.mean(sq_err[-1, term_club_idx])))
        else:
            club_cluster_rms_m = float("nan")
    else:
        club_cluster_rms_m = float("nan")

    # 5. Pelvis yaw error pct
    if "WaistLeft" in marker_labels and "WaistRight" in marker_labels:
        wl_i = marker_labels.index("WaistLeft")
        wr_i = marker_labels.index("WaistRight")
        if term_val[wl_i] and term_val[wr_i]:
            vp = pred[-1, wr_i, :2] - pred[-1, wl_i, :2]
            vt = target[-1, wr_i, :2] - target[-1, wl_i, :2]
            yaw_t = float(np.degrees(np.arctan2(vt[1], vt[0])))
            yaw_p = float(np.degrees(np.arctan2(vp[1], vp[0])))
            diff_deg = float((yaw_p - yaw_t + 180.0) % 360.0 - 180.0)
            pelvis_yaw_error_pct = float(abs(diff_deg) / max(abs(yaw_t), 1.0) * 100.0)
        else:
            pelvis_yaw_error_pct = float("nan")
    else:
        pelvis_yaw_error_pct = float("nan")

    return ReplayFiveMetrics(
        whole_rms_m=whole_rms_m,
        early_rms_m=early_rms_m,
        terminal_rms_m=terminal_rms_m,
        club_cluster_rms_m=club_cluster_rms_m,
        pelvis_yaw_error_pct=pelvis_yaw_error_pct,
    )
