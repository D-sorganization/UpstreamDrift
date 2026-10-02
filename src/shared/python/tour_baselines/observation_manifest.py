"""Frozen Observation Manifests and Independent Error Metrics (MMR-02-I #11105).

Provides:
1. Strictly validated ObservationManifest for per-club motion matching.
2. Disjoint, immutable calibration vs. holdout frame assignment to prevent offset overfitting.
3. Deterministic independent error metric functions (pooled Euclidean RMSE, frame-wise RMS).
4. Fail-closed validation for non-finite coordinates, zero-observation sets, and unit errors.
5. Frame-level validity frozen per marker channel (missing spans) and metric binding that
   verifies any evaluation mask against the frozen frame-by-marker validity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.tour_capture_contract import (
    MARKER_SEGMENTS,
    MARKER_SEGMENTS_IRON,
    TOUR_CAPTURE,
    TOUR_CAPTURE_IRON,
    load_tour_capture,
)
from src.shared.python.tour_baselines.calibration import (
    GeometryCalibrationResult,
    calibrate_fixed_geometry,
)
from src.shared.python.tour_baselines.events import (
    SWING_EVENTS_DRIVER,
    SWING_EVENTS_IRON,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]

_CANONICAL_CAPTURES: dict[str, Path] = {
    "driver": _REPO_ROOT / "data" / "C3D_TA_Driver.c3d",
    "iron": _REPO_ROOT / "data" / "C3D_TA_Iron.c3d",
}


@dataclass(frozen=True)
class MarkerObservationSpec:
    """Observation metadata and tracking status for a single marker channel."""

    label: str
    segment: str
    is_tracked: bool
    valid_count: int
    missing_count: int
    is_interpolated: bool = False
    exclusion_reason: str | None = None
    missing_spans: tuple[tuple[int, int], ...] = ()
    interpolated_spans: tuple[tuple[int, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "segment": self.segment,
            "is_tracked": self.is_tracked,
            "valid_count": self.valid_count,
            "missing_count": self.missing_count,
            "is_interpolated": self.is_interpolated,
            "exclusion_reason": self.exclusion_reason,
            "missing_spans": [list(span) for span in self.missing_spans],
            "interpolated_spans": [list(span) for span in self.interpolated_spans],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MarkerObservationSpec:
        return cls(
            label=str(data["label"]),
            segment=str(data["segment"]),
            is_tracked=bool(data["is_tracked"]),
            valid_count=int(data["valid_count"]),
            missing_count=int(data["missing_count"]),
            is_interpolated=bool(data.get("is_interpolated", False)),
            exclusion_reason=(
                str(data["exclusion_reason"])
                if data.get("exclusion_reason") is not None
                else None
            ),
            missing_spans=tuple(
                (int(span[0]), int(span[1])) for span in data.get("missing_spans", ())
            ),
            interpolated_spans=tuple(
                (int(span[0]), int(span[1]))
                for span in data.get("interpolated_spans", ())
            ),
        )


@dataclass(frozen=True)
class ObservationManifest:
    """Frozen per-club observation schema with disjoint calibration and holdout splits."""

    capture_kind: str  # "driver" or "iron"
    source_file: str
    source_sha256: str
    frame_count: int
    rate_hz: float
    units: str
    handedness: str
    calibration_frames: tuple[int, ...]
    holdout_frames: tuple[int, ...]
    markers: Mapping[str, MarkerObservationSpec]
    coordinate_axes: str = "source-unregistered"

    def __post_init__(self) -> None:
        if self.units != "m":
            raise ValueError(f"units must be 'm', got {self.units!r}")
        if self.frame_count <= 0:
            raise ValueError(f"frame_count must be positive, got {self.frame_count}")
        if self.rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {self.rate_hz}")

        calib_set = set(self.calibration_frames)
        hold_set = set(self.holdout_frames)
        if not calib_set.isdisjoint(hold_set):
            intersection = sorted(calib_set.intersection(hold_set))
            raise ValueError(
                f"calibration_frames and holdout_frames must be disjoint; overlap: {intersection}"
            )

        for f in calib_set:
            if not (0 <= f < self.frame_count):
                raise ValueError(
                    f"calibration frame index {f} out of bounds [0, {self.frame_count})"
                )
        for f in hold_set:
            if not (0 <= f < self.frame_count):
                raise ValueError(
                    f"holdout frame index {f} out of bounds [0, {self.frame_count})"
                )

        # Freeze the caller-owned marker map so consumers cannot mutate the manifest
        # after construction and silently change serialized benchmark evidence.
        object.__setattr__(self, "markers", MappingProxyType(dict(self.markers)))

        # The per-marker aggregates must agree with the frozen frame-level validity.
        for label, spec in self.markers.items():
            span_frames = sum(end - start + 1 for start, end in spec.missing_spans)
            if span_frames != spec.missing_count:
                raise ValueError(
                    f"marker {label!r} missing_spans cover {span_frames} frames "
                    f"but missing_count is {spec.missing_count}"
                )
            if spec.valid_count + spec.missing_count != self.frame_count:
                raise ValueError(
                    f"marker {label!r} valid_count ({spec.valid_count}) + missing_count "
                    f"({spec.missing_count}) != frame_count ({self.frame_count})"
                )

    def frame_validity(self) -> np.ndarray:
        """Return the frozen frame-by-marker validity mask (frames x markers).

        Columns follow sorted marker label order - the canonical ordering used by
        metric binding. Reconstructed from the per-marker missing spans; read-only.
        """
        labels = tuple(sorted(self.markers))
        mask = np.ones((self.frame_count, len(labels)), dtype=bool)
        for col, label in enumerate(labels):
            for start, end in self.markers[label].missing_spans:
                mask[start : end + 1, col] = False
        mask.setflags(write=False)
        return mask

    def measured_frame_validity(self) -> np.ndarray:
        """Return the frozen frame-by-marker validity mask strictly for measured observations.

        Excludes any marker channels or frame spans marked as interpolated or unmeasured.
        """
        mask = self.frame_validity().copy()
        labels = tuple(sorted(self.markers))
        for col, label in enumerate(labels):
            spec = self.markers[label]
            if spec.is_interpolated:
                mask[:, col] = False
            for start, end in spec.interpolated_spans:
                mask[start : end + 1, col] = False
        mask.setflags(write=False)
        return mask

    def require_frozen_validity(
        self,
        valid: np.ndarray,
        *,
        require_measured_only: bool = False,
    ) -> np.ndarray:
        """Verify a caller-supplied mask against the frozen frame-level validity.

        Returns the frozen mask on success; raises ValueError when the supplied mask
        differs in shape or content, so gap-filled or residual-invalid samples can
        never become measured evidence under an authoritative manifest.
        """
        mask = np.asarray(valid)
        frozen = (
            self.measured_frame_validity()
            if require_measured_only
            else self.frame_validity()
        )
        if mask.shape != frozen.shape or not np.array_equal(mask, frozen):
            n_diff = (
                int(np.count_nonzero(mask != frozen))
                if mask.shape == frozen.shape
                else -1
            )
            mode_name = "measured " if require_measured_only else ""
            raise ValueError(
                f"Validity mask does not match the manifest's frozen {mode_name}frame-level "
                f"validity ({n_diff} differing cells); metric evaluation must use "
                "or verify the frozen frame-by-marker validity"
            )
        return frozen

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_kind": self.capture_kind,
            "source_file": self.source_file,
            "source_sha256": self.source_sha256,
            "frame_count": self.frame_count,
            "rate_hz": self.rate_hz,
            "units": self.units,
            "handedness": self.handedness,
            "coordinate_axes": self.coordinate_axes,
            "calibration_frames": list(self.calibration_frames),
            "holdout_frames": list(self.holdout_frames),
            "markers": {k: v.to_dict() for k, v in sorted(self.markers.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ObservationManifest:
        markers = {
            k: MarkerObservationSpec.from_dict(v)
            for k, v in data.get("markers", {}).items()
        }
        return cls(
            capture_kind=str(data["capture_kind"]),
            source_file=str(data["source_file"]),
            source_sha256=str(data["source_sha256"]),
            frame_count=int(data["frame_count"]),
            rate_hz=float(data["rate_hz"]),
            units=str(data["units"]),
            handedness=str(data["handedness"]),
            coordinate_axes=str(data.get("coordinate_axes", "source-unregistered")),
            calibration_frames=tuple(
                int(x) for x in data.get("calibration_frames", ())
            ),
            holdout_frames=tuple(int(x) for x in data.get("holdout_frames", ())),
            markers=markers,
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> ObservationManifest:
        return cls.from_dict(json.loads(json_str))


def _validate_arrays(
    pred: np.ndarray,
    obs: np.ndarray,
    valid: np.ndarray,
    manifest: ObservationManifest | None = None,
    *,
    measured_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    obs_arr = np.asarray(obs, dtype=np.float64)
    valid_arr = np.asarray(valid, dtype=bool)

    if pred_arr.ndim != 3 or pred_arr.shape[-1] != 3:
        raise ValueError(
            f"Expected trajectory tensor of rank 3 with XYZ components, got pred {pred_arr.shape}"
        )
    if obs_arr.ndim != 3 or obs_arr.shape[-1] != 3:
        raise ValueError(
            f"Expected trajectory tensor of rank 3 with XYZ components, got obs {obs_arr.shape}"
        )
    if pred_arr.shape != obs_arr.shape:
        raise ValueError(
            f"Shape mismatch: prediction {pred_arr.shape} vs observation {obs_arr.shape}"
        )
    n_frames, n_markers, _ = pred_arr.shape
    if valid_arr.shape != (n_frames, n_markers):
        raise ValueError(
            f"Valid mask shape {valid_arr.shape} != expected {(n_frames, n_markers)}"
        )

    # Bind evaluation to the frozen frame-level validity when a manifest is supplied.
    if manifest is not None:
        valid_arr = manifest.require_frozen_validity(
            valid_arr, require_measured_only=measured_only
        )

    n_valid = int(np.count_nonzero(valid_arr))
    if n_valid == 0:
        raise ValueError("Cannot compute RMSE with zero valid observations")

    valid_pred = pred_arr[valid_arr]
    valid_obs = obs_arr[valid_arr]
    if not (np.all(np.isfinite(valid_pred)) and np.all(np.isfinite(valid_obs))):
        raise ValueError("Non-finite coordinates encountered in valid observations")

    return pred_arr, obs_arr, valid_arr, n_valid


@precondition(
    lambda pred, obs, valid, **kwargs: (
        pred is not None and obs is not None and valid is not None
    )
)
@postcondition(lambda res: isinstance(res, float) and res >= 0.0)
def compute_pooled_rmse(
    pred: np.ndarray,
    obs: np.ndarray,
    valid: np.ndarray,
    *,
    manifest: ObservationManifest | None = None,
    measured_only: bool = False,
) -> float:
    """Compute exact pooled 3D Euclidean marker RMSE: sqrt(sum(valid ||pred - obs||^2) / N_valid).

    When a manifest is bound, `valid` is verified bit-for-bit against the manifest's
    frozen frame-by-marker validity before any metric is produced. When `measured_only=True`,
    the mask is verified strictly against measured (non-interpolated) observations.

    Note: Distinct from median or mean of per-frame RMS when valid counts or error distributions vary.
    """
    pred_arr, obs_arr, valid_arr, n_valid = _validate_arrays(
        pred, obs, valid, manifest=manifest, measured_only=measured_only
    )
    diff = pred_arr - obs_arr
    sq_dist = np.sum(diff**2, axis=-1)
    sum_valid_sq = float(np.sum(sq_dist[valid_arr]))
    return math.sqrt(sum_valid_sq / n_valid)


@precondition(
    lambda pred, obs, valid, **kwargs: (
        pred is not None and obs is not None and valid is not None
    )
)
def compute_frame_wise_rms(
    pred: np.ndarray,
    obs: np.ndarray,
    valid: np.ndarray,
    *,
    manifest: ObservationManifest | None = None,
    measured_only: bool = False,
) -> tuple[np.ndarray, float]:
    """Compute per-frame 3D marker RMS and overall median frame RMS.

    When a manifest is bound, `valid` is verified bit-for-bit against the manifest's
    frozen frame-by-marker validity before any metric is produced.

    Returns:
        (frame_rms, median_frame_rms)
    """
    pred_arr, obs_arr, valid_arr, _ = _validate_arrays(
        pred, obs, valid, manifest=manifest, measured_only=measured_only
    )
    n_frames = pred_arr.shape[0]
    frame_rms = np.full(n_frames, np.nan, dtype=np.float64)

    diff = pred_arr - obs_arr
    sq_dist = np.sum(diff**2, axis=-1)

    valid_frame_rms: list[float] = []
    for f in range(n_frames):
        m_valid = valid_arr[f]
        cnt = int(np.count_nonzero(m_valid))
        if cnt > 0:
            rms_f = float(np.sqrt(np.mean(sq_dist[f, m_valid])))
            frame_rms[f] = rms_f
            valid_frame_rms.append(rms_f)

    if not valid_frame_rms:
        raise ValueError("Cannot compute frame-wise RMS with zero valid observations")

    median_rms = float(np.median(valid_frame_rms))
    return frame_rms, median_rms


def _compute_phase_rmse(
    sq_dist: np.ndarray,
    valid_arr: np.ndarray,
    phase_spans: Mapping[str, tuple[int, int]],
) -> dict[str, float]:
    per_phase: dict[str, float] = {}
    for phase, (start, end) in phase_spans.items():
        sub_valid = valid_arr[start : end + 1]
        cnt = int(np.count_nonzero(sub_valid))
        if cnt > 0:
            sub_sq = sq_dist[start : end + 1][sub_valid]
            per_phase[phase] = math.sqrt(float(np.sum(sub_sq)) / cnt)
        else:
            per_phase[phase] = 0.0
    return per_phase


def _compute_segment_rmse(
    sq_dist: np.ndarray,
    valid_arr: np.ndarray,
    n_markers: int,
    manifest: ObservationManifest | None,
    segment_mapping: Mapping[str, str] | None,
) -> dict[str, float]:
    labels = tuple(sorted(manifest.markers)) if manifest is not None else ()
    seg_map: dict[str, str] = {}
    if segment_mapping is not None:
        seg_map.update(segment_mapping)
    elif manifest is not None:
        seg_map = {lbl: spec.segment for lbl, spec in manifest.markers.items()}

    segments: dict[str, list[int]] = {}
    for col in range(n_markers):
        lbl = labels[col] if col < len(labels) else f"marker_{col}"
        seg = seg_map.get(lbl, seg_map.get(f"marker_{col}", "unknown"))
        segments.setdefault(seg, []).append(col)

    per_segment: dict[str, float] = {}
    for seg, cols in sorted(segments.items()):
        sub_valid = valid_arr[:, cols]
        cnt = int(np.count_nonzero(sub_valid))
        if cnt > 0:
            sub_sq = sq_dist[:, cols][sub_valid]
            per_segment[seg] = math.sqrt(float(np.sum(sub_sq)) / cnt)
        else:
            per_segment[seg] = 0.0
    return per_segment


@dataclass(frozen=True)
class ComprehensiveErrorMetrics:
    """Comprehensive 3D tracking error metrics across frames, phases, and segments."""

    pooled_rmse: float
    median_frame_rms: float
    mean_frame_rms: float
    p95_error: float
    max_error: float
    per_phase_rmse: Mapping[str, float]
    per_segment_rmse: Mapping[str, float]
    total_valid_observations: int
    valid_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pooled_rmse": self.pooled_rmse,
            "median_frame_rms": self.median_frame_rms,
            "mean_frame_rms": self.mean_frame_rms,
            "p95_error": self.p95_error,
            "max_error": self.max_error,
            "per_phase_rmse": dict(self.per_phase_rmse),
            "per_segment_rmse": dict(self.per_segment_rmse),
            "total_valid_observations": self.total_valid_observations,
            "valid_frames": self.valid_frames,
        }


@precondition(
    lambda pred, obs, valid, **kwargs: (
        pred is not None and obs is not None and valid is not None
    )
)
def compute_comprehensive_error_metrics(
    pred: np.ndarray,
    obs: np.ndarray,
    valid: np.ndarray,
    *,
    manifest: ObservationManifest | None = None,
    measured_only: bool = False,
    phase_spans: Mapping[str, tuple[int, int]] | None = None,
    segment_mapping: Mapping[str, str] | None = None,
) -> ComprehensiveErrorMetrics:
    """Compute detailed metric breakdown: pooled RMSE, frame distribution, p95/max, per-phase and per-segment."""
    pred_arr, obs_arr, valid_arr, n_valid = _validate_arrays(
        pred, obs, valid, manifest=manifest, measured_only=measured_only
    )
    diff = pred_arr - obs_arr
    sq_dist = np.sum(diff**2, axis=-1)
    distances = np.sqrt(sq_dist[valid_arr])

    pooled_rmse = math.sqrt(float(np.sum(sq_dist[valid_arr])) / n_valid)
    p95_error = float(np.percentile(distances, 95))
    max_error = float(np.max(distances))

    frame_rms, median_rms = compute_frame_wise_rms(
        pred_arr, obs_arr, valid_arr, manifest=manifest, measured_only=measured_only
    )
    valid_frame_indices = np.where(np.isfinite(frame_rms))[0]
    valid_frames = len(valid_frame_indices)
    mean_frame_rms = (
        float(np.mean(frame_rms[valid_frame_indices])) if valid_frames > 0 else 0.0
    )

    per_phase = (
        _compute_phase_rmse(sq_dist, valid_arr, phase_spans)
        if phase_spans is not None
        else {}
    )
    per_segment = _compute_segment_rmse(
        sq_dist, valid_arr, pred_arr.shape[1], manifest, segment_mapping
    )

    return ComprehensiveErrorMetrics(
        pooled_rmse=pooled_rmse,
        median_frame_rms=median_rms,
        mean_frame_rms=mean_frame_rms,
        p95_error=p95_error,
        max_error=max_error,
        per_phase_rmse=MappingProxyType(per_phase),
        per_segment_rmse=MappingProxyType(per_segment),
        total_valid_observations=n_valid,
        valid_frames=valid_frames,
    )


@dataclass(frozen=True)
class CommonTargetComparison:
    """Comparison between two candidate models on their common valid observation subset."""

    common_pooled_rmse_a: float
    common_pooled_rmse_b: float
    common_p95_a: float
    common_p95_b: float
    valid_count_a: int
    valid_count_b: int
    common_valid_count: int
    exclusive_valid_count_a: int
    exclusive_valid_count_b: int
    coverage_ratio_a: float
    coverage_ratio_b: float
    common_coverage_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "common_pooled_rmse_a": self.common_pooled_rmse_a,
            "common_pooled_rmse_b": self.common_pooled_rmse_b,
            "common_p95_a": self.common_p95_a,
            "common_p95_b": self.common_p95_b,
            "valid_count_a": self.valid_count_a,
            "valid_count_b": self.valid_count_b,
            "common_valid_count": self.common_valid_count,
            "exclusive_valid_count_a": self.exclusive_valid_count_a,
            "exclusive_valid_count_b": self.exclusive_valid_count_b,
            "coverage_ratio_a": self.coverage_ratio_a,
            "coverage_ratio_b": self.coverage_ratio_b,
            "common_coverage_ratio": self.common_coverage_ratio,
        }


def compute_common_target_comparison(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    obs: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    *,
    manifest: ObservationManifest | None = None,
) -> CommonTargetComparison:
    """Evaluate two candidate models on their common valid observation target while keeping coverage differences explicit."""
    arr_a = np.asarray(pred_a, dtype=np.float64)
    arr_b = np.asarray(pred_b, dtype=np.float64)
    obs_arr = np.asarray(obs, dtype=np.float64)
    mask_a = np.asarray(valid_a, dtype=bool)
    mask_b = np.asarray(valid_b, dtype=bool)

    if arr_a.shape != obs_arr.shape or arr_b.shape != obs_arr.shape:
        raise ValueError(
            f"Shape mismatch: pred_a {arr_a.shape}, pred_b {arr_b.shape}, obs {obs_arr.shape}"
        )
    if mask_a.shape != obs_arr.shape[:2] or mask_b.shape != obs_arr.shape[:2]:
        raise ValueError(
            f"Mask shape mismatch: mask_a {mask_a.shape}, mask_b {mask_b.shape}, expected {obs_arr.shape[:2]}"
        )

    common_valid = mask_a & mask_b
    n_common = int(np.count_nonzero(common_valid))
    if n_common == 0:
        raise ValueError(
            "No common valid observations between candidate A and candidate B"
        )

    total_elements = mask_a.size
    n_a = int(np.count_nonzero(mask_a))
    n_b = int(np.count_nonzero(mask_b))

    rmse_a = compute_pooled_rmse(arr_a, obs_arr, common_valid)
    rmse_b = compute_pooled_rmse(arr_b, obs_arr, common_valid)

    diff_a = arr_a - obs_arr
    diff_b = arr_b - obs_arr
    dist_a = np.sqrt(np.sum(diff_a**2, axis=-1)[common_valid])
    dist_b = np.sqrt(np.sum(diff_b**2, axis=-1)[common_valid])
    p95_a = float(np.percentile(dist_a, 95))
    p95_b = float(np.percentile(dist_b, 95))

    return CommonTargetComparison(
        common_pooled_rmse_a=rmse_a,
        common_pooled_rmse_b=rmse_b,
        common_p95_a=p95_a,
        common_p95_b=p95_b,
        valid_count_a=n_a,
        valid_count_b=n_b,
        common_valid_count=n_common,
        exclusive_valid_count_a=n_a - n_common,
        exclusive_valid_count_b=n_b - n_common,
        coverage_ratio_a=float(n_a) / total_elements,
        coverage_ratio_b=float(n_b) / total_elements,
        common_coverage_ratio=float(n_common) / total_elements,
    )


def calibrate_with_manifest_protection(
    manifest: ObservationManifest,
    shoulder_pts: np.ndarray,
    grip_pts: np.ndarray,
    clubhead_pts: np.ndarray,
) -> GeometryCalibrationResult:
    """Calibrate fixed geometry with holdout protection enforced by the manifest.

    Delegates to the real calibration entry point (`calibrate_fixed_geometry`) with the
    manifest bound, so only `manifest.calibration_frames` rows can reach the estimator
    and holdout observations can never update calibrated parameters.
    """
    return calibrate_fixed_geometry(
        shoulder_pts,
        grip_pts,
        clubhead_pts,
        manifest=manifest,
    )


def _load_canonical_capture(kind_lower: str, capture_path: Path | str | None):
    """Load the canonical C3D capture and verify it against its frozen spec."""
    path = (
        Path(capture_path)
        if capture_path is not None
        else _CANONICAL_CAPTURES[kind_lower]
    )
    if not path.is_file():
        raise ValueError(
            f"Canonical capture not found: {path}; the frozen observation manifest "
            "requires the canonical capture to derive frame-level validity"
        )
    return load_tour_capture(path)


def build_frozen_observation_manifest(
    kind: str,
    capture_path: Path | str | None = None,
) -> ObservationManifest:
    """Build the authoritative frozen observation manifest for 'driver' or 'iron'.

    The holdout begins at the captured top-of-backswing event from the frozen event
    authority (driver frame 397, iron frame 394), and each marker's frame-level
    validity spans are derived from the canonical capture (SHA-verified) instead of
    private aggregate tables.
    """
    kind_lower = kind.strip().lower()
    if kind_lower == "driver":
        spec = TOUR_CAPTURE
        segments = MARKER_SEGMENTS
        events = SWING_EVENTS_DRIVER
        source_file = "C3D_TA_Driver.c3d"
    elif kind_lower == "iron":
        spec = TOUR_CAPTURE_IRON
        segments = MARKER_SEGMENTS_IRON
        events = SWING_EVENTS_IRON
        source_file = "C3D_TA_Iron.c3d"
    else:
        raise ValueError(f"Unknown capture kind: {kind!r}; expected 'driver' or 'iron'")

    capture = _load_canonical_capture(kind_lower, capture_path)

    # Event-based boundary: address through the backswing calibration phase, then the
    # top-of-backswing event starts the downswing/impact/follow-through holdout.
    top_of_backswing = events.top_of_backswing.frame_index
    calib_frames = tuple(range(top_of_backswing))
    holdout_frames = tuple(range(top_of_backswing, spec.frames))

    unassigned_markers = set(segments.get("unassigned", ()))
    markers: dict[str, MarkerObservationSpec] = {}

    for lbl in sorted(spec.labels):
        col = capture.index(lbl)
        valid_cnt = int(np.count_nonzero(capture.valid[:, col]))
        missing_cnt = int(spec.frames - valid_cnt)
        is_unassigned = lbl in unassigned_markers
        seg_found = "unassigned" if is_unassigned else "body"
        for s_name, s_labels in segments.items():
            if lbl in s_labels:
                seg_found = s_name
                break

        reason = (
            "Marker unassigned from anatomical model / sentinel tracking channel"
            if is_unassigned
            else None
        )
        markers[lbl] = MarkerObservationSpec(
            label=lbl,
            segment=seg_found,
            is_tracked=not is_unassigned,
            valid_count=valid_cnt,
            missing_count=missing_cnt,
            is_interpolated=False,
            exclusion_reason=reason,
            missing_spans=capture.missing_spans(lbl),
        )

    return ObservationManifest(
        capture_kind=kind_lower,
        source_file=source_file,
        source_sha256=spec.sha256,
        frame_count=spec.frames,
        rate_hz=spec.rate_hz,
        units="m",
        handedness=spec.handedness,
        coordinate_axes="source-unregistered",
        calibration_frames=calib_frames,
        holdout_frames=holdout_frames,
        markers=markers,
    )
