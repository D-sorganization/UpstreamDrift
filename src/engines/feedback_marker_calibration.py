"""Versioned provisional marker-placement evidence from frozen calibration data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from src.engines.feedback_native_execution import NativeAdapterBinding
from src.engines.feedback_native_markers import (
    NativeMarkerAttachment,
    NativeMarkerMap,
    native_binding_identity,
    require_sha256,
)
from src.shared.python.motion_matching.marker_calibration import (
    Pose,
    score_frozen_marker_offsets,
    static_marker_offsets,
)
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

FloatArray = NDArray[np.float64]


class _HashAccumulator(Protocol):
    def update(self, data: bytes, /) -> None: ...


@dataclass(frozen=True)
class NativeMarkerCalibrationRequest:
    """Inputs and provenance needed to estimate and later revalidate offsets."""

    capture: TourCapture
    frame_ids_by_label: Mapping[str, str]
    poses: Sequence[Mapping[str, Pose]]
    pose_time_s: FloatArray
    binding: NativeAdapterBinding
    native_engine_id: str
    pose_provider_id: str
    pose_provider_sha256: str
    capture_frame_id: str
    capture_timebase_id: str


def _hash_array(digest: _HashAccumulator, name: str, values: NDArray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
    digest.update(array.tobytes())


def capture_observation_sha256(capture: TourCapture) -> str:
    """Hash observation bytes and ordered labels without serializing capture data."""
    digest = hashlib.sha256()
    digest.update(json.dumps(capture.labels, separators=(",", ":")).encode())
    _hash_array(digest, "points_m", np.asarray(capture.points_m, dtype="<f8"))
    _hash_array(digest, "valid", np.asarray(capture.valid, dtype=np.uint8))
    _hash_array(digest, "time_s", np.asarray(capture.time_s, dtype="<f8"))
    return digest.hexdigest()


def _snapshot_capture(capture: TourCapture) -> TourCapture:
    """Revalidate and detach capture arrays from writable caller-owned views."""
    return TourCapture(
        np.array(capture.time_s, dtype=np.float64, copy=True),
        capture.labels,
        np.array(capture.points_m, dtype=np.float64, copy=True),
        np.array(capture.valid, dtype=np.bool_, copy=True),
        capture.source_sha256,
    )


def _pose_history_sha256(
    frame_ids: Mapping[str, str], poses: Sequence[Mapping[str, Pose]]
) -> str:
    digest = hashlib.sha256()
    ordered = tuple(frame_ids.items())
    digest.update(json.dumps(ordered, separators=(",", ":")).encode())
    for pose_set in poses:
        for _, frame_id in ordered:
            if frame_id not in pose_set:
                raise ValueError(f"Pose history omits native frame {frame_id}")
            rotation, translation = pose_set[frame_id]
            rotation_array = np.asarray(rotation, dtype="<f8")
            translation_array = np.asarray(translation, dtype="<f8")
            if (
                rotation_array.shape != (3, 3)
                or translation_array.shape != (3,)
                or not np.isfinite(rotation_array).all()
                or not np.isfinite(translation_array).all()
                or not np.allclose(
                    rotation_array.T @ rotation_array,
                    np.eye(3),
                    rtol=0,
                    atol=1e-10,
                )
                or not np.isclose(
                    np.linalg.det(rotation_array), 1.0, rtol=0, atol=1e-10
                )
            ):
                raise ValueError(f"Pose for native frame {frame_id} is not rigid")
            _hash_array(digest, "rotation", rotation_array)
            _hash_array(digest, "translation", translation_array)
    return digest.hexdigest()


def _validate_binding_identity(
    artifact: NativeMarkerAttachmentCalibrationArtifact,
    binding: NativeAdapterBinding,
) -> None:
    identity = (
        artifact.native_model_id,
        artifact.native_variant_id,
        artifact.native_execution_provider_id,
        artifact.native_execution_provider_sha256,
        artifact.source_model_sha256,
        artifact.loaded_native_model_sha256,
    )
    if identity != native_binding_identity(binding):
        raise ValueError("marker calibration binding identity differs")


@dataclass(frozen=True)
class NativeMarkerAttachmentCalibrationArtifact:
    """Integrity receipt for estimated body-fixed points; never qualification."""

    schema_version: str
    calibration_status: str
    qualification: str
    holdout_status: str
    physiology_status: str
    native_engine_id: str
    native_model_id: str
    native_variant_id: str
    native_execution_provider_id: str
    native_execution_provider_sha256: str
    pose_provider_id: str
    pose_provider_sha256: str
    source_model_sha256: str
    loaded_native_model_sha256: str
    capture_source_sha256: str
    capture_observations_sha256: str
    capture_time_grid_sha256: str
    capture_labels: tuple[str, ...]
    capture_frame_id: str
    capture_timebase_id: str
    pose_trajectory_sha256: str
    calibration_method: str
    attachments: tuple[NativeMarkerAttachment, ...]
    valid_samples_by_marker: tuple[tuple[str, int], ...]
    calibration_fit_rms_m: float
    calibration_fit_rms_by_marker_m: tuple[tuple[str, float], ...]

    @property
    def sha256(self) -> str:
        """Return the digest of the canonical public metadata and offsets."""
        encoded = json.dumps(
            {"schema_version": self.schema_version, **asdict(self)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_structure(self) -> None:
        """Check version, provisional disposition, identities and ordered points."""
        if self.schema_version != "native-marker-calibration/1.0.0":
            raise ValueError("marker calibration artifact schema is unsupported")
        if (
            self.calibration_status != "provisional_estimated"
            or self.qualification != "unqualified"
            or self.holdout_status != "not_evaluated"
            or self.physiology_status != "unqualified"
        ):
            raise ValueError("marker calibration cannot promote qualification")
        identity_fields = (
            self.native_engine_id,
            self.native_model_id,
            self.native_variant_id,
            self.native_execution_provider_id,
            self.pose_provider_id,
            self.capture_frame_id,
            self.capture_timebase_id,
            self.calibration_method,
        )
        if any(
            not isinstance(value, str) or not value.strip() for value in identity_fields
        ):
            raise ValueError("marker calibration identity fields must be nonempty")
        for name, digest in (
            ("native execution provider", self.native_execution_provider_sha256),
            ("pose provider", self.pose_provider_sha256),
            ("source model", self.source_model_sha256),
            ("loaded model", self.loaded_native_model_sha256),
            ("capture source", self.capture_source_sha256),
            ("capture observations", self.capture_observations_sha256),
            ("capture clock", self.capture_time_grid_sha256),
            ("pose trajectory", self.pose_trajectory_sha256),
        ):
            require_sha256(name, digest)
        labels = tuple(item.label for item in self.attachments)
        if (
            not labels
            or labels != self.capture_labels
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                "calibrated marker order must exactly match capture labels"
            )
        if tuple(label for label, _ in self.valid_samples_by_marker) != labels:
            raise ValueError("calibration sample counts must match marker order")
        for label, count in self.valid_samples_by_marker:
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"marker {label} requires valid calibration samples")
        fit_values = (self.calibration_fit_rms_m,) + tuple(
            value for _, value in self.calibration_fit_rms_by_marker_m
        )
        if not np.isfinite(fit_values).all() or any(value < 0 for value in fit_values):
            raise ValueError(
                "calibration fit residuals must be finite nonnegative metres"
            )
        if tuple(label for label, _ in self.calibration_fit_rms_by_marker_m) != labels:
            raise ValueError("calibration fit residuals must follow marker order")
        for attachment in self.attachments:
            if not attachment.frame_id.strip():
                raise ValueError("native marker frame identity must be explicit")
            offset = np.asarray(attachment.local_position_m, dtype=np.float64)
            if offset.shape != (3,) or not np.isfinite(offset).all():
                raise ValueError("calibrated marker offsets must be finite metres")

    def validate(self, request: NativeMarkerCalibrationRequest) -> None:
        """Rehash inputs and compare expected provider and capture coordinates."""
        capture = _snapshot_capture(request.capture)
        poses = request.poses
        self.validate_structure()
        _validate_binding_identity(self, request.binding)
        require_sha256("pose provider", request.pose_provider_sha256)
        if (self.pose_provider_id, self.pose_provider_sha256) != (
            request.pose_provider_id,
            request.pose_provider_sha256,
        ):
            raise ValueError("marker calibration pose provider identity differs")
        if self.native_engine_id != request.native_engine_id:
            raise ValueError("marker calibration native engine identity differs")
        if (self.capture_frame_id, self.capture_timebase_id) != (
            request.capture_frame_id,
            request.capture_timebase_id,
        ):
            raise ValueError("marker calibration capture coordinates differ")
        if capture.source_sha256 != self.capture_source_sha256:
            raise ValueError("marker calibration capture source differs")
        if capture.labels != self.capture_labels:
            raise ValueError("marker calibration capture label order differs")
        frame_ids = tuple(request.frame_ids_by_label.items())
        expected_frame_ids = tuple(
            (attachment.label, attachment.frame_id) for attachment in self.attachments
        )
        if frame_ids != expected_frame_ids:
            raise ValueError("marker calibration native frame mapping differs")
        times = np.asarray(request.pose_time_s, dtype=np.float64)
        if times.shape != capture.time_s.shape or not np.array_equal(
            times, capture.time_s
        ):
            raise ValueError("marker calibration pose clock differs from capture")
        if capture_observation_sha256(capture) != self.capture_observations_sha256:
            raise ValueError("marker calibration capture observations differ")
        require_sha256("capture clock", _array_sha256(capture.time_s))
        if _array_sha256(capture.time_s) != self.capture_time_grid_sha256:
            raise ValueError("marker calibration capture clock differs")
        if len(poses) != capture.frames:
            raise ValueError("marker calibration pose count differs from capture")
        if (
            _pose_history_sha256(
                {item.label: item.frame_id for item in self.attachments}, poses
            )
            != self.pose_trajectory_sha256
        ):
            raise ValueError("marker calibration pose trajectory differs")
        expected_offsets = static_marker_offsets(
            capture, request.frame_ids_by_label, poses
        )
        if any(
            item.local_position_m != expected_offsets[item.label][1]
            for item in self.attachments
        ):
            raise ValueError("marker calibration offsets differ from frozen inputs")
        expected_rms, expected_per_marker = score_frozen_marker_offsets(
            capture, expected_offsets, poses
        )
        if self.calibration_fit_rms_m != expected_rms or any(
            residual != expected_per_marker[label]
            for label, residual in self.calibration_fit_rms_by_marker_m
        ):
            raise ValueError("marker calibration residuals differ from frozen inputs")

    def to_native_marker_map(
        self, binding: NativeAdapterBinding, *, output_timebase_id: str
    ) -> NativeMarkerMap:
        """Create an explicitly unqualified FK map linked to this artifact."""
        self.validate_structure()
        _validate_binding_identity(self, binding)
        return NativeMarkerMap(
            self.native_engine_id,
            self.native_model_id,
            self.native_variant_id,
            self.native_execution_provider_id,
            self.native_execution_provider_sha256,
            self.source_model_sha256,
            self.loaded_native_model_sha256,
            "world",
            output_timebase_id,
            self.attachments,
            self.sha256,
        )

    def as_dict(self) -> dict[str, object]:
        """Return metadata and offsets, omitting observations and local paths."""
        return {**asdict(self), "sha256": self.sha256}


def _array_sha256(values: NDArray) -> str:
    digest = hashlib.sha256()
    _hash_array(digest, "values", np.asarray(values, dtype="<f8"))
    return digest.hexdigest()


def calibrate_static_marker_attachments(
    request: NativeMarkerCalibrationRequest,
) -> NativeMarkerAttachmentCalibrationArtifact:
    """Estimate fixed offsets from native poses and emit provisional evidence.

    The supplied pose history must come from the named source-bound provider and
    use the exact capture clock. This function hashes inputs for integrity but
    does not authenticate the caller or claim independent heldout validation.
    """
    capture = _snapshot_capture(request.capture)
    frame_ids_by_label = request.frame_ids_by_label
    poses = request.poses
    if capture.source_sha256 is None:
        raise ValueError("marker calibration requires a capture source SHA-256")
    require_sha256("capture source", capture.source_sha256)
    require_sha256("pose provider", request.pose_provider_sha256)
    if tuple(frame_ids_by_label) != capture.labels:
        raise ValueError("native marker frame mapping must follow capture label order")
    if any(
        not isinstance(frame, str) or not frame.strip()
        for frame in frame_ids_by_label.values()
    ):
        raise ValueError("native marker frame IDs must be explicit")
    if len(poses) != capture.frames:
        raise ValueError("one native pose set per capture frame is required")
    times = np.asarray(request.pose_time_s, dtype=np.float64)
    if times.shape != capture.time_s.shape or not np.array_equal(times, capture.time_s):
        raise ValueError("marker calibration pose clock differs from capture")
    offsets = static_marker_offsets(capture, frame_ids_by_label, poses)
    fit_rms_m, fit_rms_by_marker_m = score_frozen_marker_offsets(
        capture, offsets, poses
    )
    attachments = tuple(
        NativeMarkerAttachment(label, frame_ids_by_label[label], offsets[label][1])
        for label in capture.labels
    )
    artifact = NativeMarkerAttachmentCalibrationArtifact(
        "native-marker-calibration/1.0.0",
        "provisional_estimated",
        "unqualified",
        "not_evaluated",
        "unqualified",
        request.native_engine_id,
        request.binding.native_model_id,
        request.binding.native_variant_id,
        request.binding.native_execution_provider_id,
        request.binding.native_execution_provider_sha256,
        request.pose_provider_id,
        request.pose_provider_sha256,
        request.binding.source_model_sha256,
        request.binding.loaded_native_model_sha256,
        capture.source_sha256,
        capture_observation_sha256(capture),
        _array_sha256(capture.time_s),
        capture.labels,
        request.capture_frame_id,
        request.capture_timebase_id,
        _pose_history_sha256(frame_ids_by_label, poses),
        "body_frame_mean_via_static_marker_offsets/1.0.0",
        attachments,
        tuple(
            (label, int(capture.valid[:, index].sum()))
            for index, label in enumerate(capture.labels)
        ),
        fit_rms_m,
        tuple((label, fit_rms_by_marker_m[label]) for label in capture.labels),
    )
    artifact.validate_structure()
    return artifact
