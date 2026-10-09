"""Model-owned forward kinematics for validated native replay trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from src.shared.python._seam_redirect import extend_sidekick_lab_path

extend_sidekick_lab_path()

from src.shared.python.motion_matching.replay_metrics import NativeMarkerPositionOutput

from src.engines.feedback_native_execution import (
    NativeAdapterBinding,
    NativeExecutionReceipt,
    NativeReplayRequest,
    execute_native_replay_with_output,
)
from src.engines.feedback_comparison import FeedbackComparisonRegistry
from src.engines.model_inventory import TARGET_ENGINES


@dataclass(frozen=True)
class NativeMarkerAttachment:
    """An explicit local 3-D marker point attached to one native body/frame."""

    label: str
    frame_id: str
    local_position_m: tuple[float, float, float]


@dataclass(frozen=True)
class NativeMarkerMap:
    """Ordered marker bindings tied to exact source, loaded model and provider."""

    engine_id: str
    native_model_id: str
    native_variant_id: str
    native_execution_provider_id: str
    native_execution_provider_sha256: str
    source_model_sha256: str
    loaded_native_model_sha256: str
    output_frame_id: str
    timebase_id: str
    attachments: tuple[NativeMarkerAttachment, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "schema_version": "native-marker-map/1.0.0",
            **asdict(self),
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self, binding: NativeAdapterBinding, timebase_id: str) -> None:
        """Reject maps whose model, adapter, frame, clock or points are stale."""
        identity = (
            self.native_model_id,
            self.native_variant_id,
            self.native_execution_provider_id,
            self.native_execution_provider_sha256,
            self.source_model_sha256,
            self.loaded_native_model_sha256,
        )
        expected = (
            binding.native_model_id,
            binding.native_variant_id,
            binding.native_execution_provider_id,
            binding.native_execution_provider_sha256,
            binding.source_model_sha256,
            binding.loaded_native_model_sha256,
        )
        if identity != expected:
            raise ValueError("native marker map model or provider identity differs")
        if self.timebase_id != timebase_id:
            raise ValueError("native marker map timebase differs from replay bundle")
        if self.output_frame_id != "world":
            raise ValueError("native marker output frame must be explicit world frame")
        if not self.engine_id or not self.native_model_id or not self.native_variant_id:
            raise ValueError("native marker map identity fields must be non-empty")
        for name, digest in (
            ("adapter provider", self.native_execution_provider_sha256),
            ("source model", self.source_model_sha256),
            ("loaded model", self.loaded_native_model_sha256),
        ):
            if len(digest) != 64 or any(
                char not in "0123456789abcdef" for char in digest
            ):
                raise ValueError(f"native marker map {name} identity must be SHA-256")
        if not self.attachments:
            raise ValueError("native marker map requires at least one attachment")
        labels = tuple(item.label for item in self.attachments)
        if any(not label.strip() for label in labels) or len(labels) != len(
            set(labels)
        ):
            raise ValueError("native marker labels must be non-empty and unique")
        for attachment in self.attachments:
            if not attachment.frame_id.strip():
                raise ValueError("native marker frame identity must be explicit")
            point = np.asarray(attachment.local_position_m, dtype=np.float64)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError(
                    "native marker point must be a finite 3-vector in metres"
                )

    def adapter_attachments(
        self,
    ) -> tuple[tuple[str, str, tuple[float, float, float]], ...]:
        return tuple(
            (item.label, item.frame_id, item.local_position_m)
            for item in self.attachments
        )


@dataclass(frozen=True)
class NativeMarkerReplayEvidence:
    """Native marker output and its replay, model, provider and map identities."""

    schema_version: str
    qualification: str
    receipt_sha256: str
    native_execution_receipt: NativeExecutionReceipt
    marker_map_sha256: str
    marker_output_sha256: str
    native_output: NativeMarkerPositionOutput

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "qualification": self.qualification,
            "receipt_sha256": self.receipt_sha256,
            "engine": self.native_execution_receipt.engine,
            "source_model_sha256": self.native_execution_receipt.source_model_sha256,
            "native_execution_provider_sha256": (
                self.native_execution_receipt.native_execution_provider_sha256
            ),
            "loaded_native_model_sha256": (
                self.native_execution_receipt.loaded_native_model_sha256
            ),
            "state_output_sha256": self.native_execution_receipt.output_state_sha256,
            "native_execution_receipt": asdict(self.native_execution_receipt),
            "marker_map_sha256": self.marker_map_sha256,
            "marker_output_sha256": self.marker_output_sha256,
            "frame_id": self.native_output.frame_id,
            "timebase_id": self.native_output.timebase_id,
            "marker_labels": list(self.native_output.marker_labels),
            "sample_count": len(self.native_output.time_s),
        }


@dataclass(frozen=True)
class NativeMarkerRequest:
    """One F09c replay request paired with its explicit native marker map."""

    replay_request: NativeReplayRequest
    marker_map: NativeMarkerMap

    @property
    def key(self) -> tuple[str, str, Any]:
        binding = self.replay_request.binding
        return binding.package_id, binding.variant_id, binding.drive_mode


@dataclass(frozen=True)
class NativeMarkerRowResult:
    package_id: str
    variant_id: str
    drive_mode: str
    engine: str
    required: bool
    support: str
    availability: str
    qualification: str
    status: str
    evidence: NativeMarkerReplayEvidence | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "variant_id": self.variant_id,
            "drive_mode": self.drive_mode,
            "engine": self.engine,
            "required": self.required,
            "support": self.support,
            "availability": self.availability,
            "qualification": self.qualification,
            "status": self.status,
            "evidence": self.evidence.as_dict() if self.evidence else None,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class NativeMarkerReplayReport:
    rows: tuple[NativeMarkerRowResult, ...]
    required_engine_ids: tuple[str, ...]

    @property
    def missing_engine_ids(self) -> tuple[str, ...]:
        covered = {
            row.engine
            for row in self.rows
            if row.required and row.status == "marker_output_generated"
        }
        return tuple(
            engine for engine in self.required_engine_ids if engine not in covered
        )

    @property
    def has_full_required_fk_coverage(self) -> bool:
        """Whether every required cell has markers; never a physics qualification."""
        return (
            bool(self.rows)
            and not self.missing_engine_ids
            and all(
                not row.required or row.status == "marker_output_generated"
                for row in self.rows
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "native-marker-replay-report/1.0.0",
            "required_engine_ids": list(self.required_engine_ids),
            "missing_engine_ids": list(self.missing_engine_ids),
            "has_full_required_fk_coverage": self.has_full_required_fk_coverage,
            "rows": [row.as_dict() for row in self.rows],
        }


def _receipt_sha256(receipt: NativeExecutionReceipt) -> str:
    payload = json.dumps(
        asdict(receipt), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _marker_output_sha256(output: NativeMarkerPositionOutput) -> str:
    digest = hashlib.sha256()
    metadata = {
        "frame_id": output.frame_id,
        "timebase_id": output.timebase_id,
        "marker_labels": tuple(output.marker_labels),
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, values in (
        ("time_s", output.time_s),
        ("positions_m", output.positions_m),
    ):
        array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<u8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def execute_native_marker_replay(
    request: NativeReplayRequest,
    registry: FeedbackComparisonRegistry,
    marker_map: NativeMarkerMap,
) -> NativeMarkerReplayEvidence:
    """Execute one replay, then map each actual native configuration through FK."""
    bundle = request.bundle
    marker_map.validate(request.binding, bundle.input_history.timebase_id)
    inventory_row = registry.get(
        request.binding.package_id,
        request.binding.variant_id,
        request.binding.drive_mode,
    )
    if marker_map.engine_id != inventory_row.engine:
        raise ValueError("native marker map engine differs from inventory row")
    execution = execute_native_replay_with_output(request, registry)
    receipt = execution.receipt
    if marker_map.engine_id != receipt.engine:
        raise ValueError("native marker map engine differs from replay receipt")
    if receipt.qualification != "unqualified" or not receipt.full_state:
        raise ValueError("native marker FK requires a full-state replay receipt")
    attachments = marker_map.adapter_attachments()
    if receipt.engine == "mujoco":
        from src.engines.physics_engines.mujoco.python.native_torque_replay import (
            NativeTorqueReplay,
            native_marker_positions_from_replay as mujoco_marker_positions,
        )

        if not isinstance(execution.output, NativeTorqueReplay):
            raise ValueError("MuJoCo replay returned an unknown native output type")
        marker_output = mujoco_marker_positions(
            bundle, request.model_path, execution.output, attachments
        )
    elif receipt.engine == "drake":
        from src.engines.physics_engines.drake.python.native_torque_replay import (
            NativeDrakeTorqueReplay,
            native_marker_positions_from_replay as drake_marker_positions,
        )

        if not isinstance(execution.output, NativeDrakeTorqueReplay):
            raise ValueError("Drake replay returned an unknown native output type")
        marker_output = drake_marker_positions(
            bundle, request.model_path, execution.output, attachments
        )
    else:
        raise ValueError("no native marker FK provider is registered for this engine")
    times = np.asarray(marker_output.time_s, dtype=np.float64)
    positions = np.asarray(marker_output.positions_m, dtype=np.float64)
    expected_times = np.asarray(bundle.input_history.time_seconds, dtype=np.float64)
    expected_labels = tuple(item.label for item in marker_map.attachments)
    if (
        marker_output.frame_id != marker_map.output_frame_id
        or marker_output.timebase_id != marker_map.timebase_id
        or tuple(marker_output.marker_labels) != expected_labels
        or not np.array_equal(times, expected_times)
        or positions.shape != (len(times), len(expected_labels), 3)
        or not np.isfinite(positions).all()
        or not np.isfinite(times).all()
        or not math.isclose(times[-1], receipt.horizon_s, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("native marker FK output differs from frozen replay contract")
    return NativeMarkerReplayEvidence(
        "native-marker-replay/1.0.0",
        "unqualified",
        _receipt_sha256(receipt),
        receipt,
        marker_map.sha256,
        _marker_output_sha256(marker_output),
        marker_output,
    )


def build_native_marker_replay_report(
    registry: FeedbackComparisonRegistry,
    requests: tuple[NativeMarkerRequest, ...],
) -> NativeMarkerReplayReport:
    """Keep every inventory cell and required engine visible in FK coverage."""
    by_key = {request.key: request for request in requests}
    if len(by_key) != len(requests):
        raise ValueError("duplicate native marker replay request row")
    inventory_keys = {
        (row.package_id, row.variant_id, row.drive_mode) for row in registry.rows
    }
    if set(by_key) - inventory_keys:
        raise ValueError(
            "native marker request references an unregistered inventory row"
        )
    results: list[NativeMarkerRowResult] = []
    for row in registry.rows:
        key = (row.package_id, row.variant_id, row.drive_mode)
        request = by_key.get(key)
        status = "missing_marker_map"
        reason = "no explicit native marker request"
        evidence: NativeMarkerReplayEvidence | None = None
        if request is not None:
            if row.engine not in {"mujoco", "drake"}:
                status, reason = (
                    "unsupported_marker_fk",
                    "no reviewed native FK provider",
                )
            elif row.support != "supported":
                status, reason = (
                    "unsupported_inventory",
                    "inventory row is not supported",
                )
            elif row.availability != "available":
                status, reason = (
                    "unavailable",
                    "inventory availability is not available",
                )
            else:
                try:
                    evidence = execute_native_marker_replay(
                        request.replay_request, registry, request.marker_map
                    )
                except (ImportError, OSError) as error:
                    status, reason = "runtime_unavailable", type(error).__name__
                except (TypeError, ValueError, RuntimeError) as error:
                    status, reason = "rejected", type(error).__name__
                else:
                    status, reason = "marker_output_generated", ""
        results.append(
            NativeMarkerRowResult(
                row.package_id,
                row.variant_id,
                row.drive_mode.value,
                row.engine,
                row.required,
                row.support,
                row.availability,
                row.qualification,
                status,
                evidence,
                reason,
            )
        )
    return NativeMarkerReplayReport(tuple(results), tuple(sorted(TARGET_ENGINES)))
