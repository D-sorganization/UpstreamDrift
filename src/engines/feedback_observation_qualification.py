"""F09 observation scoring from admitted replay evidence and native outputs.

The module transports no physics claim by itself. It binds observed marker
scores to F01 replay identity and leaves every registered model row in the
qualification denominator until its existing native gates are satisfied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sidekick.lab.mocap import ExperimentReplayBundle
    from src.engines.feedback_native_markers import NativeMarkerReplayEvidence

from src.engines.feedback_comparison import (
    ComparisonEvidence,
    ComparisonLevel,
    ComparisonRow,
    FeedbackComparisonRegistry,
)
from src.engines.model_inventory import TARGET_ENGINES
from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    AcceptanceVerdict,
    Horizon,
    evaluate,
)
from src.shared.python.motion_matching.pelvis_yaw import compute_pelvis_yaw_metrics
from src.shared.python.motion_matching.replay_metrics import (
    NativeMarkerPositionOutput,
    ObservedMarkerPositions,
    PositionInterpolation,
    ReplayObservationAlignment,
    ReplayFiveMetrics,
    align_native_positions_to_observations,
)

__all__ = [
    "FEEDBACK_OBSERVATION_SCHEMA_VERSION",
    "NativeObservationCase",
    "ObservationScoreReceipt",
    "FeedbackObservationRow",
    "FeedbackObservationReport",
    "comparison_key",
    "feedback_replay_identity_sha256",
    "build_feedback_observation_report",
]

FEEDBACK_OBSERVATION_SCHEMA_VERSION = "feedback-observation-qualification/1.0.0"
_NATIVE_TIMEBASE = "simulation_relative"
_REPLAY_COMPARISON_CONTRACT_VERSION = "feedback-comparison/1.1.0"
_REPLAY_IDENTITY_FIELDS = (
    "package_id",
    "variant_id",
    "drive_mode",
    "source_model_sha256",
    "provider_id",
    "provider_sha256",
    "state_schema_sha256",
    "policy_sha256",
    "applied_input_sha256",
    "input_kind",
    "input_interpolation",
    "timebase_id",
    "replay_policy",
    "evidence_mode",
    "horizon_s",
    "channel_ids",
    "full_state",
    "full_horizon",
    "state_resets",
    "nq",
    "nv",
    "bundle_schema",
    "initial_state_sha256",
    "comparison_contract_version",
    "physical_model_sha256",
    "loaded_native_model_sha256",
    "time_grid_sha256",
    "physics_sha256",
    "contact_sha256",
    "integrator_sha256",
    "input_channel_schema_sha256",
)


def _experiment_replay_bundle_type() -> type[Any]:
    """Resolve the Tools bundle only when a qualified bundle is consumed."""
    from src.engines.native_replay_contracts import native_replay_contract_types

    return native_replay_contract_types().ExperimentReplayBundle


@dataclass(frozen=True)
class NativeObservationCase:
    """One admitted replay with native marker output and measured positions."""

    evidence: ComparisonEvidence
    replay_bundle: ExperimentReplayBundle
    source_replay_identity_sha256: str
    native_output: NativeMarkerPositionOutput
    observations: ObservedMarkerPositions
    native_acceptance_receipt: Mapping[str, Any]
    horizon: Horizon
    interpolation: PositionInterpolation = PositionInterpolation.LINEAR_POSITION
    gates: AcceptanceGates | None = None
    capture: str | None = None
    native_marker_evidence: NativeMarkerReplayEvidence | None = None


@dataclass(frozen=True)
class _AlignedReplay:
    case: NativeObservationCase
    row: ComparisonRow
    replay_identity: str
    output_times: np.ndarray
    alignment: ReplayObservationAlignment
    metrics: ReplayFiveMetrics
    pelvis_yaw_diff_deg: float | None


@dataclass(frozen=True)
class ObservationScoreReceipt:
    """Content-addressed observation metrics plus the existing gate verdict."""

    receipt_sha256: str
    source_replay_identity_sha256: str
    evidence_identity_sha256: str
    initial_state_sha256: str
    package_id: str
    variant_id: str
    drive_mode: str
    engine: str
    bundle_schema: str
    capture: str | None
    native_acceptance_receipt_sha256: str
    native_marker_replay_evidence_sha256: str
    native_execution_receipt_sha256: str
    native_marker_map_sha256: str
    native_marker_output_sha256: str
    frame_id: str
    timebase_id: str
    marker_labels: tuple[str, ...]
    interpolation: str
    native_output_identity_sha256: str
    observation_identity_sha256: str
    alignment_identity_sha256: str
    native_output_time_grid_sha256: str
    observation_time_grid_sha256: str
    observation_times_s: tuple[float, ...]
    metrics: ReplayFiveMetrics
    pelvis_yaw_diff_deg: float | None
    criteria_sha256: str
    criterion_ids: tuple[str, ...]
    acceptance_verdict: AcceptanceVerdict

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": FEEDBACK_OBSERVATION_SCHEMA_VERSION,
                "receipt_sha256": self.receipt_sha256,
                "source_replay_identity_sha256": self.source_replay_identity_sha256,
                "evidence_identity_sha256": self.evidence_identity_sha256,
                "initial_state_sha256": self.initial_state_sha256,
                "package_id": self.package_id,
                "variant_id": self.variant_id,
                "drive_mode": self.drive_mode,
                "engine": self.engine,
                "bundle_schema": self.bundle_schema,
                "capture": self.capture,
                "native_acceptance_receipt_sha256": self.native_acceptance_receipt_sha256,
                "native_marker_replay_evidence_sha256": self.native_marker_replay_evidence_sha256,
                "native_execution_receipt_sha256": self.native_execution_receipt_sha256,
                "native_marker_map_sha256": self.native_marker_map_sha256,
                "native_marker_output_sha256": self.native_marker_output_sha256,
                "frame_id": self.frame_id,
                "timebase_id": self.timebase_id,
                "marker_labels": list(self.marker_labels),
                "interpolation": self.interpolation,
                "native_output_identity_sha256": self.native_output_identity_sha256,
                "observation_identity_sha256": self.observation_identity_sha256,
                "alignment_identity_sha256": self.alignment_identity_sha256,
                "native_output_time_grid_sha256": self.native_output_time_grid_sha256,
                "observation_time_grid_sha256": self.observation_time_grid_sha256,
                "observation_times_s": list(self.observation_times_s),
                "metrics": {
                    **self.metrics.as_dict(),
                    "pelvis_yaw_diff_deg": self.pelvis_yaw_diff_deg,
                },
                "criteria_sha256": self.criteria_sha256,
                "criterion_ids": list(self.criterion_ids),
                "acceptance_verdict": self.acceptance_verdict.as_dict(),
            }
        )


@dataclass(frozen=True)
class FeedbackObservationRow:
    """One required F01 inventory cell and its current observation evidence."""

    key: str
    package_id: str
    variant_id: str
    drive_mode: str
    engine: str
    required: bool
    support: str
    availability: str
    qualification: str
    status: str
    score: ObservationScoreReceipt | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "package_id": self.package_id,
            "variant_id": self.variant_id,
            "drive_mode": self.drive_mode,
            "engine": self.engine,
            "required": self.required,
            "support": self.support,
            "availability": self.availability,
            "qualification": self.qualification,
            "status": self.status,
            "score": self.score.as_dict() if self.score is not None else None,
        }


@dataclass(frozen=True)
class FeedbackObservationReport:
    """Per-cell observation evidence without qualification-axis promotion."""

    schema_version: str
    generated_at: str
    required_engine_ids: tuple[str, ...]
    observed_engine_ids: tuple[str, ...]
    missing_engine_ids: tuple[str, ...]
    required_row_count: int
    scored_row_count: int
    physically_accepted_row_count: int
    rows: tuple[FeedbackObservationRow, ...]

    @property
    def is_qualified(self) -> bool:
        """Only F01's qualified state plus every required passing gate can qualify."""
        return (
            not self.missing_engine_ids
            and self.scored_row_count == self.required_row_count
            and all(
                not row.required
                or (
                    row.status == "scored"
                    and row.qualification == "qualified"
                    and _row_is_physically_accepted(row)
                )
                for row in self.rows
            )
        )

    @property
    def status(self) -> str:
        if self.is_qualified:
            return "qualified"
        if self.scored_row_count < self.required_row_count or self.missing_engine_ids:
            return "evidence_incomplete"
        return "scored_unqualified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "required_engine_ids": list(self.required_engine_ids),
            "observed_engine_ids": list(self.observed_engine_ids),
            "missing_engine_ids": list(self.missing_engine_ids),
            "required_row_count": self.required_row_count,
            "scored_row_count": self.scored_row_count,
            "physically_accepted_row_count": self.physically_accepted_row_count,
            "is_qualified": self.is_qualified,
            "status": self.status,
            "rows": [row.as_dict() for row in self.rows],
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"unsupported receipt value: {type(value).__name__}")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_payload(evidence: ComparisonEvidence) -> dict[str, Any]:
    payload = {
        field.name: _json_safe(getattr(evidence, field.name))
        for field in fields(evidence)
    }
    # F01c owns these versioned replay-admission fields. Keeping them explicit
    # here makes a missing field fail closed even when an older registry object
    # is present at runtime.
    for name in ("initial_state_sha256", "comparison_contract_version"):
        payload[name] = _json_safe(getattr(evidence, name, ""))
    return payload


def feedback_replay_identity_sha256(
    evidence: ComparisonEvidence, replay_bundle: ExperimentReplayBundle
) -> str:
    """Bind F01 evidence to T01's fully hashed physical initial-state payload."""
    if not isinstance(replay_bundle, _experiment_replay_bundle_type()):
        raise TypeError("replay_bundle must be a validated ExperimentReplayBundle")
    return _canonical_sha256(
        {
            "schema": "feedback-native-replay-identity/2.0.0",
            "execution": {
                name: _json_safe(getattr(evidence, name))
                for name in _REPLAY_IDENTITY_FIELDS
            },
            "experiment_replay": {
                "schema_version": replay_bundle.schema_version,
                "experiment_id": replay_bundle.experiment_id,
                "initial_state_sha256": replay_bundle.integrity.initial_state_sha256,
                "model_identity_sha256": replay_bundle.integrity.model_identity_sha256,
                "capability_declarations_sha256": replay_bundle.integrity.capability_declarations_sha256,
            },
        }
    )


def _validate_replay_bundle(
    evidence: ComparisonEvidence,
    row: ComparisonRow,
    bundle: ExperimentReplayBundle,
) -> None:
    """Reject mismatched transport evidence before replay or scoring admission."""
    history = bundle.input_history
    model = bundle.model
    policy = bundle.policy
    mismatches = {
        "bundle_schema": evidence.bundle_schema == bundle.schema_version,
        "engine": model.engine_id == row.engine,
        "source_model": model.source_model_sha256 == evidence.source_model_sha256,
        "loaded_native_model": (
            model.loaded_native_model_sha256 == evidence.loaded_native_model_sha256
        ),
        "state_schema": bundle.state_schema_sha256 == evidence.state_schema_sha256,
        "integration_policy": bundle.policy_sha256 == evidence.policy_sha256,
        "applied_input": bundle.applied_input_sha256 == evidence.applied_input_sha256,
        "input_grid": bundle.time_grid_sha256 == evidence.time_grid_sha256,
        "input_channel_schema": (
            bundle.input_channel_schema_sha256 == evidence.input_channel_schema_sha256
        ),
        "channel_ids": tuple(channel.channel_id for channel in history.channels)
        == evidence.channel_ids,
        "input_kind": history.input_kind.value == evidence.input_kind.value,
        "input_interpolation": history.interpolation.value
        == evidence.input_interpolation,
        "timebase": history.timebase_id == evidence.timebase_id,
        "replay_policy": evidence.replay_policy.value == "independent_time_only"
        and not policy.observation_access
        and not policy.state_feedback_access
        and not policy.state_reset_allowed,
        "full_state": evidence.full_state
        and evidence.state_resets == 0
        and len(bundle.initial_state) == len(model.state_schema.components),
        "full_horizon": evidence.full_horizon,
        "nq": evidence.nq
        == sum(
            component.dimension
            for component in model.state_schema.components
            if component.role.value == "position"
        ),
        "nv": evidence.nv
        == sum(
            component.dimension
            for component in model.state_schema.components
            if component.role.value == "velocity"
        ),
        "evidence_mode": policy.replay_mode.value == evidence.evidence_mode.value,
        "horizon": math.isclose(
            history.time_seconds[-1], evidence.horizon_s, rel_tol=0.0, abs_tol=1e-12
        ),
        "contact_policy": (
            policy.contact_policy_sha256 is None
            or policy.contact_policy_sha256 == evidence.contact_sha256
        ),
        "comparison_contract_version": getattr(
            evidence, "comparison_contract_version", ""
        )
        == _REPLAY_COMPARISON_CONTRACT_VERSION,
        "initial_state_payload": getattr(evidence, "initial_state_sha256", "")
        == bundle.integrity.initial_state_sha256,
        "initial_state_complete": bool(bundle.initial_state)
        and len(bundle.initial_state) == len(model.state_schema.components),
    }
    failed = [name for name, matches in mismatches.items() if not matches]
    if failed:
        raise ValueError(
            "T01 replay bundle differs from comparison evidence: " + ", ".join(failed)
        )


def comparison_key(package_id: str, variant_id: str, drive_mode: str) -> str:
    """Stable key for one existing package/variant/coarse-drive row."""
    if not package_id.strip() or not variant_id.strip() or not drive_mode.strip():
        raise ValueError("comparison key fields must be non-empty")
    return f"{package_id}/{variant_id}:{drive_mode}"


def _case_key(case: NativeObservationCase) -> str:
    evidence = case.evidence
    drive_mode = evidence.drive_mode
    return comparison_key(
        evidence.package_id,
        evidence.variant_id,
        drive_mode.value,
    )


def _row_is_physically_accepted(row: FeedbackObservationRow) -> bool:
    score = row.score
    if score is None:
        return False
    verdict = score.acceptance_verdict
    return verdict.is_physically_accepted


def _metrics_and_yaw(
    alignment: ReplayObservationAlignment,
) -> tuple[ReplayFiveMetrics, float | None]:
    metrics = alignment.compute_replay_five_metrics()
    pelvis_yaw_diff_deg: float | None = None
    left = "WaistLeft"
    right = "WaistRight"
    if (
        left in alignment.marker_labels
        and right in alignment.marker_labels
        and bool(alignment.observation_valid[-1, alignment.marker_labels.index(left)])
        and bool(alignment.observation_valid[-1, alignment.marker_labels.index(right)])
    ):
        yaw = compute_pelvis_yaw_metrics(
            np.asarray(alignment.predicted_positions_m[-1]),
            np.asarray(alignment.observation_positions_m[-1]),
            alignment.marker_labels.index(left),
            alignment.marker_labels.index(right),
        )
        if yaw.valid:
            pelvis_yaw_diff_deg = float(yaw.yaw_diff_deg)
    return metrics, pelvis_yaw_diff_deg


def _criteria_identity(
    horizon: Horizon,
    gates: AcceptanceGates,
    verdict: AcceptanceVerdict,
    capture: str | None,
) -> tuple[str, tuple[str, ...]]:
    criterion_ids = tuple(gate.name for gate in verdict.gates)
    digest = _canonical_sha256(
        {
            "horizon": horizon.value,
            "capture": capture,
            "gate_configuration": asdict(gates),
            "criterion_ids": criterion_ids,
        }
    )
    return digest, criterion_ids


def _admit_and_align(
    registry: FeedbackComparisonRegistry, case: NativeObservationCase
) -> _AlignedReplay:
    evidence = case.evidence
    row = registry.get(evidence.package_id, evidence.variant_id, evidence.drive_mode)
    _validate_replay_bundle(evidence, row, case.replay_bundle)
    replay_identity = feedback_replay_identity_sha256(evidence, case.replay_bundle)
    if case.source_replay_identity_sha256 != replay_identity:
        raise ValueError("source replay identity is stale or does not match evidence")
    _validate_native_marker_evidence(case, row)
    if not isinstance(case.horizon, Horizon):
        raise ValueError("horizon must be Horizon enum")
    if not isinstance(case.native_acceptance_receipt, Mapping):
        raise ValueError("native_acceptance_receipt must be a mapping")
    if evidence.timebase_id != _NATIVE_TIMEBASE:
        raise ValueError("evidence timebase must be simulation_relative")
    if case.native_output.timebase_id != evidence.timebase_id:
        raise ValueError("native output timebase differs from replay evidence")
    output_times = np.asarray(case.native_output.time_s, dtype=np.float64)
    if output_times.ndim != 1 or len(output_times) < 2:
        raise ValueError("native output requires a full increasing time grid")
    if output_times[0] != 0.0 or not math.isclose(
        output_times[-1], evidence.horizon_s, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "native output horizon differs from the declared replay horizon"
        )

    registry.admit(evidence, ComparisonLevel.WITHIN_ENGINE_REPLAY)
    alignment = align_native_positions_to_observations(
        case.native_output,
        case.observations,
        interpolation=case.interpolation,
        source_identity_sha256=replay_identity,
    )
    metrics, pelvis_yaw_diff_deg = _metrics_and_yaw(alignment)
    return _AlignedReplay(
        case=case,
        row=row,
        replay_identity=replay_identity,
        output_times=output_times,
        alignment=alignment,
        metrics=metrics,
        pelvis_yaw_diff_deg=pelvis_yaw_diff_deg,
    )


def _validate_native_marker_evidence(
    case: NativeObservationCase, row: ComparisonRow
) -> None:
    """Bind scored marker bytes to the existing native execution receipt."""
    from src.engines.feedback_native_markers import NativeMarkerReplayEvidence

    evidence = case.native_marker_evidence
    if not isinstance(evidence, NativeMarkerReplayEvidence):
        raise ValueError("native marker replay evidence is required for scoring")
    evidence.validate_output(case.native_output)
    receipt = evidence.native_execution_receipt
    bundle = case.replay_bundle
    model = bundle.model
    history = bundle.input_history
    replay_mode = bundle.policy.replay_mode
    if not np.array_equal(
        np.asarray(case.native_output.time_s, dtype=np.float64),
        np.asarray(history.time_seconds, dtype=np.float64),
    ):
        raise ValueError("native marker output clock differs from replay input grid")
    expected = {
        "package_id": row.package_id,
        "variant_id": row.variant_id,
        "drive_mode": row.drive_mode.value,
        "engine": row.engine,
        "required": row.required,
        "support": row.support,
        "availability": row.availability,
        "source_model_sha256": row.source_model_sha256,
        "inventory_provider_id": row.provider_id,
        "inventory_provider_sha256": row.provider_sha256,
        "native_model_id": model.model_id,
        "native_variant_id": model.variant_id,
        "native_execution_provider_id": model.provider_id,
        "native_execution_provider_sha256": model.provider_sha256,
        "loaded_native_model_sha256": model.loaded_native_model_sha256,
        "state_schema_sha256": bundle.state_schema_sha256,
        "initial_state_sha256": bundle.integrity.initial_state_sha256,
        "input_channel_schema_sha256": bundle.input_channel_schema_sha256,
        "applied_input_sha256": bundle.applied_input_sha256,
        "policy_sha256": bundle.policy_sha256,
        "time_grid_sha256": bundle.time_grid_sha256,
        "channel_ids": tuple(channel.channel_id for channel in history.channels),
        "input_kind": history.input_kind.value,
        "interpolation": history.interpolation.value,
        "timebase_id": history.timebase_id,
        "evidence_mode": replay_mode.value,
        "horizon_s": float(history.time_seconds[-1] - history.time_seconds[0]),
        "state_sample_count": len(history.time_seconds),
        "applied_input_sample_count": len(history.values) - 1,
        "nq": case.evidence.nq,
        "nv": case.evidence.nv,
        "full_state": True,
        "full_horizon": True,
        "state_reset_allowed": False,
        "state_reset_count": None,
    }
    if receipt.schema_version != "native-execution/1.0.0" or any(
        getattr(receipt, name) != value for name, value in expected.items()
    ):
        raise ValueError("native marker execution receipt differs from replay bundle")
    if receipt.qualification != "unqualified":
        raise ValueError("native marker replay cannot promote qualification")


def _evaluate_native_acceptance(
    aligned: _AlignedReplay,
) -> tuple[str | None, str, tuple[str, ...], AcceptanceVerdict, str]:
    case = aligned.case
    row = aligned.row
    output_times = aligned.output_times
    metrics = aligned.metrics
    pelvis_yaw_diff_deg = aligned.pelvis_yaw_diff_deg
    accepted_receipt = dict(case.native_acceptance_receipt)
    if accepted_receipt.get("engine", row.engine) != row.engine:
        raise ValueError("native acceptance receipt engine differs from registry row")
    accepted_receipt.update(
        {
            "engine": row.engine,
            "duration_s": float(output_times[-1] - output_times[0]),
            "whole_marker_rmse_m": metrics.whole_rms_m,
            "early_marker_rmse_m": metrics.early_rms_m,
            "terminal_marker_rmse_m": metrics.terminal_rms_m,
            "club_marker_rmse_m": metrics.club_cluster_rms_m,
        }
    )
    declared_capture = case.capture or accepted_receipt.get("capture")
    if case.capture is not None and accepted_receipt.get("capture") not in (
        None,
        case.capture,
    ):
        raise ValueError("capture differs between case and native receipt")
    if case.horizon is Horizon.G3 and declared_capture not in {"driver", "iron"}:
        raise ValueError("G3 acceptance requires an explicit driver or iron capture")
    if pelvis_yaw_diff_deg is not None:
        accepted_receipt["pelvis_yaw_diff_deg"] = pelvis_yaw_diff_deg
    gates = case.gates or AcceptanceGates()
    verdict = evaluate(
        accepted_receipt,
        horizon=case.horizon,
        gates=gates,
        capture=declared_capture,
    )
    criteria_sha256, criterion_ids = _criteria_identity(
        case.horizon, gates, verdict, declared_capture
    )
    native_acceptance_receipt_sha256 = _canonical_sha256(case.native_acceptance_receipt)
    return (
        declared_capture,
        criteria_sha256,
        criterion_ids,
        verdict,
        native_acceptance_receipt_sha256,
    )


def _build_score_receipt(
    aligned: _AlignedReplay,
    declared_capture: str | None,
    criteria_sha256: str,
    criterion_ids: tuple[str, ...],
    verdict: AcceptanceVerdict,
    native_acceptance_receipt_sha256: str,
) -> ObservationScoreReceipt:
    case = aligned.case
    row = aligned.row
    replay_identity = aligned.replay_identity
    alignment = aligned.alignment
    metrics = aligned.metrics
    pelvis_yaw_diff_deg = aligned.pelvis_yaw_diff_deg
    evidence = case.evidence
    bundle = case.replay_bundle
    integrity = bundle.integrity
    initial_state_sha256 = integrity.initial_state_sha256
    drive_mode = row.drive_mode
    evidence_sha256 = _canonical_sha256(_evidence_payload(evidence))
    marker_evidence = case.native_marker_evidence
    if marker_evidence is None:
        raise ValueError("native marker replay evidence is required for scoring")
    marker_evidence_sha256 = _canonical_sha256(marker_evidence.as_dict())
    receipt_payload = {
        "schema_version": FEEDBACK_OBSERVATION_SCHEMA_VERSION,
        "source_replay_identity_sha256": replay_identity,
        "evidence_identity_sha256": evidence_sha256,
        "initial_state_sha256": initial_state_sha256,
        "package_id": row.package_id,
        "variant_id": row.variant_id,
        "drive_mode": drive_mode.value,
        "engine": row.engine,
        "bundle_schema": evidence.bundle_schema,
        "capture": declared_capture,
        "native_acceptance_receipt_sha256": native_acceptance_receipt_sha256,
        "native_marker_replay_evidence_sha256": marker_evidence_sha256,
        "native_execution_receipt_sha256": marker_evidence.receipt_sha256,
        "native_marker_map_sha256": marker_evidence.marker_map_sha256,
        "native_marker_output_sha256": marker_evidence.marker_output_sha256,
        "frame_id": alignment.frame_id,
        "timebase_id": alignment.timebase_id,
        "marker_labels": alignment.marker_labels,
        "interpolation": alignment.interpolation.value,
        "native_output_identity_sha256": alignment.output_identity_sha256,
        "observation_identity_sha256": alignment.observation_identity_sha256,
        "alignment_identity_sha256": alignment.alignment_identity_sha256,
        "native_output_time_grid_sha256": alignment.native_output_time_grid_sha256,
        "observation_time_grid_sha256": alignment.observation_time_grid_sha256,
        "observation_times_s": tuple(float(t) for t in alignment.observation_time_s),
        "metrics": {
            **metrics.as_dict(),
            "pelvis_yaw_diff_deg": pelvis_yaw_diff_deg,
        },
        "criteria_sha256": criteria_sha256,
        "criterion_ids": criterion_ids,
        "acceptance_verdict": verdict.as_dict(),
    }
    receipt_sha256 = _canonical_sha256(receipt_payload)
    return ObservationScoreReceipt(
        receipt_sha256=receipt_sha256,
        source_replay_identity_sha256=replay_identity,
        evidence_identity_sha256=evidence_sha256,
        initial_state_sha256=initial_state_sha256,
        package_id=row.package_id,
        variant_id=row.variant_id,
        drive_mode=drive_mode.value,
        engine=row.engine,
        bundle_schema=evidence.bundle_schema,
        capture=declared_capture,
        native_acceptance_receipt_sha256=native_acceptance_receipt_sha256,
        native_marker_replay_evidence_sha256=marker_evidence_sha256,
        native_execution_receipt_sha256=marker_evidence.receipt_sha256,
        native_marker_map_sha256=marker_evidence.marker_map_sha256,
        native_marker_output_sha256=marker_evidence.marker_output_sha256,
        frame_id=alignment.frame_id,
        timebase_id=alignment.timebase_id,
        marker_labels=alignment.marker_labels,
        interpolation=alignment.interpolation.value,
        native_output_identity_sha256=alignment.output_identity_sha256,
        observation_identity_sha256=alignment.observation_identity_sha256,
        alignment_identity_sha256=alignment.alignment_identity_sha256,
        native_output_time_grid_sha256=alignment.native_output_time_grid_sha256,
        observation_time_grid_sha256=alignment.observation_time_grid_sha256,
        observation_times_s=tuple(float(t) for t in alignment.observation_time_s),
        metrics=metrics,
        pelvis_yaw_diff_deg=pelvis_yaw_diff_deg,
        criteria_sha256=criteria_sha256,
        criterion_ids=criterion_ids,
        acceptance_verdict=verdict,
    )


def _score_case(
    registry: FeedbackComparisonRegistry, case: NativeObservationCase
) -> tuple[ComparisonRow, ObservationScoreReceipt]:
    evidence = case.evidence
    aligned = _admit_and_align(registry, case)
    declared_capture, criteria_sha256, criterion_ids, verdict, acceptance_sha256 = (
        _evaluate_native_acceptance(aligned)
    )
    score = _build_score_receipt(
        aligned,
        declared_capture,
        criteria_sha256,
        criterion_ids,
        verdict,
        acceptance_sha256,
    )
    registry.admit(
        replace(
            evidence,
            observation_score_receipt_sha256=score.receipt_sha256,
            observation_time_grid_sha256=aligned.alignment.observation_time_grid_sha256,
        ),
        ComparisonLevel.OBSERVATION_ACCURACY,
    )
    return aligned.row, score


def _missing_row(row: ComparisonRow) -> FeedbackObservationRow:
    return FeedbackObservationRow(
        key=comparison_key(row.package_id, row.variant_id, row.drive_mode.value),
        package_id=row.package_id,
        variant_id=row.variant_id,
        drive_mode=row.drive_mode.value,
        engine=row.engine,
        required=row.required,
        support=row.support,
        availability=row.availability,
        qualification=row.qualification,
        status="missing_evidence",
    )


def build_feedback_observation_report(
    registry: FeedbackComparisonRegistry,
    cases: Sequence[NativeObservationCase],
    *,
    generated_at: str | None = None,
) -> FeedbackObservationReport:
    """Score supplied native rows and retain every required inventory cell."""
    case_map: dict[str, NativeObservationCase] = {}
    for observation_case in cases:
        key = _case_key(observation_case)
        if key in case_map:
            raise ValueError(f"duplicate native observation row: {key}")
        case_map[key] = observation_case

    inventory_rows = {
        comparison_key(row.package_id, row.variant_id, row.drive_mode.value): row
        for row in registry.rows
    }
    unknown = set(case_map) - set(inventory_rows)
    if unknown:
        raise ValueError(f"unknown comparison row(s): {sorted(unknown)}")

    rows: list[FeedbackObservationRow] = []
    for key, inventory_row in sorted(inventory_rows.items()):
        case = case_map.get(key)
        if case is None:
            rows.append(_missing_row(inventory_row))
            continue
        admitted_row, score = _score_case(registry, case)
        rows.append(
            FeedbackObservationRow(
                key=key,
                package_id=admitted_row.package_id,
                variant_id=admitted_row.variant_id,
                drive_mode=admitted_row.drive_mode.value,
                engine=admitted_row.engine,
                required=admitted_row.required,
                support=admitted_row.support,
                availability=admitted_row.availability,
                qualification=admitted_row.qualification,
                status="scored",
                score=score,
            )
        )

    required_rows = tuple(row for row in rows if row.required)
    required_engine_ids = tuple(sorted(TARGET_ENGINES))
    observed_engine_ids = tuple(
        sorted({row.engine for row in required_rows if row.status == "scored"})
    )
    missing_engine_ids = tuple(
        engine for engine in required_engine_ids if engine not in observed_engine_ids
    )
    score_count = sum(row.status == "scored" for row in required_rows)
    accepted_count = sum(_row_is_physically_accepted(row) for row in required_rows)
    timestamp = generated_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return FeedbackObservationReport(
        schema_version=FEEDBACK_OBSERVATION_SCHEMA_VERSION,
        generated_at=timestamp,
        required_engine_ids=required_engine_ids,
        observed_engine_ids=observed_engine_ids,
        missing_engine_ids=missing_engine_ids,
        required_row_count=len(required_rows),
        scored_row_count=score_count,
        physically_accepted_row_count=accepted_count,
        rows=tuple(rows),
    )
