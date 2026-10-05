"""DIME Per-Engine and Capture Qualification Matrix (#11433).

Part of Epic #11421.
Dependencies: #11427 (Contact Constraints), #11430 (Offline Smoothing & Replay),
              #11432 (Shared Reports and LaTeX Methods).

Provides:
1. Truthful per-engine and capture qualification contracts.
2. Rejection of unevidenced capability claims (API skeletons, contact-free as GRF,
   joint convention mismatches, missing native binaries).
3. Independent continuous forward replay evaluation against frozen numeric acceptance thresholds.
4. Fleet-wide qualification matrix tracking qualified, provisional, unsupported, and blocked engines.
5. Report bundle and manifest exporter preserving privacy and provenance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Final, Literal

from src.shared.python.contracts import PreconditionError, require
from src.shared.python.estimation.dime_manifest import (
    CANONICAL_DIME_UNITS,
    NumericAcceptanceThresholds,
)

DIME_QUALIFICATION_SCHEMA_VERSION: Final[str] = "dime-engine-qualification-v1"

# Metrics compared against frozen thresholds; all must be present and finite.
_GATED_METRIC_KEYS: Final[tuple[str, ...]] = (
    "max_drift_m",
    "max_angular_drift_rad",
    "alignment",
)


class EngineQualificationStatus(str, Enum):
    """Engine qualification state for forward dynamics matching."""

    QUALIFIED = "qualified"
    UNQUALIFIED = "unqualified"  # required evidence missing or not measured
    PROVISIONAL = "provisional"
    UNSUPPORTED = "unsupported"
    BLOCKED = "blocked"
    REJECTED = "rejected"


class ContactModelKind(str, Enum):
    """Supported contact physical interaction model."""

    WHOLE_BODY_GRF = "whole_body_grf"
    POINT_CONTACT = "point_contact"
    CONTACT_FREE = "contact_free"
    SURFACE_PATCH = "surface_patch"


class JointConvention(str, Enum):
    """Kinematic joint coordinate convention."""

    EULER_XYZ = "euler_xyz"
    EULER_ZXY = "euler_zxy"
    QUATERNION = "quaternion"
    SPATIAL_VECTOR = "spatial_vector"


@dataclass(frozen=True)
class EngineCapabilitySpec:
    """Declared capability specification of a physics engine."""

    engine_name: str
    version: str
    supported_states: tuple[str, ...]
    supported_controls: tuple[str, ...]
    contact_model: ContactModelKind
    joint_convention: JointConvention
    has_forward_dynamics: bool
    has_analytic_derivatives: bool
    has_continuous_replay: bool
    accepted_model_formats: tuple[str, ...]
    is_api_skeleton_only: bool = False
    native_binary_present: bool = True

    def __post_init__(self) -> None:
        if not self.engine_name:
            raise PreconditionError("engine_name cannot be empty")
        if not self.version:
            raise PreconditionError("version cannot be empty")


@dataclass(frozen=True)
class CaptureProvenance:
    """Provenance and physical claims of a mocap/biomechanics capture."""

    capture_id: str
    capture_type: Literal["tour", "owner", "synthetic"]
    model_hash: str
    data_hash: str
    sampling_rate_hz: float
    subject_id: str
    claimed_contact: ContactModelKind
    claimed_joint_convention: JointConvention

    def __post_init__(self) -> None:
        if not self.capture_id:
            raise PreconditionError("capture_id cannot be empty")
        if self.sampling_rate_hz <= 0.0:
            raise PreconditionError(
                f"sampling_rate_hz must be positive, got {self.sampling_rate_hz}"
            )


@dataclass(frozen=True)
class EngineQualificationEntry:
    """Qualification verdict for an engine evaluated on a specific capture."""

    engine_name: str
    capture_id: str
    status: EngineQualificationStatus
    reasons: tuple[str, ...]
    max_position_drift_m: float
    max_angular_drift_rad: float
    mean_control_norm_nm: float
    alignment_metric: float
    matched_forward_dynamics: bool
    report_bundle_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_name": self.engine_name,
            "capture_id": self.capture_id,
            "status": self.status.value,
            "reasons": list(self.reasons),
            "max_position_drift_m": self.max_position_drift_m,
            "max_angular_drift_rad": self.max_angular_drift_rad,
            "mean_control_norm_nm": self.mean_control_norm_nm,
            "alignment_metric": self.alignment_metric,
            "matched_forward_dynamics": self.matched_forward_dynamics,
            "report_bundle_path": self.report_bundle_path,
        }


@dataclass(frozen=True)
class EngineQualificationMatrix:
    """Full cross-engine and capture qualification matrix."""

    entries: tuple[EngineQualificationEntry, ...]
    evaluated_at: str
    schema_version: str = DIME_QUALIFICATION_SCHEMA_VERSION

    @property
    def total_evaluated(self) -> int:
        return len(self.entries)

    @property
    def total_qualified(self) -> int:
        return sum(
            1 for e in self.entries if e.status == EngineQualificationStatus.QUALIFIED
        )

    @property
    def total_blocked(self) -> int:
        return sum(
            1 for e in self.entries if e.status == EngineQualificationStatus.BLOCKED
        )

    @property
    def total_rejected(self) -> int:
        return sum(
            1 for e in self.entries if e.status == EngineQualificationStatus.REJECTED
        )

    @property
    def total_unsupported(self) -> int:
        return sum(
            1 for e in self.entries if e.status == EngineQualificationStatus.UNSUPPORTED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluated_at": self.evaluated_at,
            "total_evaluated": self.total_evaluated,
            "total_qualified": self.total_qualified,
            "total_blocked": self.total_blocked,
            "total_rejected": self.total_rejected,
            "total_unsupported": self.total_unsupported,
            "entries": [e.to_dict() for e in self.entries],
        }


def _evaluate_capability_preconditions(
    engine_spec: EngineCapabilitySpec,
    capture: CaptureProvenance,
) -> tuple[EngineQualificationStatus, str] | None:
    """Evaluate fail-closed engine and capture capability alignment."""
    if engine_spec.is_api_skeleton_only or not engine_spec.has_forward_dynamics:
        return (
            EngineQualificationStatus.BLOCKED,
            "API skeleton only: forward dynamics is not implemented or qualified",
        )
    if not engine_spec.native_binary_present:
        return (
            EngineQualificationStatus.BLOCKED,
            f"Missing native dependencies for engine '{engine_spec.engine_name}'",
        )
    if (
        engine_spec.contact_model == ContactModelKind.CONTACT_FREE
        and capture.claimed_contact == ContactModelKind.WHOLE_BODY_GRF
    ):
        return (
            EngineQualificationStatus.REJECTED,
            "Contact-free model cannot satisfy whole-body GRF ground reaction force contract",
        )
    if engine_spec.joint_convention != capture.claimed_joint_convention:
        return (
            EngineQualificationStatus.REJECTED,
            f"Joint convention mismatch: engine uses {engine_spec.joint_convention.value} "
            f"but capture claims {capture.claimed_joint_convention.value}",
        )
    return None


def _missing_metric_reasons(
    trajectory_metrics: Mapping[str, float] | None,
) -> tuple[str, ...]:
    """Name every gated metric that is absent or non-finite (not measured)."""
    metrics = trajectory_metrics or {}
    return tuple(
        f"Missing trajectory metric '{key}': not measured"
        for key in _GATED_METRIC_KEYS
        if metrics.get(key) is None or not math.isfinite(float(metrics[key]))
    )


def _evaluate_physical_tolerances(
    trajectory_metrics: Mapping[str, float] | None,
    thresholds: NumericAcceptanceThresholds,
) -> tuple[tuple[str, ...], float, float, float, float]:
    """Evaluate replay trajectory metrics against frozen physical tolerance thresholds."""
    metrics = trajectory_metrics or {}
    require(
        not _missing_metric_reasons(trajectory_metrics),
        "gated trajectory metrics must be present and finite",
    )
    reasons: list[str] = []
    max_drift = float(metrics["max_drift_m"])
    max_angular_drift = float(metrics["max_angular_drift_rad"])
    mean_control = float(metrics.get("mean_control_nm", 0.0))
    alignment = float(metrics["alignment"])

    if max_drift > thresholds.max_drift_m:
        reasons.append(
            f"Position drift {max_drift:.4f}m exceeds frozen threshold {thresholds.max_drift_m:.4f}m"
        )
    if max_angular_drift > thresholds.max_angular_drift_rad:
        reasons.append(
            f"Angular drift {max_angular_drift:.4f}rad exceeds frozen threshold {thresholds.max_angular_drift_rad:.4f}rad"
        )
    if alignment < thresholds.min_alignment:
        reasons.append(
            f"Alignment {alignment:.4f} below frozen minimum {thresholds.min_alignment:.4f}"
        )

    return tuple(reasons), max_drift, max_angular_drift, mean_control, alignment


def evaluate_engine_qualification(
    engine_spec: EngineCapabilitySpec,
    capture: CaptureProvenance,
    *,
    trajectory_metrics: Mapping[str, float] | None = None,
    thresholds: NumericAcceptanceThresholds | None = None,
) -> EngineQualificationEntry:
    """Evaluate qualification of an engine against capture requirements and physics gates."""
    thresholds = thresholds or NumericAcceptanceThresholds()

    precondition_failure = _evaluate_capability_preconditions(engine_spec, capture)
    if precondition_failure is not None:
        status, reason = precondition_failure
        return EngineQualificationEntry(
            engine_name=engine_spec.engine_name,
            capture_id=capture.capture_id,
            status=status,
            reasons=(reason,),
            max_position_drift_m=float("inf"),
            max_angular_drift_rad=float("inf"),
            mean_control_norm_nm=float("inf"),
            alignment_metric=0.0,
            matched_forward_dynamics=False,
        )

    missing = _missing_metric_reasons(trajectory_metrics)
    if missing:
        # Fail closed: unavailable evidence is never evidence of qualification.
        return EngineQualificationEntry(
            engine_name=engine_spec.engine_name,
            capture_id=capture.capture_id,
            status=EngineQualificationStatus.UNQUALIFIED,
            reasons=missing,
            max_position_drift_m=float("inf"),
            max_angular_drift_rad=float("inf"),
            mean_control_norm_nm=float("inf"),
            alignment_metric=0.0,
            matched_forward_dynamics=False,
        )

    reasons, max_drift, max_angular, mean_ctrl, align = _evaluate_physical_tolerances(
        trajectory_metrics, thresholds
    )
    is_qualified = len(reasons) == 0
    status = (
        EngineQualificationStatus.QUALIFIED
        if is_qualified
        else EngineQualificationStatus.REJECTED
    )

    return EngineQualificationEntry(
        engine_name=engine_spec.engine_name,
        capture_id=capture.capture_id,
        status=status,
        reasons=reasons,
        max_position_drift_m=max_drift,
        max_angular_drift_rad=max_angular,
        mean_control_norm_nm=mean_ctrl,
        alignment_metric=align,
        matched_forward_dynamics=is_qualified,
    )


def build_fleet_qualification_matrix(
    engine_specs: Sequence[EngineCapabilitySpec],
    captures: Sequence[CaptureProvenance],
    solve_receipts: Mapping[tuple[str, str], Mapping[str, float]],
    thresholds: NumericAcceptanceThresholds | None = None,
) -> EngineQualificationMatrix:
    """Build the cross-engine, cross-capture qualification matrix."""
    thresholds = thresholds or NumericAcceptanceThresholds()
    entries: list[EngineQualificationEntry] = []

    for engine in engine_specs:
        for capture in captures:
            key = (engine.engine_name, capture.capture_id)
            metrics = solve_receipts.get(key)
            entry = evaluate_engine_qualification(
                engine,
                capture,
                trajectory_metrics=metrics,
                thresholds=thresholds,
            )
            entries.append(entry)

    now_iso = datetime.now(timezone.utc).isoformat()
    return EngineQualificationMatrix(
        entries=tuple(entries),
        evaluated_at=now_iso,
    )


def export_qualification_bundle(
    matrix: EngineQualificationMatrix,
    output_dir: Path | str,
    *,
    generate_manifest: bool = True,
) -> Path:
    """Export qualification matrix JSON and artifact manifest to output directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    matrix_file = out_path / "qualification_matrix.json"
    matrix_file.write_text(json.dumps(matrix.to_dict(), indent=2), encoding="utf-8")

    if generate_manifest:
        manifest = {
            "schema_version": matrix.schema_version,
            "generated_at": matrix.evaluated_at,
            "units": dict(CANONICAL_DIME_UNITS),
            "artifacts": ["qualification_matrix.json"],
            "summary": {
                "total_evaluated": matrix.total_evaluated,
                "total_qualified": matrix.total_qualified,
                "total_blocked": matrix.total_blocked,
                "total_rejected": matrix.total_rejected,
            },
        }
        manifest_file = out_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return out_path
