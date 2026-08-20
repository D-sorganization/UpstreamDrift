"""Version 2 wire contract for traceable launch-monitor analysis.

The numerical implementation remains in :mod:`flexible_analysis`. This module
is the canonical serialization boundary shared by API, desktop, web, notebook,
and cross-repository consumers. Contract v1 remains intact; v2 wraps it with
the evidence required to interpret or reproduce a result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from hashlib import sha256
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.shared.python.launch_monitor.flexible_analysis import (
    CONTRACT_VERSION,
    FlexibleAnalysisRequest,
    FlexibleAnalysisResult,
    analyze_variables,
)
from src.shared.python.launch_monitor.schema import METRICS

CONTRACT_VERSION_V2: Literal["2.0.0"] = "2.0.0"
PlayerIdentityTrust = Literal[
    "not_provided",
    "explicit_user_attested",
    "pseudonymous_stable",
    "verified_external",
    "untrusted_inferred",
]
SessionIdentityTrust = Literal[
    "not_provided",
    "explicit_user_attested",
    "source_reported",
    "verified_external",
    "untrusted_inferred",
]
OrderEvidenceTrust = Literal[
    "not_provided",
    "explicit_user_attested",
    "source_reported",
    "verified_external",
    "untrusted_inferred",
]
OrderKind = Literal["timestamp", "ordinal", "source_sequence"]
AvailabilityState = Literal["available", "partial", "unavailable"]
UnlinkedReason = Literal[
    "no_source_reference_declared",
    "session_not_linked_to_source_reference",
]

_FORBIDDEN_PLAYER_IDENTIFIERS = frozenset(
    {
        "session",
        "session_id",
        "club",
        "club_id",
        "source",
        "source_id",
        "file",
        "filename",
        "file_name",
        "row_order",
        "source_row",
    }
)


def _normalized_identifier(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("-", "_")
    return normalized.replace(" ", "_")


class _ContractModel(BaseModel):
    """Strict, immutable base for every externally serialized v2 record."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetAuthorityV2(_ContractModel):
    """Commit-addressable authority for the analyzed dataset."""

    dataset_id: str = Field(min_length=1)
    repository: str | None = None
    commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    dataset_path: str | None = None
    manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PlayerIdentityV2(_ContractModel):
    """Declared identity evidence; identity is never inferred from layout."""

    trust_level: PlayerIdentityTrust = "not_provided"
    identifier_column: str | None = Field(
        default=None,
        description=(
            "Explicit player identifier column; session, club, source, file, and "
            "row-position fields are forbidden."
        ),
        json_schema_extra={
            "not": {"enum": cast(list[Any], sorted(_FORBIDDEN_PLAYER_IDENTIFIERS))}
        },
    )
    evidence: str | None = None

    @model_validator(mode="after")
    def require_explicit_evidence(self) -> PlayerIdentityV2:
        normalized_column = (
            _normalized_identifier(self.identifier_column)
            if self.identifier_column is not None
            else None
        )
        if normalized_column in _FORBIDDEN_PLAYER_IDENTIFIERS:
            raise ValueError(
                f"{self.identifier_column!r} cannot be used as player identity; "
                "declare session and order evidence separately"
            )
        if self.trust_level in {
            "explicit_user_attested",
            "pseudonymous_stable",
            "verified_external",
        } and (not self.identifier_column or not self.evidence):
            raise ValueError(
                "trusted player identity requires identifier_column and evidence"
            )
        return self


class SessionIdentityV2(_ContractModel):
    """Evidence for repeated-observation session boundaries, never a player ID."""

    trust_level: SessionIdentityTrust = "not_provided"
    identifier_column: str | None = None
    evidence: str | None = None

    @model_validator(mode="after")
    def require_complete_evidence(self) -> SessionIdentityV2:
        if self.trust_level != "not_provided" and (
            not self.identifier_column or not self.evidence
        ):
            raise ValueError(
                "declared session identity requires identifier_column and evidence"
            )
        if self.trust_level == "not_provided" and (
            self.identifier_column is not None or self.evidence is not None
        ):
            raise ValueError(
                "session identity fields require a non-default trust_level"
            )
        return self


class OrderEvidenceV2(_ContractModel):
    """Evidence defining chronological or ordinal order for longitudinal work."""

    trust_level: OrderEvidenceTrust = "not_provided"
    order_column: str | None = None
    order_kind: OrderKind | None = None
    unit: str | None = None
    evidence: str | None = None

    @model_validator(mode="after")
    def require_complete_evidence(self) -> OrderEvidenceV2:
        evidence_fields = (
            self.order_column,
            self.order_kind,
            self.unit,
            self.evidence,
        )
        if self.trust_level != "not_provided" and not all(evidence_fields):
            raise ValueError(
                "declared order evidence requires order_column, order_kind, unit, "
                "and evidence"
            )
        if self.trust_level == "not_provided" and any(
            field is not None for field in evidence_fields
        ):
            raise ValueError("order evidence fields require a non-default trust_level")
        return self


class TransformRecordV2(_ContractModel):
    """Versioned transformation applied before analysis."""

    transform_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    parameters_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceFileReferenceV2(_ContractModel):
    """Content-addressed source and the sessions it backs."""

    source_id: str = Field(min_length=1)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    session_ids: tuple[str, ...] = ()
    source_uri: str | None = None
    rights_status: Literal[
        "public_redistributable",
        "restricted_internal",
        "permission_required",
        "unknown",
    ] = "unknown"


class AnalysisContextV2(_ContractModel):
    """Caller-supplied lineage and identity assertions."""

    authority: DatasetAuthorityV2 | None = None
    player_identity: PlayerIdentityV2 = Field(default_factory=PlayerIdentityV2)
    session_identity: SessionIdentityV2 = Field(default_factory=SessionIdentityV2)
    order_evidence: OrderEvidenceV2 = Field(default_factory=OrderEvidenceV2)
    transformations: tuple[TransformRecordV2, ...] = ()
    sources: tuple[SourceFileReferenceV2, ...] = ()
    source_units: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source_links(self) -> AnalysisContextV2:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("context source_id values must be unique")
        session_ids = [
            session for source in self.sources for session in source.session_ids
        ]
        if len(session_ids) != len(set(session_ids)):
            raise ValueError("a session_id cannot link to multiple source references")
        invalid_units = [
            name
            for name, unit in self.source_units.items()
            if not name or not unit.strip()
        ]
        if invalid_units:
            raise ValueError("source_units keys and values must be non-empty")
        return self


class MetricUnitsV2(_ContractModel):
    canonical_unit: str
    display_unit: str
    authority: Literal["canonical_registry", "source_declared", "unknown"]


class BackingRecordV2(_ContractModel):
    """Stable reference to exactly one input record without copying values."""

    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shot_id: str | None = None
    session_id: str | None = None
    source_row: int | str | None = None
    source_id: str | None = None
    unlinked_reason: UnlinkedReason | None = None

    @model_validator(mode="after")
    def require_link_or_reason(self) -> BackingRecordV2:
        if (self.source_id is None) == (self.unlinked_reason is None):
            raise ValueError(
                "backing record requires exactly one source_id or unlinked_reason"
            )
        return self


class AnalysisLineageV2(_ContractModel):
    dataset_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority: DatasetAuthorityV2 | None = None
    transformations: tuple[TransformRecordV2, ...] = ()
    sources: tuple[SourceFileReferenceV2, ...] = ()
    backing_records: tuple[BackingRecordV2, ...]


class MissingnessV2(_ContractModel):
    input_row_count: int = Field(ge=0)
    complete_row_count: int = Field(ge=0)
    missing_by_variable: dict[str, int]
    non_numeric_by_variable: dict[str, int]
    excluded_by_reason: dict[str, int]
    policy: Literal["pairwise", "listwise", "fail"]


class AvailabilityV2(_ContractModel):
    result_path: str
    state: Literal["available", "unavailable"]
    reason_code: str | None = None
    message: str | None = None
    observed_count: int | None = Field(default=None, ge=0)
    required_count: int | None = Field(default=None, ge=0)


class UncertaintyV2(_ContractModel):
    confidence_level: float = Field(gt=0.5, lt=1.0)
    correlation_interval: str
    regression_interval: str
    multiplicity_adjustment: str
    assumptions: tuple[str, ...]


class VendorProvenanceV2(_ContractModel):
    vendor: str
    models: tuple[str, ...]
    software_versions: tuple[str, ...]
    row_count: int = Field(ge=1)
    metric_statuses: dict[str, tuple[str, ...]]


class ModelProvenanceV2(_ContractModel):
    model_id: str
    version: str
    code_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    configuration_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    relationship_to_vendor: Literal[
        "independent_physics",
        "vendor_comparable_surrogate",
        "vendor_reported_output",
        "unknown",
    ] = "unknown"


class ClaimsV2(_ContractModel):
    vendor_comparison: Literal["descriptive", "matched_agreement"] = "descriptive"
    device_emulation: bool = False
    device_certification: bool = False
    causal_inference: bool = False


class LaunchMonitorAnalysisResultV2(_ContractModel):
    """Complete OpenAPI-compatible v2 result envelope."""

    contract_version: Literal["2.0.0"] = CONTRACT_VERSION_V2
    status: AvailabilityState
    analysis: dict[str, Any] | None
    units: dict[str, MetricUnitsV2]
    lineage: AnalysisLineageV2
    missingness: MissingnessV2
    availability: tuple[AvailabilityV2, ...]
    uncertainty: UncertaintyV2
    player_identity: PlayerIdentityV2
    session_identity: SessionIdentityV2 = Field(default_factory=SessionIdentityV2)
    order_evidence: OrderEvidenceV2 = Field(default_factory=OrderEvidenceV2)
    vendor_provenance: tuple[VendorProvenanceV2, ...]
    model_provenance: tuple[ModelProvenanceV2, ...] = ()
    claims: ClaimsV2 = Field(default_factory=ClaimsV2)
    warnings: tuple[str, ...] = ()


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _record_digest(record: dict[str, Any]) -> str:
    normalized = {str(key): _json_value(value) for key, value in record.items()}
    payload = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _backing_records(
    frame: pd.DataFrame, context: AnalysisContextV2
) -> tuple[BackingRecordV2, ...]:
    declared_sources = {source.source_id for source in context.sources}
    source_by_session = {
        session_id: source.source_id
        for source in context.sources
        for session_id in source.session_ids
    }
    records: list[BackingRecordV2] = []
    for raw in frame.to_dict(orient="records"):
        source_row = _json_value(raw.get("source_row"))
        if source_row is not None and not isinstance(source_row, (int, str)):
            source_row = str(source_row)
        session_id = _optional_text(raw.get("session_id"))
        source_id = _optional_text(raw.get("source_id"))
        if source_id is not None and source_id not in declared_sources:
            raise ValueError(
                f"backing source_id {source_id!r} is not declared in context.sources"
            )
        source_id = source_id or (
            source_by_session.get(session_id) if session_id is not None else None
        )
        unlinked_reason: UnlinkedReason | None
        if source_id is not None:
            unlinked_reason = None
        elif context.sources:
            unlinked_reason = "session_not_linked_to_source_reference"
        else:
            unlinked_reason = "no_source_reference_declared"
        records.append(
            BackingRecordV2(
                record_sha256=_record_digest(raw),
                shot_id=_optional_text(raw.get("shot_id")),
                session_id=session_id,
                source_row=source_row,
                source_id=source_id,
                unlinked_reason=unlinked_reason,
            )
        )
    return tuple(records)


def _missingness(
    frame: pd.DataFrame, request: FlexibleAnalysisRequest
) -> MissingnessV2:
    selected = (request.outcome, *request.predictors)
    raw = frame[list(selected)]
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    missing = {column: int(raw[column].isna().sum()) for column in selected}
    non_numeric = {
        column: int((raw[column].notna() & numeric[column].isna()).sum())
        for column in selected
    }
    complete = int(numeric.dropna().shape[0])
    exclusions = {
        "regression_incomplete": len(frame) - complete,
        **{
            f"correlation_incomplete::{predictor}": int(
                numeric[[request.outcome, predictor]].isna().any(axis=1).sum()
            )
            for predictor in request.predictors
        },
    }
    return MissingnessV2(
        input_row_count=len(frame),
        complete_row_count=complete,
        missing_by_variable=missing,
        non_numeric_by_variable=non_numeric,
        excluded_by_reason=exclusions,
        policy=request.missing_policy,
    )


def _unique_text(frame: pd.DataFrame, column: str) -> tuple[str, ...]:
    if column not in frame:
        return ()
    values = frame[column].dropna().astype(str)
    return tuple(sorted(value for value in values.unique() if value.strip()))


def _vendor_provenance(
    frame: pd.DataFrame, selected: tuple[str, ...]
) -> tuple[VendorProvenanceV2, ...]:
    if "monitor_vendor" not in frame:
        return ()
    items: list[VendorProvenanceV2] = []
    vendors = frame["monitor_vendor"].fillna("").astype(str)
    for vendor in sorted(value for value in vendors.unique() if value.strip()):
        subset = frame.loc[vendors == vendor]
        statuses: dict[str, tuple[str, ...]] = {}
        for metric in selected:
            column = f"status::{metric}"
            if column in subset:
                statuses[metric] = _unique_text(subset, column)
        items.append(
            VendorProvenanceV2(
                vendor=vendor,
                models=_unique_text(subset, "monitor_model"),
                software_versions=_unique_text(subset, "software_version"),
                row_count=len(subset),
                metric_statuses=statuses,
            )
        )
    return tuple(items)


def _availability(
    result: FlexibleAnalysisResult, request: FlexibleAnalysisRequest
) -> tuple[AvailabilityV2, ...]:
    items: list[AvailabilityV2] = []
    for estimate in result.correlations:
        path = f"correlations.{estimate.predictor}"
        if np.isfinite(estimate.coefficient):
            items.append(AvailabilityV2(result_path=path, state="available"))
        else:
            items.append(
                AvailabilityV2(
                    result_path=path,
                    state="unavailable",
                    reason_code="insufficient_samples",
                    message="The complete pair count is below min_samples.",
                    observed_count=estimate.sample_count,
                    required_count=request.min_samples,
                )
            )
    if request.analysis_mode in {"regression", "comprehensive"}:
        available = result.regression is not None
        items.append(
            AvailabilityV2(
                result_path="regression",
                state="available" if available else "unavailable",
                reason_code=None if available else "not_computed",
                message=None if available else "Regression was not computed.",
            )
        )
    return tuple(items)


def _overall_status(items: tuple[AvailabilityV2, ...]) -> AvailabilityState:
    if not items or all(item.state == "unavailable" for item in items):
        return "unavailable"
    if any(item.state == "unavailable" for item in items):
        return "partial"
    return "available"


def _assert_identity_safe(
    request: FlexibleAnalysisRequest, context: AnalysisContextV2
) -> None:
    player_columns = {"player", "player_id"}
    if context.player_identity.identifier_column:
        player_columns.add(context.player_identity.identifier_column)
    if request.group_by not in player_columns:
        return
    if context.player_identity.trust_level in {"not_provided", "untrusted_inferred"}:
        raise ValueError(
            "player grouping requires explicit trusted player identity; "
            "session, club, row order, and file layout are not identities"
        )
    if request.group_by != context.player_identity.identifier_column:
        raise ValueError("group_by must match the declared player identifier_column")


def _metric_units(metric: str, context: AnalysisContextV2) -> MetricUnitsV2:
    if metric in METRICS:
        definition = METRICS[metric]
        return MetricUnitsV2(
            canonical_unit=definition.canonical_unit,
            display_unit=definition.display_unit,
            authority="canonical_registry",
        )
    source_unit = context.source_units.get(metric)
    if source_unit:
        return MetricUnitsV2(
            canonical_unit=source_unit,
            display_unit=source_unit,
            authority="source_declared",
        )
    return MetricUnitsV2(
        canonical_unit="unknown",
        display_unit="unknown",
        authority="unknown",
    )


def analysis_lineage_v2(
    frame: pd.DataFrame,
    context: AnalysisContextV2,
    selected_columns: tuple[str, ...],
) -> AnalysisLineageV2:
    """Build reusable v2 lineage from selected columns and exact input rows.

    The dataset fingerprint binds the ordered backing-record hashes and selected
    columns. Every backing row either joins a declared source or carries the
    existing explicit unlinked reason.
    """

    if not selected_columns or any(not column for column in selected_columns):
        raise ValueError("selected_columns must contain non-empty names")
    missing = sorted(set(selected_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Columns not present in dataset: {', '.join(missing)}")
    records = _backing_records(frame, context)
    fingerprint_payload = {
        "selected_columns": selected_columns,
        "record_sha256": [record.record_sha256 for record in records],
    }
    serialized = json.dumps(fingerprint_payload, separators=(",", ":"))
    return AnalysisLineageV2(
        dataset_fingerprint_sha256=sha256(serialized.encode("utf-8")).hexdigest(),
        authority=context.authority,
        transformations=context.transformations,
        sources=context.sources,
        backing_records=records,
    )


def metric_units_v2(metric: str, context: AnalysisContextV2) -> MetricUnitsV2:
    """Resolve canonical or explicitly source-declared units for one metric."""

    return _metric_units(metric, context)


def vendor_provenance_v2(
    frame: pd.DataFrame, selected: tuple[str, ...]
) -> tuple[VendorProvenanceV2, ...]:
    """Collect bounded vendor/model provenance for reusable v2 analyses."""

    return _vendor_provenance(frame, selected)


def _uncertainty(request: FlexibleAnalysisRequest) -> UncertaintyV2:
    """Describe the uncertainty methods requested by the analysis contract."""

    correlation_requested = request.analysis_mode != "regression"
    regression_requested = request.analysis_mode != "correlation"
    return UncertaintyV2(
        confidence_level=request.confidence_level,
        correlation_interval=(
            "fisher-z"
            if correlation_requested and request.correlation_method == "pearson"
            else "unavailable"
            if correlation_requested
            else "not_requested"
        ),
        regression_interval="student-t" if regression_requested else "not_requested",
        multiplicity_adjustment=(
            "benjamini-hochberg" if correlation_requested else "not_requested"
        ),
        assumptions=(
            "Correlation does not establish causality.",
            "OLS intervals assume an adequate linear model and independent errors.",
            "Unmatched vendor samples are descriptive, not calibration evidence.",
        ),
    )


def analyze_variables_v2(
    frame: pd.DataFrame,
    request: FlexibleAnalysisRequest,
    *,
    context: AnalysisContextV2 | None = None,
    model_provenance: tuple[ModelProvenanceV2, ...] = (),
) -> LaunchMonitorAnalysisResultV2:
    """Analyze variables and return the canonical evidence-bearing v2 envelope.

    The input frame is not mutated. Player grouping fails closed unless the
    caller supplies explicit identity evidence; no file or row convention is
    accepted as identity.
    """

    resolved_context = context or AnalysisContextV2()
    _assert_identity_safe(request, resolved_context)
    analysis_payload: dict[str, Any] | None
    try:
        result = analyze_variables(frame, request)
        missingness = _missingness(frame, request)
        availability = _availability(result, request)
        analysis_payload = result.to_dict()
    except ValueError as error:
        unavailable_messages = {
            "Too few complete observations for regression": (
                "insufficient_complete_rows",
                max(request.min_samples, len(request.predictors) + 3),
            ),
            "Regression design matrix is rank deficient": (
                "rank_deficient_design",
                len(request.predictors) + 1,
            ),
        }
        detail = unavailable_messages.get(str(error))
        if detail is None or request.analysis_mode == "correlation":
            raise
        missingness = _missingness(frame, request)
        fallback_request = replace(request, analysis_mode="correlation")
        result = analyze_variables(frame, fallback_request)
        correlation_availability = (
            _availability(result, fallback_request)
            if request.analysis_mode == "comprehensive"
            else ()
        )
        reason_code, required_count = detail
        availability = (
            *correlation_availability,
            AvailabilityV2(
                result_path="regression",
                state="unavailable",
                reason_code=reason_code,
                message=str(error),
                observed_count=missingness.complete_row_count,
                required_count=required_count,
            ),
        )
        if request.analysis_mode == "comprehensive":
            analysis_payload = result.to_dict()
            analysis_payload["request"] = asdict(request)
        else:
            analysis_payload = None
    selected = (request.outcome, *request.predictors)
    units = {metric: _metric_units(metric, resolved_context) for metric in selected}
    return LaunchMonitorAnalysisResultV2(
        status=_overall_status(availability),
        analysis=analysis_payload,
        units=units,
        lineage=AnalysisLineageV2(
            dataset_fingerprint_sha256=result.dataset.fingerprint_sha256,
            authority=resolved_context.authority,
            transformations=resolved_context.transformations,
            sources=resolved_context.sources,
            backing_records=_backing_records(frame, resolved_context),
        ),
        missingness=missingness,
        availability=availability,
        uncertainty=_uncertainty(request),
        player_identity=resolved_context.player_identity,
        session_identity=resolved_context.session_identity,
        order_evidence=resolved_context.order_evidence,
        vendor_provenance=_vendor_provenance(frame, selected),
        model_provenance=model_provenance,
        warnings=result.warnings,
    )


def contract_v2_json_schema() -> dict[str, Any]:
    """Return the canonical JSON Schema used by OpenAPI and static clients."""

    return LaunchMonitorAnalysisResultV2.model_json_schema()


def adapt_v2_to_v1(result: LaunchMonitorAnalysisResultV2) -> dict[str, Any]:
    """Return the embedded legacy payload for clients pinned to contract v1."""

    if result.analysis is None:
        raise ValueError("No v1 analysis is available for this v2 result")
    payload = dict(result.analysis)
    payload["contract_version"] = CONTRACT_VERSION
    return payload


__all__ = [
    "CONTRACT_VERSION_V2",
    "AnalysisContextV2",
    "AnalysisLineageV2",
    "AvailabilityV2",
    "BackingRecordV2",
    "ClaimsV2",
    "DatasetAuthorityV2",
    "LaunchMonitorAnalysisResultV2",
    "MetricUnitsV2",
    "ModelProvenanceV2",
    "OrderEvidenceV2",
    "PlayerIdentityV2",
    "SessionIdentityV2",
    "SourceFileReferenceV2",
    "TransformRecordV2",
    "UncertaintyV2",
    "VendorProvenanceV2",
    "adapt_v2_to_v1",
    "analysis_lineage_v2",
    "analyze_variables_v2",
    "contract_v2_json_schema",
    "metric_units_v2",
    "vendor_provenance_v2",
]
