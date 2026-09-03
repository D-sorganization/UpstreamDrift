"""Wire models for governed launch-monitor scoring analyses."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from hashlib import sha256
from math import isfinite
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.tools.launch_monitor_model.contract_v2 import AnalysisContextV2

BASELINE_CONTRACT_VERSION: Literal["launch-monitor-strokes-gained-baseline/2.0.0"] = (
    "launch-monitor-strokes-gained-baseline/2.0.0"
)
STROKES_GAINED_CONTRACT_VERSION: Literal[
    "launch-monitor-strokes-gained-analysis/1.0.0"
] = "launch-monitor-strokes-gained-analysis/1.0.0"
OUTCOME_PROXY_CONTRACT_VERSION: Literal["launch-monitor-outcome-proxy/1.0.0"] = (
    "launch-monitor-outcome-proxy/1.0.0"
)

TrustedGrouping = Literal[
    "explicit_user_attested",
    "pseudonymous_stable",
    "verified_external",
]
ResultStatus = Literal["available", "partial", "unavailable"]
DistanceUnit = Literal["yd", "m"]


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExpectedStrokesStateV2(_ContractModel):
    """One benchmark point for an explicit target-aware course state."""

    lie: str = Field(min_length=1)
    context: str = Field(min_length=1)
    target: str = Field(min_length=1)
    distance_yards: float = Field(ge=0.0)
    expected_strokes: float = Field(ge=0.0)
    standard_error: float | None = Field(default=None, ge=0.0)

    @field_validator("lie", "context", "target")
    @classmethod
    def normalize_dimension(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("course-state dimensions must be non-empty")
        return normalized

    @model_validator(mode="after")
    def require_finite_values(self) -> ExpectedStrokesStateV2:
        values = [self.distance_yards, self.expected_strokes]
        if self.standard_error is not None:
            values.append(self.standard_error)
        if not all(isfinite(value) for value in values):
            raise ValueError("benchmark values must be finite")
        return self


class ExpectedStrokesBaselineV2(_ContractModel):
    """Hash-verified expected-strokes benchmark and publication metadata."""

    contract_version: Literal["launch-monitor-strokes-gained-baseline/2.0.0"] = (
        BASELINE_CONTRACT_VERSION
    )
    baseline_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    states: tuple[ExpectedStrokesStateV2, ...] = Field(min_length=2)

    @field_validator("baseline_id", "version", "license")
    @classmethod
    def strip_metadata(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("baseline metadata must be non-empty")
        return normalized

    @field_validator("source_url")
    @classmethod
    def require_http_source(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source_url must be HTTP(S) with a host")
        return value

    @model_validator(mode="after")
    def verify_table(self) -> ExpectedStrokesBaselineV2:
        identities = [
            (state.lie, state.context, state.target, state.distance_yards)
            for state in self.states
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("baseline contains duplicate course states")
        if baseline_table_sha256(self.states) != self.table_sha256:
            raise ValueError("baseline table_sha256 does not match canonical states")
        return self


class CourseStateColumnsV1(_ContractModel):
    lie_column: str = Field(min_length=1)
    context_column: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    distance_column: str = Field(min_length=1)
    distance_unit: DistanceUnit


class GroupingDimensionV1(_ContractModel):
    dimension: Literal["player", "session", "club"]
    column: str = Field(min_length=1)
    trust_level: TrustedGrouping
    evidence: str = Field(min_length=1)

    @field_validator("column", "evidence")
    @classmethod
    def strip_group_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("grouping column and evidence must be non-empty")
        return normalized


class LongitudinalDimensionV1(_ContractModel):
    order_column: str = Field(min_length=1)
    order_unit: str = Field(min_length=1)
    group_column: str | None = None
    group_dimension: Literal["player", "session", "club"] | None = None
    trust_level: TrustedGrouping
    evidence: str = Field(min_length=1)
    min_samples: int = Field(default=3, ge=3)

    @field_validator("order_column", "order_unit", "evidence")
    @classmethod
    def strip_longitudinal_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("longitudinal fields and evidence must be non-empty")
        return normalized

    @model_validator(mode="after")
    def require_complete_group_mapping(self) -> LongitudinalDimensionV1:
        if (self.group_column is None) != (self.group_dimension is None):
            raise ValueError("longitudinal group column and dimension must be paired")
        return self


class StrokesGainedRequestV1(_ContractModel):
    start: CourseStateColumnsV1
    finish: CourseStateColumnsV1
    shot_id_column: str | None = None
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    min_samples: int = Field(default=3, ge=1)
    summaries: tuple[GroupingDimensionV1, ...] = ()
    longitudinal: LongitudinalDimensionV1 | None = None

    @model_validator(mode="after")
    def require_unique_summary_dimensions(self) -> StrokesGainedRequestV1:
        dimensions = [summary.dimension for summary in self.summaries]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("summary dimensions must be unique")
        return self


class CourseStateValueV1(_ContractModel):
    lie: str
    context: str
    target: str
    distance_yards: float = Field(ge=0.0)


class InterpolationV1(_ContractModel):
    lower_distance_yards: float
    upper_distance_yards: float
    fraction: float = Field(ge=0.0, le=1.0)


class StrokesGainedRowV1(_ContractModel):
    source_index: int = Field(ge=0)
    shot_id: str | None = None
    input_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start: CourseStateValueV1
    finish: CourseStateValueV1
    expected_start: float
    expected_finish: float
    benchmark_standard_error: float | None = Field(default=None, ge=0.0)
    strokes_gained: float
    start_interpolation: InterpolationV1
    finish_interpolation: InterpolationV1
    groups: dict[str, str] = Field(default_factory=dict)
    longitudinal_order: float | None = None


class ExcludedRowV1(_ContractModel):
    source_index: int = Field(ge=0)
    shot_id: str | None = None
    reason_code: Literal[
        "missing_course_state",
        "invalid_distance",
        "outside_baseline",
    ]
    message: str


class ExclusionSummaryV1(_ContractModel):
    input_row_count: int = Field(ge=0)
    included_row_count: int = Field(ge=0)
    total_excluded: int = Field(ge=0)
    by_reason: dict[str, int]


class ConfidenceIntervalV1(_ContractModel):
    lower: float
    upper: float
    level: float
    method: str


class EstimateSummaryV1(_ContractModel):
    count: int = Field(ge=0)
    mean: float | None = None
    standard_deviation: float | None = Field(default=None, ge=0.0)
    standard_error: float | None = Field(default=None, ge=0.0)
    confidence_interval: ConfidenceIntervalV1 | None = None


class AvailabilityV1(_ContractModel):
    state: Literal["available", "unavailable"]
    reason_code: str | None = None
    message: str | None = None
    observed_count: int = Field(ge=0)
    required_count: int = Field(ge=0)


class StrokesGainedUncertaintyV1(_ContractModel):
    sampling_method: str
    confidence_level: float
    benchmark_method: str
    benchmark_standard_error_mean: float | None = Field(default=None, ge=0.0)
    assumptions: tuple[str, ...]


class GroupSummaryV1(_ContractModel):
    dimension: Literal["player", "session", "club"]
    group_value: str
    estimate: EstimateSummaryV1
    trust_level: TrustedGrouping
    evidence: str


class LongitudinalSummaryV1(_ContractModel):
    group_dimension: Literal["player", "session", "club", "all"]
    group_value: str
    sample_count: int = Field(ge=3)
    slope: float
    intercept: float
    r_squared: float = Field(ge=0.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    slope_unit: str
    trust_level: TrustedGrouping
    evidence: str


class BaselineProvenanceV1(_ContractModel):
    baseline_id: str
    version: str
    source_url: str
    license: str
    table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_version: str


class StrokesGainedClaimsV1(_ContractModel):
    is_strokes_gained: Literal[True] = True
    source_backed: Literal[True] = True
    device_emulation: Literal[False] = False
    device_certification: Literal[False] = False
    causal_inference: Literal[False] = False


class StrokesGainedAnalysisResultV1(_ContractModel):
    contract_version: Literal["launch-monitor-strokes-gained-analysis/1.0.0"] = (
        STROKES_GAINED_CONTRACT_VERSION
    )
    status: ResultStatus
    metric_name: Literal["source_backed_strokes_gained"] = (
        "source_backed_strokes_gained"
    )
    unit: Literal["strokes"] = "strokes"
    value_summary: EstimateSummaryV1
    baseline: BaselineProvenanceV1
    formula: str
    units: dict[str, str]
    availability: AvailabilityV1
    uncertainty: StrokesGainedUncertaintyV1
    row_results: tuple[StrokesGainedRowV1, ...]
    excluded_rows: tuple[ExcludedRowV1, ...]
    exclusions: ExclusionSummaryV1
    group_summaries: tuple[GroupSummaryV1, ...] = ()
    longitudinal_summaries: tuple[LongitudinalSummaryV1, ...] = ()
    analysis_context: AnalysisContextV2
    dataset_fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claims: StrokesGainedClaimsV1 = Field(default_factory=StrokesGainedClaimsV1)
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...]


class OutcomeProxyRequestV1(_ContractModel):
    carry_column: str = Field(min_length=1)
    lateral_column: str = Field(min_length=1)
    carry_unit: DistanceUnit
    lateral_unit: DistanceUnit
    target_distance_yards: float = Field(gt=0.0)
    shot_id_column: str | None = None
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    min_samples: int = Field(default=1, ge=1)


class OutcomeProxyRowV1(_ContractModel):
    source_index: int = Field(ge=0)
    shot_id: str | None = None
    carry_yards: float
    lateral_yards: float
    target_distance_yards: float
    radial_error_yards: float = Field(ge=0.0)


class OutcomeProxyClaimsV1(_ContractModel):
    is_strokes_gained: Literal[False] = False
    source_backed: Literal[False] = False
    causal_inference: Literal[False] = False


class OutcomeProxyResultV1(_ContractModel):
    contract_version: Literal["launch-monitor-outcome-proxy/1.0.0"] = (
        OUTCOME_PROXY_CONTRACT_VERSION
    )
    status: ResultStatus
    metric_name: Literal["expected_proximity_dispersion_proxy"] = (
        "expected_proximity_dispersion_proxy"
    )
    unit: Literal["yd"] = "yd"
    value_summary: EstimateSummaryV1
    row_results: tuple[OutcomeProxyRowV1, ...]
    exclusions: ExclusionSummaryV1
    formula: str
    units: dict[str, str]
    claims: OutcomeProxyClaimsV1 = Field(default_factory=OutcomeProxyClaimsV1)
    limitations: tuple[str, ...]


def _canonical_number(value: float | int) -> str:
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError("baseline numbers must be finite")
    normalized = f"{numeric:.12f}".rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def _coerce_state(value: ExpectedStrokesStateV2 | Mapping[str, Any]) -> dict[str, Any]:
    state = (
        value
        if isinstance(value, ExpectedStrokesStateV2)
        else ExpectedStrokesStateV2.model_validate(value)
    )
    return {
        "context": state.context,
        "distance_yards": _canonical_number(state.distance_yards),
        "expected_strokes": _canonical_number(state.expected_strokes),
        "lie": state.lie,
        "standard_error": (
            None
            if state.standard_error is None
            else _canonical_number(state.standard_error)
        ),
        "target": state.target,
    }


def _state_sort_key(state: Mapping[str, Any]) -> tuple[str, str, str, float]:
    return (
        str(state["lie"]),
        str(state["context"]),
        str(state["target"]),
        float(state["distance_yards"]),
    )


def baseline_table_sha256(
    states: Iterable[ExpectedStrokesStateV2 | Mapping[str, Any]],
) -> str:
    """Hash normalized states independent of JSON number spelling and row order."""

    canonical = [_coerce_state(state) for state in states]
    canonical.sort(key=_state_sort_key)
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


__all__ = [
    "BASELINE_CONTRACT_VERSION",
    "OUTCOME_PROXY_CONTRACT_VERSION",
    "STROKES_GAINED_CONTRACT_VERSION",
    "AvailabilityV1",
    "BaselineProvenanceV1",
    "ConfidenceIntervalV1",
    "CourseStateColumnsV1",
    "CourseStateValueV1",
    "EstimateSummaryV1",
    "ExcludedRowV1",
    "ExclusionSummaryV1",
    "ExpectedStrokesBaselineV2",
    "ExpectedStrokesStateV2",
    "GroupingDimensionV1",
    "GroupSummaryV1",
    "InterpolationV1",
    "LongitudinalDimensionV1",
    "LongitudinalSummaryV1",
    "OutcomeProxyRequestV1",
    "OutcomeProxyResultV1",
    "OutcomeProxyRowV1",
    "StrokesGainedAnalysisResultV1",
    "StrokesGainedRequestV1",
    "StrokesGainedRowV1",
    "StrokesGainedUncertaintyV1",
    "baseline_table_sha256",
]
