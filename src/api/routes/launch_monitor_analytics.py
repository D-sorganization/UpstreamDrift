"""Traceable launch-monitor statistical analysis routes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.api.middleware.error_handler import handle_api_errors
from src.shared.python.launch_monitor import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_V2,
    OUTCOME_PROXY_CONTRACT_VERSION,
    STROKES_GAINED_CONTRACT_VERSION,
    AnalysisContextV2,
    AnalysisMode,
    ExpectedStrokesBaselineV2,
    CorrelationMethod,
    FlexibleAnalysisRequest,
    LaunchMonitorAnalysisResultV2,
    ModelProvenanceV2,
    MissingPolicy,
    OutcomeProxyRequestV1,
    OutcomeProxyResultV1,
    StrokesGainedAnalysisResultV1,
    StrokesGainedRequestV1,
    analyze_outcome_proxy,
    analyze_source_backed_strokes_gained,
    analyze_variables,
    analyze_variables_v2,
    contract_v2_json_schema,
    strokes_gained_contract_json_schema,
)

router = APIRouter(
    prefix="/tools/launch-monitor-analytics", tags=["launch-monitor-analytics"]
)


class FlexibleAnalysisPayload(BaseModel):
    """Serialized form of :class:`FlexibleAnalysisRequest`."""

    outcome: str = Field(min_length=1)
    predictors: list[str] = Field(min_length=1)
    analysis_mode: AnalysisMode = "comprehensive"
    correlation_method: CorrelationMethod = "pearson"
    missing_policy: MissingPolicy = "pairwise"
    group_by: str | None = None
    confidence_level: float = Field(0.95, gt=0.5, lt=1.0)
    min_samples: int = Field(10, ge=3)
    allow_aggregate: bool = False

    def to_domain(self) -> FlexibleAnalysisRequest:
        return FlexibleAnalysisRequest(
            outcome=self.outcome,
            predictors=tuple(self.predictors),
            analysis_mode=self.analysis_mode,
            correlation_method=self.correlation_method,
            missing_policy=self.missing_policy,
            group_by=self.group_by,
            confidence_level=self.confidence_level,
            min_samples=self.min_samples,
            allow_aggregate=self.allow_aggregate,
        )


class AnalyzePayload(BaseModel):
    """Bounded inline data and an analysis request; paths are never accepted."""

    records: list[dict[str, Any]] = Field(min_length=3, max_length=20_000)
    analysis: FlexibleAnalysisPayload


class AnalyzePayloadV2(AnalyzePayload):
    """V2 request with explicit dataset, transformation, and identity context."""

    context: AnalysisContextV2 = Field(default_factory=AnalysisContextV2)
    model_provenance: tuple[ModelProvenanceV2, ...] = ()


class StrokesGainedPayloadV1(BaseModel):
    """Bounded records, verified benchmark, and governed SG request."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=20_000)
    baseline: ExpectedStrokesBaselineV2
    request: StrokesGainedRequestV1
    context: AnalysisContextV2 = Field(default_factory=AnalysisContextV2)


class OutcomeProxyPayloadV1(BaseModel):
    """Bounded records and explicitly non-SG outcome-proxy request."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=20_000)
    request: OutcomeProxyRequestV1


@router.get("/capabilities")
@handle_api_errors
async def capabilities() -> dict[str, object]:
    """Describe the stable contract consumed by desktop and web clients."""

    return {
        "contract_version": CONTRACT_VERSION,
        "supported_contract_versions": [CONTRACT_VERSION, CONTRACT_VERSION_V2],
        "analysis_modes": ["correlation", "regression", "comprehensive"],
        "correlation_methods": ["pearson", "spearman", "kendall"],
        "missing_policies": ["pairwise", "listwise", "fail"],
        "aggregate_regression_allowed": False,
        "maximum_inline_records": 20_000,
        "source_backed_scoring": True,
        "strokes_gained_contract_version": STROKES_GAINED_CONTRACT_VERSION,
        "outcome_proxy_contract_version": OUTCOME_PROXY_CONTRACT_VERSION,
        "outcome_proxy_is_strokes_gained": False,
    }


@router.get("/contracts/v2")
@handle_api_errors
async def contract_v2() -> dict[str, object]:
    """Publish the canonical JSON Schema used by OpenAPI v2 clients."""

    return contract_v2_json_schema()


@router.get("/contracts/strokes-gained/v1")
@handle_api_errors
async def strokes_gained_contract_v1() -> dict[str, object]:
    """Publish the canonical source-backed scoring result schema."""

    return strokes_gained_contract_json_schema()


@router.post("/analyze")
@handle_api_errors
async def analyze(payload: AnalyzePayload) -> dict[str, object]:
    """Analyze caller-supplied records without filesystem or URL access."""

    frame = pd.DataFrame.from_records(payload.records)
    result = analyze_variables(frame, payload.analysis.to_domain())
    return {"contract_version": CONTRACT_VERSION, "result": result.to_dict()}


@router.post(
    "/v2/analyze",
    response_model=LaunchMonitorAnalysisResultV2,
    response_model_exclude_none=True,
)
@handle_api_errors
async def analyze_v2(payload: AnalyzePayloadV2) -> LaunchMonitorAnalysisResultV2:
    """Analyze inline records with the evidence-bearing v2 contract."""

    frame = pd.DataFrame.from_records(payload.records)
    result = analyze_variables_v2(
        frame,
        payload.analysis.to_domain(),
        context=payload.context,
        model_provenance=payload.model_provenance,
    )
    return result


@router.post(
    "/v2/strokes-gained",
    response_model=StrokesGainedAnalysisResultV1,
    response_model_exclude_none=True,
)
@handle_api_errors
async def analyze_strokes_gained_v1(
    payload: StrokesGainedPayloadV1,
) -> StrokesGainedAnalysisResultV1:
    """Score explicit course states against a hash-verified benchmark."""

    return analyze_source_backed_strokes_gained(
        pd.DataFrame.from_records(payload.records),
        payload.baseline,
        payload.request,
        context=payload.context,
    )


@router.post(
    "/v2/outcome-proxy",
    response_model=OutcomeProxyResultV1,
    response_model_exclude_none=True,
)
@handle_api_errors
async def analyze_outcome_proxy_v1(
    payload: OutcomeProxyPayloadV1,
) -> OutcomeProxyResultV1:
    """Compute a proximity proxy whose contract forbids an SG claim."""

    return analyze_outcome_proxy(
        pd.DataFrame.from_records(payload.records), payload.request
    )


__all__ = ["CONTRACT_VERSION", "router"]
