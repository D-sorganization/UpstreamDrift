"""Traceable launch-monitor statistical analysis routes."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.api.middleware.error_handler import handle_api_errors
from src.shared.python.launch_monitor import (
    CONTRACT_VERSION,
    AnalysisMode,
    CorrelationMethod,
    FlexibleAnalysisRequest,
    MissingPolicy,
    analyze_variables,
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


@router.get("/capabilities")
@handle_api_errors
async def capabilities() -> dict[str, object]:
    """Describe the stable contract consumed by desktop and web clients."""

    return {
        "contract_version": CONTRACT_VERSION,
        "analysis_modes": ["correlation", "regression", "comprehensive"],
        "correlation_methods": ["pearson", "spearman", "kendall"],
        "missing_policies": ["pairwise", "listwise", "fail"],
        "aggregate_regression_allowed": False,
        "maximum_inline_records": 20_000,
    }


@router.post("/analyze")
@handle_api_errors
async def analyze(payload: AnalyzePayload) -> dict[str, object]:
    """Analyze caller-supplied records without filesystem or URL access."""

    frame = pd.DataFrame.from_records(payload.records)
    result = analyze_variables(frame, payload.analysis.to_domain())
    return {"contract_version": CONTRACT_VERSION, "result": result.to_dict()}


__all__ = ["CONTRACT_VERSION", "router"]
