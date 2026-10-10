"""Traceable launch-monitor statistical analysis routes."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

from src.api.middleware.error_handler import handle_api_errors
from src.api.services.launch_monitor_dataset_jobs import (
    DatasetJobCapacityError,
    DatasetJobResultPageV1,
    DatasetJobService,
    DatasetJobStatusV1,
    DatasetRootRegistry,
)
from src.tools.launch_monitor_model import (
    CONTRACT_VERSION,
    CONTRACT_VERSION_V2,
    LONGITUDINAL_SESSION_CONTRACT_VERSION,
    OUTCOME_PROXY_CONTRACT_VERSION,
    PLAYER_COVARIATION_CONTRACT_VERSION,
    STROKES_GAINED_CONTRACT_VERSION,
    AnalysisContextV2,
    AnalysisMode,
    ChangeCandidate,
    DispersionResult,
    ExpectedStrokesBaselineV2,
    CorrelationMethod,
    CorrelationResult,
    FlexibleAnalysisRequest,
    LaunchMonitorAnalysisResultV2,
    LongitudinalSessionRequestV1,
    LongitudinalSessionResultV1,
    ModelProvenanceV2,
    MissingPolicy,
    MonitorComparisonResult,
    OutcomeProxyRequestV1,
    OutcomeProxyResultV1,
    PCAResult,
    PlayerCovariationRequestV1,
    PlayerCovariationResultV1,
    PlayerCovariationScanRequestV1,
    PlayerCovariationScanResultV1,
    PredictiveModelResult,
    StrokesGainedAnalysisResultV1,
    StrokesGainedRequestV1,
    TemporalTrendResult,
    VIFResult,
    analyze_dispersion,
    analyze_longitudinal_sessions,
    analyze_outcome_proxy,
    analyze_player_covariation_v1,
    analyze_source_backed_strokes_gained,
    analyze_trend,
    analyze_variables,
    analyze_variables_v2,
    compare_monitors,
    compute_correlations,
    compute_pca,
    compute_vif,
    contract_v2_json_schema,
    fit_predictive_model,
    longitudinal_session_contract_json_schema,
    player_covariation_contract_json_schema,
    scan_player_covariation_v1,
    strokes_gained_contract_json_schema,
)
from shared.python.launch_monitor.dataset_reference import (
    MAX_PAGE_SIZE,
    DatasetJobRequestV1,
    dataset_job_contract_json_schema,
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


class PlayerCovariationPayloadV1(BaseModel):
    """Bounded records, selected pair, and explicit evidence context."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=20_000)
    request: PlayerCovariationRequestV1
    context: AnalysisContextV2 = Field(default_factory=AnalysisContextV2)


class PlayerCovariationScanPayloadV1(BaseModel):
    """Bounded records and a bounded exploratory pair-scan request."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=20_000)
    request: PlayerCovariationScanRequestV1
    context: AnalysisContextV2 = Field(default_factory=AnalysisContextV2)


class LongitudinalSessionPayloadV1(BaseModel):
    """Bounded rows plus attested identity, order, and session-unit design."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=20_000)
    request: LongitudinalSessionRequestV1
    context: AnalysisContextV2


class TrendPayloadV2(BaseModel):
    """Bounded inline records and the PyQt Trends tab's widget-derived inputs.

    Mirrors ``_TrendParams`` from ``src/tools/launch_monitor_analytics/gui.py``
    (``_read_trend_params``): ``rolling_window`` keeps the same default and
    the ``[3, 500]`` range as the Trends tab's spinbox
    (``_build_trends_tab``), so the API and desktop paths accept identical
    inputs for :func:`analyze_trend`.
    """

    records: list[dict[str, Any]] = Field(min_length=3, max_length=20_000)
    metric: str = Field(min_length=1)
    time_column: str = Field("captured_at", min_length=1)
    rolling_window: int = Field(10, ge=3, le=500)


class DispersionPayloadV2(BaseModel):
    """Bounded inline records and the PyQt Dispersion tab's widget-derived inputs.

    Mirrors ``_DispersionParams`` from
    ``src/tools/launch_monitor_analytics/gui.py`` (``_read_dispersion_params``):
    ``group_column`` is ``None`` when the Dispersion tab's "Group By" combo box
    reads "(all shots)" (``_build_dispersion_tab``), so the API and desktop
    paths accept identical inputs for :func:`analyze_dispersion`.
    """

    records: list[dict[str, Any]] = Field(min_length=3, max_length=20_000)
    forward: str = Field("carry_distance", min_length=1)
    lateral: str = Field("lateral_carry", min_length=1)
    group_column: Literal["monitor_vendor", "session_id", "club"] | None = None


class RelationshipsPayloadV2(BaseModel):
    """Bounded inline records and the PyQt Relationships tab's widget inputs.

    Mirrors ``_RelationshipParams`` from
    ``src/tools/launch_monitor_analytics/gui.py`` (``_read_relationship_params``):
    ``method`` offers the same three choices, ``edge_threshold`` keeps the same
    default and ``[0, 1]`` range as the "Network Edge Threshold" spinbox
    (``_build_relationships_tab``), and controls that are also selected metrics
    are dropped as the desktop tab drops them, so the API and desktop paths
    accept identical inputs for :func:`compute_correlations`.
    """

    records: list[dict[str, Any]] = Field(min_length=3, max_length=20_000)
    metrics: list[str] = Field(min_length=2)
    controls: list[str] = Field(default_factory=list)
    method: CorrelationMethod = "pearson"
    edge_threshold: float = Field(0.3, ge=0.0, le=1.0)

    @field_validator("metrics")
    @classmethod
    def _metrics_are_unique(cls, value: list[str]) -> list[str]:
        """Precondition: ``metrics`` names each column at most once."""
        if len(set(value)) != len(value):
            raise ValueError("metrics must not contain duplicates")
        return value

    def effective_controls(self) -> tuple[str, ...]:
        """Return ``controls`` without the selected metrics (desktop parity)."""
        return tuple(item for item in self.controls if item not in self.metrics)


class MultivariatePayloadV2(BaseModel):
    """Bounded inline records and the PyQt Relationships tab's PCA/VIF inputs.

    Mirrors ``_compute_multivariate`` from
    ``src/tools/launch_monitor_analytics/gui.py``: the same ``metrics``
    selection feeds both :func:`compute_pca` and :func:`compute_vif`, so the
    API and desktop paths accept identical inputs.
    """

    records: list[dict[str, Any]] = Field(min_length=3, max_length=20_000)
    metrics: list[str] = Field(min_length=2)


class ComparisonPayloadV2(BaseModel):
    """Bounded inline records and the PyQt Monitor Comparison tab's inputs.

    Mirrors ``_ComparisonParams`` from
    ``src/tools/launch_monitor_analytics/gui.py``
    (``_read_comparison_params``): ``match_column`` is ``None`` when the
    Monitor Comparison tab's "Matched-Shot Column" combo reads "(unmatched)"
    (``_build_comparison_tab``), and an empty ``reference_monitor`` (the
    combo's blank state, ``currentText() or None``) is treated the same as
    ``None``, so the API and desktop paths accept identical inputs for
    :func:`compare_monitors`.
    """

    records: list[dict[str, Any]] = Field(min_length=3, max_length=20_000)
    metric: str = Field(min_length=1)
    match_column: str | None = None
    reference_monitor: str | None = None

    @field_validator("reference_monitor")
    @classmethod
    def _blank_reference_is_none(cls, value: str | None) -> str | None:
        """Precondition: an empty reference behaves like an absent one (desktop parity)."""
        return value or None


ModelName = Literal["linear", "ridge", "lasso", "elastic_net", "mlp"]


class ModelPayloadV2(BaseModel):
    """Bounded inline records and the PyQt Models tab's widget-derived inputs.

    Mirrors ``_ModelParams`` from ``src/tools/launch_monitor_analytics/gui.py``
    (``_read_model_params``): ``model`` offers the same five choices,
    ``random_seed`` keeps the same default and ``[0, 2_147_483_647]`` range as
    the "Random Seed" spinbox, and ``group_column`` is ``None`` when the
    "Grouped Holdout" combo reads "(random split)" (``_build_models_tab``), so
    the API and desktop paths accept identical inputs for
    :func:`fit_predictive_model`.
    """

    records: list[dict[str, Any]] = Field(min_length=3, max_length=20_000)
    target: str = Field(min_length=1)
    features: list[str] = Field(min_length=1)
    model: ModelName = "linear"
    random_seed: int = Field(42, ge=0, le=2_147_483_647)
    group_column: Literal["session_id", "monitor_vendor", "club"] | None = None

    @field_validator("features")
    @classmethod
    def _features_are_unique(cls, value: list[str]) -> list[str]:
        """Precondition: ``features`` names each column at most once."""
        if len(set(value)) != len(value):
            raise ValueError("features must not contain duplicates")
        return value


@lru_cache(maxsize=1)
def get_launch_monitor_dataset_job_service() -> DatasetJobService:
    """Return the bounded process-local service for administrator roots."""
    return DatasetJobService(DatasetRootRegistry.from_environment())


@asynccontextmanager
async def launch_monitor_dataset_jobs_lifespan(
    _app: object,
) -> AsyncIterator[None]:
    """Join cached private-data workers during FastAPI application shutdown."""
    try:
        yield
    finally:
        if get_launch_monitor_dataset_job_service.cache_info().currsize:
            get_launch_monitor_dataset_job_service().close()
            get_launch_monitor_dataset_job_service.cache_clear()


router = APIRouter(
    prefix="/tools/launch-monitor-analytics",
    tags=["launch-monitor-analytics"],
    lifespan=launch_monitor_dataset_jobs_lifespan,
)


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
        "dataset_reference_jobs": True,
        "dataset_job_maximum_page_size": MAX_PAGE_SIZE,
        "dataset_job_inline_rows_allowed": False,
        "player_covariation_contract_version": (PLAYER_COVARIATION_CONTRACT_VERSION),
        "population_meta_analysis": True,
        "longitudinal_session_contract_version": (
            LONGITUDINAL_SESSION_CONTRACT_VERSION
        ),
        "longitudinal_primary_unit": "player_session_stratum",
        "longitudinal_causal_improvement": False,
    }


@router.get("/contracts/v2")
@handle_api_errors
async def contract_v2() -> dict[str, object]:
    """Publish the canonical JSON Schema used by OpenAPI v2 clients."""

    schema: dict[str, object] = contract_v2_json_schema()
    return schema


@router.get("/contracts/strokes-gained/v1")
@handle_api_errors
async def strokes_gained_contract_v1() -> dict[str, object]:
    """Publish the canonical source-backed scoring result schema."""

    schema: dict[str, object] = strokes_gained_contract_json_schema()
    return schema


@router.get("/contracts/dataset-jobs/v1")
@handle_api_errors
async def dataset_jobs_contract_v1() -> dict[str, object]:
    """Publish the immutable dataset-reference job request schema."""

    schema: dict[str, object] = dataset_job_contract_json_schema()
    return schema


@router.get("/contracts/longitudinal-sessions/v1")
@handle_api_errors
async def longitudinal_sessions_contract_v1() -> dict[str, object]:
    """Publish the attested session-unit longitudinal result schema."""

    schema: dict[str, object] = longitudinal_session_contract_json_schema()
    return schema


@router.post(
    "/v2/dataset-jobs",
    response_model=DatasetJobStatusV1,
    status_code=status.HTTP_202_ACCEPTED,
)
@handle_api_errors
async def create_dataset_job(
    payload: DatasetJobRequestV1,
    service: DatasetJobService = Depends(get_launch_monitor_dataset_job_service),
) -> DatasetJobStatusV1:
    """Queue an aggregate job by immutable reference, never inline records."""

    try:
        return service.submit(payload)
    except DatasetJobCapacityError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "code": "dataset_job_capacity_exhausted",
                "message": "Dataset job capacity is temporarily exhausted.",
                "retryable": True,
            },
            headers={"Retry-After": "5"},
        ) from exc


@router.get(
    "/v2/dataset-jobs/{job_id}",
    response_model=DatasetJobStatusV1,
)
@handle_api_errors
async def get_dataset_job(
    job_id: str,
    service: DatasetJobService = Depends(get_launch_monitor_dataset_job_service),
) -> DatasetJobStatusV1:
    """Return a data-free job status or a structured unavailable reason."""

    try:
        return service.status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dataset job not found") from exc


@router.get(
    "/v2/dataset-jobs/{job_id}/results",
    response_model=DatasetJobResultPageV1,
)
@handle_api_errors
async def get_dataset_job_results(
    job_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=MAX_PAGE_SIZE),
    service: DatasetJobService = Depends(get_launch_monitor_dataset_job_service),
) -> DatasetJobResultPageV1:
    """Return one bounded page of aggregate/source-backing results."""

    try:
        return service.results(job_id, offset=offset, limit=limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dataset job not found") from exc


@router.get("/contracts/player-covariation/v1")
@handle_api_errors
async def player_covariation_contract_v1() -> dict[str, object]:
    """Publish the canonical player/population covariation schema."""

    schema: dict[str, object] = player_covariation_contract_json_schema()
    return schema


@router.post("/analyze")
@handle_api_errors
async def analyze(payload: AnalyzePayload) -> dict[str, object]:
    """Analyze caller-supplied records without filesystem or URL access."""

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

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

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    frame = pd.DataFrame.from_records(payload.records)
    result = analyze_variables_v2(
        frame,
        payload.analysis.to_domain(),
        context=payload.context,
        model_provenance=payload.model_provenance,
    )
    return result


@router.post(
    "/v2/player-covariation",
    response_model=PlayerCovariationResultV1,
    response_model_exclude_none=True,
)
@handle_api_errors
async def analyze_player_covariation(
    payload: PlayerCovariationPayloadV1,
) -> PlayerCovariationResultV1:
    """Analyze one variable pair across explicitly identified players."""

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    return analyze_player_covariation_v1(
        pd.DataFrame.from_records(payload.records),
        payload.request,
        context=payload.context,
    )


@router.post(
    "/v2/player-covariation/scan",
    response_model=PlayerCovariationScanResultV1,
    response_model_exclude_none=True,
)
@handle_api_errors
async def scan_player_covariation(
    payload: PlayerCovariationScanPayloadV1,
) -> PlayerCovariationScanResultV1:
    """Rank a bounded exploratory set of variable pairs."""

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    return scan_player_covariation_v1(
        pd.DataFrame.from_records(payload.records),
        payload.request,
        context=payload.context,
    )


@router.post(
    "/v2/longitudinal-sessions",
    response_model=LongitudinalSessionResultV1,
    response_model_exclude_none=True,
)
@handle_api_errors
async def analyze_longitudinal_sessions_v1(
    payload: LongitudinalSessionPayloadV1,
) -> LongitudinalSessionResultV1:
    """Estimate descriptive direction after session-level aggregation."""

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    return analyze_longitudinal_sessions(
        pd.DataFrame.from_records(payload.records),
        payload.request,
        context=payload.context,
    )


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

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

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

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    return analyze_outcome_proxy(
        pd.DataFrame.from_records(payload.records), payload.request
    )


def _json_safe_float(value: float) -> float | None:
    """Return ``value``, or ``None`` when it is NaN/infinite.

    Postcondition: an unavailable statistic never serializes as ``0``.
    """
    return value if math.isfinite(value) else None


def _change_candidate_to_dict(candidate: ChangeCandidate) -> dict[str, object]:
    """Serialize one :class:`ChangeCandidate` to a JSON-safe dict."""
    return {
        "captured_at": candidate.captured_at.isoformat(),
        "row_index": candidate.row_index,
        "before_mean": _json_safe_float(candidate.before_mean),
        "after_mean": _json_safe_float(candidate.after_mean),
        "effect_size": _json_safe_float(candidate.effect_size),
    }


def _rolling_series_to_records(
    rolling: Any, time_column: str
) -> list[dict[str, object]]:
    """Serialize the rolling-statistics frame to JSON-safe row dicts."""
    records: list[dict[str, Any]] = rolling.to_dict(orient="records")
    for entry in records:
        entry[time_column] = entry[time_column].isoformat()
        for key, value in entry.items():
            if isinstance(value, float):
                entry[key] = _json_safe_float(value)
    return records


def _trend_result_to_dict(
    result: TemporalTrendResult, time_column: str
) -> dict[str, object]:
    """Serialize :class:`TemporalTrendResult` to the ``/v2/trend`` response.

    Postcondition: every numeric field is JSON-safe — NaN/infinite values
    become ``null``, never ``0`` — and ``rolling``/``change_candidates`` are
    plain JSON lists.
    """
    return {
        "metric": result.metric,
        "sample_count": result.sample_count,
        "slope_per_day": _json_safe_float(result.slope_per_day),
        "robust_slope_per_day": _json_safe_float(result.robust_slope_per_day),
        "p_value": _json_safe_float(result.p_value),
        "earliest_mean": _json_safe_float(result.earliest_mean),
        "latest_mean": _json_safe_float(result.latest_mean),
        "rolling": _rolling_series_to_records(result.rolling, time_column),
        "change_candidates": [
            _change_candidate_to_dict(candidate)
            for candidate in result.change_candidates
        ],
    }


@router.post("/v2/trend")
@handle_api_errors
async def analyze_trend_v2(payload: TrendPayloadV2) -> dict[str, object]:
    """Analyze a longitudinal metric trend with the PyQt Trends tab's inputs.

    Calls the same :func:`analyze_trend` the desktop Trends tab calls
    (``src/tools/launch_monitor_analytics/gui.py`` ``_compute_trend``), so the
    API and PyQt paths share one contract. Precondition: ``records`` holds
    3-20,000 inline rows and ``rolling_window`` is in ``[3, 500]`` (validated
    by the request schema). Postcondition: the response is a JSON-safe
    serialization of every :class:`TemporalTrendResult` field; see
    :func:`_trend_result_to_dict`.
    """

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    frame = pd.DataFrame.from_records(payload.records)
    result = analyze_trend(
        frame,
        metric=payload.metric,
        time_column=payload.time_column,
        rolling_window=payload.rolling_window,
    )
    return _trend_result_to_dict(result, payload.time_column)


def _dispersion_result_to_dict(
    name: str, result: DispersionResult
) -> dict[str, object]:
    """Serialize one group's :class:`DispersionResult` for ``/v2/dispersion``.

    Postcondition: every float field is JSON-safe — NaN/infinite values
    become ``null``, never ``0`` — and the result carries its group ``name``.
    """
    fields: dict[str, object] = dataclasses.asdict(result)
    for key, value in fields.items():
        if isinstance(value, float):
            fields[key] = _json_safe_float(value)
    return {"group": name, **fields}


@router.post("/v2/dispersion")
@handle_api_errors
async def analyze_dispersion_v2(payload: DispersionPayloadV2) -> dict[str, object]:
    """Analyze shot dispersion with the PyQt Dispersion tab's inputs.

    Calls the same :func:`analyze_dispersion` the desktop Dispersion tab
    calls (``src/tools/launch_monitor_analytics/gui.py``
    ``_compute_dispersion``), so the API and PyQt paths share one contract.
    Precondition: ``records`` holds 3-20,000 inline rows (validated by the
    request schema). Grouping mirrors ``_compute_dispersion``: a single
    "All Shots" group when ``group_column`` is ``None`` or the column is
    absent from the frame, otherwise one group per
    ``frame.groupby(group_column, dropna=False)`` value, named ``str(name)``.
    Postcondition: the response is a JSON-safe serialization of every
    :class:`DispersionResult` field per group; see
    :func:`_dispersion_result_to_dict`.
    """

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    frame = pd.DataFrame.from_records(payload.records)
    groups: list[tuple[object, pd.DataFrame]] = [("All Shots", frame)]
    if payload.group_column is not None and payload.group_column in frame:
        groups = list(frame.groupby(payload.group_column, dropna=False))

    return {
        "forward": payload.forward,
        "lateral": payload.lateral,
        "group_column": payload.group_column,
        "groups": [
            _dispersion_result_to_dict(
                str(name),
                analyze_dispersion(
                    group, forward=payload.forward, lateral=payload.lateral
                ),
            )
            for name, group in groups
        ],
    }


def _matrix_to_rows(matrix: Any) -> list[list[float | None]]:
    """Serialize a labelled matrix to JSON-safe rows in its index order.

    Postcondition: NaN/infinite cells become ``null``, never ``0``.
    """
    return [
        [_json_safe_float(float(value)) for value in row]
        for row in matrix.to_numpy(dtype=float)
    ]


def _relationships_result_to_dict(result: CorrelationResult) -> dict[str, object]:
    """Serialize :class:`CorrelationResult` to the ``/v2/relationships`` response.

    Postcondition: every matrix is a list of rows ordered like ``metrics``;
    an optional matrix the analysis did not produce is ``null``; every
    non-finite number is ``null``, never ``0``.
    """
    adjusted = result.adjusted_p_values
    partial = result.partial_coefficients
    return {
        "method": result.method,
        "metrics": [str(metric) for metric in result.coefficients.index],
        "coefficients": _matrix_to_rows(result.coefficients),
        "p_values": _matrix_to_rows(result.p_values),
        "adjusted_p_values": None if adjusted is None else _matrix_to_rows(adjusted),
        "pair_counts": result.pair_counts.to_numpy(dtype=int).tolist(),
        "partial_coefficients": None if partial is None else _matrix_to_rows(partial),
        "derived_metrics": list(result.derived_metrics),
        "boolean_projected": list(result.boolean_projected),
        "edges": [
            {
                key: _json_safe_float(value) if isinstance(value, float) else value
                for key, value in dataclasses.asdict(edge).items()
            }
            for edge in result.edges
        ],
    }


@router.post("/v2/relationships")
@handle_api_errors
async def analyze_relationships_v2(
    payload: RelationshipsPayloadV2,
) -> dict[str, object]:
    """Map metric interdependencies with the PyQt Relationships tab's inputs.

    Calls the same :func:`compute_correlations` the desktop Relationships tab
    calls (``src/tools/launch_monitor_analytics/gui.py``
    ``_compute_relationship``), so the API and PyQt paths share one contract.
    Precondition: ``records`` holds 3-20,000 inline rows, ``metrics`` names at
    least two columns and ``edge_threshold`` is in ``[0, 1]`` (validated by
    the request schema); a column absent from the records returns 400.
    Postcondition: the response is a JSON-safe serialization of every
    :class:`CorrelationResult` field; see :func:`_relationships_result_to_dict`.
    """

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    result = compute_correlations(
        pd.DataFrame.from_records(payload.records),
        metrics=tuple(payload.metrics),
        method=payload.method,
        controls=payload.effective_controls(),
        edge_threshold=payload.edge_threshold,
    )
    return _relationships_result_to_dict(result)


def _pca_result_to_dict(result: PCAResult) -> dict[str, object]:
    """Serialize :class:`PCAResult` to the ``/v2/multivariate`` response.

    Postcondition: ``loadings`` rows follow ``metrics`` order and
    ``scores`` rows follow the complete-case sample order, both with
    columns ordered like ``component_names``; every non-finite number is
    ``null``, never ``0``.
    """
    return {
        "metrics": list(result.metrics),
        "component_names": list(result.explained_variance_ratio.index),
        "explained_variance_ratio": [
            _json_safe_float(float(value))
            for value in result.explained_variance_ratio.to_numpy(dtype=float)
        ],
        "loadings": _matrix_to_rows(result.loadings),
        "scores": _matrix_to_rows(result.scores),
        "sample_count": result.sample_count,
    }


def _vif_result_to_dict(result: VIFResult) -> dict[str, object]:
    """Serialize :class:`VIFResult` to the ``/v2/multivariate`` response.

    Postcondition: ``values`` is a metric-keyed mapping; an infinite VIF
    (perfectly collinear metrics) serializes as ``null``, never ``0``.
    """
    return {
        "values": {
            str(metric): _json_safe_float(float(value))
            for metric, value in result.values.items()
        },
        "sample_count": result.sample_count,
        "warning_metrics": list(result.warning_metrics),
    }


@router.post("/v2/multivariate")
@handle_api_errors
async def analyze_multivariate_v2(payload: MultivariatePayloadV2) -> dict[str, object]:
    """Compute PCA and VIF diagnostics with the PyQt multivariate inputs.

    Calls the same :func:`compute_pca` and :func:`compute_vif` the desktop
    Relationships tab calls (``src/tools/launch_monitor_analytics/gui.py``
    ``_compute_multivariate``), so the API and PyQt paths share one
    contract. Precondition: ``records`` holds 3-20,000 inline rows and
    ``metrics`` names at least two columns (validated by the request
    schema); an unknown metric or too few complete rows returns 400.
    Postcondition: the response is a JSON-safe serialization of every
    :class:`PCAResult` and :class:`VIFResult` field; see
    :func:`_pca_result_to_dict` and :func:`_vif_result_to_dict`.
    """

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    frame = pd.DataFrame.from_records(payload.records)
    metrics = tuple(payload.metrics)
    pca = compute_pca(frame, metrics=metrics)
    vif = compute_vif(frame, metrics=metrics)
    return {
        "pca": _pca_result_to_dict(pca),
        "vif": _vif_result_to_dict(vif),
    }


def _dataclass_to_json_safe_dict(instance: Any) -> dict[str, object]:
    """Serialize one frozen dataclass instance to a JSON-safe dict.

    Postcondition: every float field is JSON-safe — NaN/infinite values
    become ``null``, never ``0``; every other field is unchanged.
    """
    return {
        key: _json_safe_float(value) if isinstance(value, float) else value
        for key, value in dataclasses.asdict(instance).items()
    }


def _comparison_result_to_dict(result: MonitorComparisonResult) -> dict[str, object]:
    """Serialize :class:`MonitorComparisonResult` to the ``/v2/comparison`` response.

    Postcondition: every float field of each summary and pairwise entry is
    JSON-safe — NaN/infinite values become ``null``, never ``0``.
    """
    return {
        "metric": result.metric,
        "summaries": [_dataclass_to_json_safe_dict(item) for item in result.summaries],
        "pairwise": [_dataclass_to_json_safe_dict(item) for item in result.pairwise],
    }


@router.post("/v2/comparison")
@handle_api_errors
async def analyze_comparison_v2(payload: ComparisonPayloadV2) -> dict[str, object]:
    """Compare monitor behavior with the PyQt Monitor Comparison tab's inputs.

    Calls the same :func:`compare_monitors` the desktop Monitor Comparison
    tab calls (``src/tools/launch_monitor_analytics/gui.py``
    ``_compute_comparison``), so the API and PyQt paths share one contract.
    Precondition: ``records`` holds 3-20,000 inline rows and ``metric`` names
    a non-empty column (validated by the request schema); fewer than two
    monitors, an unknown reference monitor, or fewer than three matched pairs
    is a domain ``ValueError`` -> 400. Postcondition: the response is a
    JSON-safe serialization of every :class:`MonitorComparisonResult` field;
    see :func:`_comparison_result_to_dict`.
    """

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    result = compare_monitors(
        pd.DataFrame.from_records(payload.records),
        metric=payload.metric,
        match_column=payload.match_column,
        reference_monitor=payload.reference_monitor,
    )
    return {
        "match_column": payload.match_column,
        "reference_monitor": payload.reference_monitor,
        **_comparison_result_to_dict(result),
    }


def _predictions_to_records(predictions: Any) -> list[dict[str, object]]:
    """Serialize the held-out predictions frame to JSON-safe row dicts.

    Postcondition: every float value is JSON-safe — NaN/infinite values
    become ``null``, never ``0``.
    """
    records: list[dict[str, Any]] = predictions.to_dict(orient="records")
    for entry in records:
        for key, value in entry.items():
            if isinstance(value, float):
                entry[key] = _json_safe_float(value)
    return records


def _model_result_to_dict(result: PredictiveModelResult) -> dict[str, object]:
    """Serialize :class:`PredictiveModelResult` to the ``/v2/model`` response.

    Postcondition: every float in ``metrics``/``coefficients`` is JSON-safe
    — NaN/infinite values become ``null``, never ``0`` — and
    ``predictions`` is a JSON-safe list of row dicts; see
    :func:`_predictions_to_records`.
    """
    coefficients = result.coefficients
    return {
        "model": result.model,
        "target": result.target,
        "features": list(result.features),
        "metrics": {
            key: _json_safe_float(value) for key, value in result.metrics.items()
        },
        "coefficients": (
            None
            if coefficients is None
            else {key: _json_safe_float(value) for key, value in coefficients.items()}
        ),
        "random_seed": result.random_seed,
        "train_count": result.train_count,
        "test_count": result.test_count,
        "predictions": _predictions_to_records(result.predictions),
    }


@router.post("/v2/model")
@handle_api_errors
async def fit_model_v2(payload: ModelPayloadV2) -> dict[str, object]:
    """Fit a predictive model with the PyQt Models tab's widget-derived inputs.

    Calls the same :func:`fit_predictive_model` the desktop Models tab calls
    (``src/tools/launch_monitor_analytics/gui.py`` ``_compute_model``), so the
    API and PyQt paths share one contract. Precondition: ``records`` holds
    3-20,000 inline rows, ``features`` names 1+ distinct columns, and
    ``random_seed`` is in ``[0, 2_147_483_647]`` (validated by the request
    schema); target leakage, an unknown column, or insufficient complete rows
    is a domain ``ValueError`` -> 400. A model whose optional dependency is
    missing (``ImportError``, e.g. ``mlp`` without scikit-learn) is reported
    as unavailable -> 503, the same failure the desktop tab shows.
    Postcondition: the response is a JSON-safe serialization of every
    :class:`PredictiveModelResult` field; see :func:`_model_result_to_dict`.
    """

    import pandas as pd  # deferred import: pandas must not load at API boot (issue #8943)

    try:
        result = fit_predictive_model(
            pd.DataFrame.from_records(payload.records),
            target=payload.target,
            features=payload.features,
            model=payload.model,
            random_seed=payload.random_seed,
            group_column=payload.group_column,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model {payload.model!r} is unavailable: {exc}",
        ) from exc
    return _model_result_to_dict(result)


__all__ = ["CONTRACT_VERSION", "router"]
