"""Canonical source-backed strokes-gained analysis."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, sqrt
from typing import Literal

import pandas as pd

from src.shared.python.launch_monitor._scoring_statistics import (
    estimate_summary,
    group_summaries,
    longitudinal_summaries,
)
from src.shared.python.launch_monitor.contract_v2 import (
    AnalysisContextV2,
    _record_digest,
)
from src.shared.python.launch_monitor.strokes_gained_types import (
    AvailabilityV1,
    BaselineProvenanceV1,
    CourseStateColumnsV1,
    CourseStateValueV1,
    EstimateSummaryV1,
    ExcludedRowV1,
    ExclusionSummaryV1,
    ExpectedStrokesBaselineV2,
    ExpectedStrokesStateV2,
    InterpolationV1,
    StrokesGainedAnalysisResultV1,
    StrokesGainedRequestV1,
    StrokesGainedRowV1,
    StrokesGainedUncertaintyV1,
)

YARDS_PER_METRE = 1.0936132983377078


@dataclass(frozen=True)
class _Lookup:
    expected: float
    standard_error: float | None
    interpolation: InterpolationV1


class _RowIssue(ValueError):
    def __init__(
        self,
        reason_code: Literal[
            "missing_course_state", "invalid_distance", "outside_baseline"
        ],
        message: str,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _required_columns(request: StrokesGainedRequestV1) -> set[str]:
    columns = {
        request.start.lie_column,
        request.start.context_column,
        request.start.target_column,
        request.start.distance_column,
        request.finish.lie_column,
        request.finish.context_column,
        request.finish.target_column,
        request.finish.distance_column,
    }
    columns.update(summary.column for summary in request.summaries)
    if request.shot_id_column:
        columns.add(request.shot_id_column)
    if request.longitudinal:
        columns.add(request.longitudinal.order_column)
        if request.longitudinal.group_column:
            columns.add(request.longitudinal.group_column)
    return columns


def _validate_columns(frame: pd.DataFrame, request: StrokesGainedRequestV1) -> None:
    missing = sorted(_required_columns(request).difference(frame.columns))
    if missing:
        raise ValueError(f"Columns not present in launch-monitor records: {missing}")


def _text(value: object, label: str) -> str:
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    normalized = "" if missing else str(value).strip().lower()
    if not normalized:
        raise _RowIssue("missing_course_state", f"{label} is missing")
    return normalized


def _yards(value: object, unit: str, label: str) -> float:
    if isinstance(value, bool):
        raise _RowIssue("invalid_distance", f"{label} must be numeric")
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise _RowIssue("missing_course_state", f"{label} is missing")
    distance = float(numeric)
    if not isfinite(distance) or distance < 0:
        raise _RowIssue("invalid_distance", f"{label} must be finite and nonnegative")
    return distance * (YARDS_PER_METRE if unit == "m" else 1.0)


def _course_state(
    row: pd.Series,
    columns: CourseStateColumnsV1,
    label: str,
) -> CourseStateValueV1:
    return CourseStateValueV1(
        lie=_text(row[columns.lie_column], f"{label} lie"),
        context=_text(row[columns.context_column], f"{label} context"),
        target=_text(row[columns.target_column], f"{label} target/hole"),
        distance_yards=_yards(
            row[columns.distance_column], columns.distance_unit, f"{label} distance"
        ),
    )


def _interpolated_error(
    lower: ExpectedStrokesStateV2,
    upper: ExpectedStrokesStateV2,
    fraction: float,
) -> float | None:
    if lower.standard_error is None or upper.standard_error is None:
        return None
    return lower.standard_error + fraction * (
        upper.standard_error - lower.standard_error
    )


def _lookup(
    baseline: ExpectedStrokesBaselineV2,
    state: CourseStateValueV1,
) -> _Lookup:
    matches = sorted(
        (
            point
            for point in baseline.states
            if (
                point.lie,
                point.context,
                point.target,
            )
            == (state.lie, state.context, state.target)
        ),
        key=lambda point: point.distance_yards,
    )
    if (
        not matches
        or state.distance_yards < matches[0].distance_yards
        or state.distance_yards > matches[-1].distance_yards
    ):
        raise _RowIssue(
            "outside_baseline",
            "course state is absent from or outside the benchmark range",
        )
    upper_index = next(
        index
        for index, point in enumerate(matches)
        if point.distance_yards >= state.distance_yards
    )
    upper = matches[upper_index]
    lower = upper if upper_index == 0 else matches[upper_index - 1]
    span = upper.distance_yards - lower.distance_yards
    fraction = (
        0.0 if span == 0 else (state.distance_yards - lower.distance_yards) / span
    )
    expected = lower.expected_strokes + fraction * (
        upper.expected_strokes - lower.expected_strokes
    )
    return _Lookup(
        expected=float(expected),
        standard_error=_interpolated_error(lower, upper, fraction),
        interpolation=InterpolationV1(
            lower_distance_yards=lower.distance_yards,
            upper_distance_yards=upper.distance_yards,
            fraction=float(fraction),
        ),
    )


def _optional_id(row: pd.Series, column: str | None) -> str | None:
    if not column:
        return None
    value = row[column]
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    normalized = str(value).strip()
    return normalized or None


def _groups(row: pd.Series, request: StrokesGainedRequestV1) -> dict[str, str]:
    output: dict[str, str] = {}
    for summary in request.summaries:
        value = _optional_id(row, summary.column)
        if value:
            output[summary.dimension] = value
    longitudinal = request.longitudinal
    if longitudinal and longitudinal.group_column and longitudinal.group_dimension:
        value = _optional_id(row, longitudinal.group_column)
        if value:
            output[longitudinal.group_dimension] = value
    return output


def _order(row: pd.Series, request: StrokesGainedRequestV1) -> float | None:
    if request.longitudinal is None:
        return None
    numeric = pd.to_numeric(
        pd.Series([row[request.longitudinal.order_column]]), errors="coerce"
    ).iloc[0]
    if pd.isna(numeric) or not isfinite(float(numeric)):
        return None
    return float(numeric)


def _row_result(
    source_index: int,
    raw_row: pd.Series,
    baseline: ExpectedStrokesBaselineV2,
    request: StrokesGainedRequestV1,
) -> StrokesGainedRowV1:
    start = _course_state(raw_row, request.start, "start")
    finish = _course_state(raw_row, request.finish, "finish")
    expected_start = _lookup(baseline, start)
    expected_finish = _lookup(baseline, finish)
    benchmark_error = None
    if (
        expected_start.standard_error is not None
        and expected_finish.standard_error is not None
    ):
        benchmark_error = sqrt(
            expected_start.standard_error**2 + expected_finish.standard_error**2
        )
    raw = {str(key): value for key, value in raw_row.to_dict().items()}
    return StrokesGainedRowV1(
        source_index=source_index,
        shot_id=_optional_id(raw_row, request.shot_id_column),
        input_record_sha256=_record_digest(raw),
        start=start,
        finish=finish,
        expected_start=expected_start.expected,
        expected_finish=expected_finish.expected,
        benchmark_standard_error=benchmark_error,
        strokes_gained=expected_start.expected - 1.0 - expected_finish.expected,
        start_interpolation=expected_start.interpolation,
        finish_interpolation=expected_finish.interpolation,
        groups=_groups(raw_row, request),
        longitudinal_order=_order(raw_row, request),
    )


def _analyze_rows(
    frame: pd.DataFrame,
    baseline: ExpectedStrokesBaselineV2,
    request: StrokesGainedRequestV1,
) -> tuple[tuple[StrokesGainedRowV1, ...], tuple[ExcludedRowV1, ...]]:
    included: list[StrokesGainedRowV1] = []
    excluded: list[ExcludedRowV1] = []
    for source_index, (_, raw_row) in enumerate(frame.iterrows()):
        try:
            included.append(_row_result(source_index, raw_row, baseline, request))
        except _RowIssue as error:
            excluded.append(
                ExcludedRowV1(
                    source_index=source_index,
                    shot_id=_optional_id(raw_row, request.shot_id_column),
                    reason_code=error.reason_code,
                    message=str(error),
                )
            )
    return tuple(included), tuple(excluded)


def _availability(
    count: int, required: int
) -> tuple[Literal["available", "unavailable"], AvailabilityV1]:
    if count >= required:
        return "available", AvailabilityV1(
            state="available", observed_count=count, required_count=required
        )
    return "unavailable", AvailabilityV1(
        state="unavailable",
        reason_code="insufficient_complete_rows",
        message="Too few complete, benchmark-covered course-state rows.",
        observed_count=count,
        required_count=required,
    )


def _dataset_fingerprint(frame: pd.DataFrame) -> str:
    digests = [
        _record_digest({str(key): value for key, value in row.to_dict().items()})
        for _, row in frame.iterrows()
    ]
    return sha256("\n".join(digests).encode("ascii")).hexdigest()


def _uncertainty(
    rows: tuple[StrokesGainedRowV1, ...], level: float
) -> StrokesGainedUncertaintyV1:
    errors = [row.benchmark_standard_error for row in rows]
    complete = [value for value in errors if value is not None]
    benchmark_mean_error = (
        sqrt(sum(value**2 for value in complete)) / len(complete)
        if complete and len(complete) == len(rows)
        else None
    )
    return StrokesGainedUncertaintyV1(
        sampling_method="student-t-descriptive-mean",
        confidence_level=level,
        benchmark_method=(
            "interpolated-state-standard-errors"
            if benchmark_mean_error is not None
            else "unavailable"
        ),
        benchmark_standard_error_mean=benchmark_mean_error,
        assumptions=(
            "The sampling interval treats included shots as independent.",
            "Benchmark state errors are combined as independent when supplied.",
            "Interpolation stays within an exact lie/context/target stratum.",
        ),
    )


def analyze_source_backed_strokes_gained(
    frame: pd.DataFrame,
    baseline: ExpectedStrokesBaselineV2,
    request: StrokesGainedRequestV1,
    *,
    context: AnalysisContextV2 | None = None,
) -> StrokesGainedAnalysisResultV1:
    """Return governed SG values without mutating caller records.

    Postcondition: every reported value links to a hash-verified benchmark and
    an explicit start and finish lie/context/target/distance state.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    _validate_columns(frame, request)
    rows, excluded = _analyze_rows(frame, baseline, request)
    availability_state, availability = _availability(len(rows), request.min_samples)
    status: Literal["available", "partial", "unavailable"] = availability_state
    if status == "available" and excluded:
        status = "partial"
    by_reason: dict[str, int] = {}
    for row in excluded:
        by_reason[row.reason_code] = by_reason.get(row.reason_code, 0) + 1
    warnings = (
        (f"{len(excluded)} rows were excluded under declared rules.",)
        if excluded
        else ()
    )
    summary = (
        estimate_summary([row.strokes_gained for row in rows], request.confidence_level)
        if availability_state == "available"
        else EstimateSummaryV1(count=len(rows))
    )
    grouped = (
        group_summaries(rows, request.summaries, request.confidence_level)
        if availability_state == "available"
        else ()
    )
    longitudinal = (
        longitudinal_summaries(rows, request.longitudinal)
        if availability_state == "available"
        else ()
    )
    return StrokesGainedAnalysisResultV1(
        status=status,
        value_summary=summary,
        baseline=BaselineProvenanceV1(
            baseline_id=baseline.baseline_id,
            version=baseline.version,
            source_url=baseline.source_url,
            license=baseline.license,
            table_sha256=baseline.table_sha256,
            contract_version=baseline.contract_version,
        ),
        formula=(
            "SG = verified E(start lie/context/target/distance) - 1 - "
            "verified E(finish lie/context/target/distance)"
        ),
        units={"strokes_gained": "strokes", "distance": "yd"},
        availability=availability,
        uncertainty=_uncertainty(rows, request.confidence_level),
        row_results=rows,
        excluded_rows=excluded,
        exclusions=ExclusionSummaryV1(
            input_row_count=len(frame),
            included_row_count=len(rows),
            total_excluded=len(excluded),
            by_reason=by_reason,
        ),
        group_summaries=grouped,
        longitudinal_summaries=longitudinal,
        analysis_context=context or AnalysisContextV2(),
        dataset_fingerprint_sha256=_dataset_fingerprint(frame),
        warnings=warnings,
        limitations=(
            "This is descriptive scoring bookkeeping, not causal inference.",
            "Target/hole, lie, and context labels must be supplied and valid.",
            "The baseline declaration is not an independent license audit.",
            "Results outside benchmark support fail closed rather than extrapolate.",
        ),
    )


def strokes_gained_contract_json_schema() -> dict[str, object]:
    """Return the canonical result schema published to static clients."""

    return StrokesGainedAnalysisResultV1.model_json_schema()


__all__ = [
    "YARDS_PER_METRE",
    "analyze_source_backed_strokes_gained",
    "strokes_gained_contract_json_schema",
]
