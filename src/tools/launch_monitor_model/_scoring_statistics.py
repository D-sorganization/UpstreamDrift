"""Shared descriptive uncertainty and trusted grouping helpers."""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy import stats

from src.tools.launch_monitor_model.strokes_gained_types import (
    ConfidenceIntervalV1,
    EstimateSummaryV1,
    GroupingDimensionV1,
    GroupSummaryV1,
    LongitudinalDimensionV1,
    LongitudinalSummaryV1,
    StrokesGainedRowV1,
)


def estimate_summary(values: list[float], confidence_level: float) -> EstimateSummaryV1:
    """Return a finite descriptive mean and Student-t interval when possible."""

    if not values:
        return EstimateSummaryV1(count=0)
    vector = np.asarray(values, dtype=float)
    mean = float(np.mean(vector))
    if len(vector) == 1:
        return EstimateSummaryV1(count=1, mean=mean)
    deviation = float(np.std(vector, ddof=1))
    standard_error = deviation / sqrt(len(vector))
    critical = float(stats.t.ppf(0.5 + confidence_level / 2.0, len(vector) - 1))
    return EstimateSummaryV1(
        count=len(vector),
        mean=mean,
        standard_deviation=deviation,
        standard_error=standard_error,
        confidence_interval=ConfidenceIntervalV1(
            lower=mean - critical * standard_error,
            upper=mean + critical * standard_error,
            level=confidence_level,
            method="student-t",
        ),
    )


def group_summaries(
    rows: tuple[StrokesGainedRowV1, ...],
    specs: tuple[GroupingDimensionV1, ...],
    confidence_level: float,
) -> tuple[GroupSummaryV1, ...]:
    """Summarize only groups explicitly mapped with evidence in the request."""

    output: list[GroupSummaryV1] = []
    for spec in specs:
        values_by_group: dict[str, list[float]] = {}
        for row in rows:
            value = row.groups.get(spec.dimension)
            if value:
                values_by_group.setdefault(value, []).append(row.strokes_gained)
        for value in sorted(values_by_group):
            output.append(
                GroupSummaryV1(
                    dimension=spec.dimension,
                    group_value=value,
                    estimate=estimate_summary(values_by_group[value], confidence_level),
                    trust_level=spec.trust_level,
                    evidence=spec.evidence,
                )
            )
    return tuple(output)


def _trend(
    rows: list[StrokesGainedRowV1],
    spec: LongitudinalDimensionV1,
    group_dimension: str,
    group_value: str,
) -> LongitudinalSummaryV1 | None:
    complete = [row for row in rows if row.longitudinal_order is not None]
    if len(complete) < spec.min_samples:
        return None
    order = np.asarray([row.longitudinal_order for row in complete], dtype=float)
    if len(np.unique(order)) < 2:
        return None
    values = np.asarray([row.strokes_gained for row in complete], dtype=float)
    estimate = stats.linregress(order, values)
    return LongitudinalSummaryV1(
        group_dimension=group_dimension,  # type: ignore[arg-type]
        group_value=group_value,
        sample_count=len(complete),
        slope=float(estimate.slope),
        intercept=float(estimate.intercept),
        r_squared=float(estimate.rvalue**2),
        p_value=float(estimate.pvalue),
        slope_unit=f"strokes/{spec.order_unit}",
        trust_level=spec.trust_level,
        evidence=spec.evidence,
    )


def longitudinal_summaries(
    rows: tuple[StrokesGainedRowV1, ...],
    spec: LongitudinalDimensionV1 | None,
) -> tuple[LongitudinalSummaryV1, ...]:
    """Fit descriptive SG trends only for explicitly evidenced order fields."""

    if spec is None:
        return ()
    if spec.group_dimension is None:
        summary = _trend(list(rows), spec, "all", "all")
        return () if summary is None else (summary,)
    grouped: dict[str, list[StrokesGainedRowV1]] = {}
    for row in rows:
        value = row.groups.get(spec.group_dimension)
        if value:
            grouped.setdefault(value, []).append(row)
    output = [
        summary
        for value in sorted(grouped)
        if (summary := _trend(grouped[value], spec, spec.group_dimension, value))
        is not None
    ]
    return tuple(output)


__all__ = ["estimate_summary", "group_summaries", "longitudinal_summaries"]
