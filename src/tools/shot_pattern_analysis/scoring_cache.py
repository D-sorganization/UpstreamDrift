"""Exact linear interpolation of source-backed public scoring API outputs.

This does not reproduce the provider's strokes-gained or baseline formula.
The public API scores every published knot for a fixed start state; only its
returned scores are interpolated. Out-of-support endpoints are rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.tools.launch_monitor_model import (
    CourseStateColumnsV1,
    ExpectedStrokesBaselineV2,
    StrokesGainedRequestV1,
    analyze_source_backed_strokes_gained,
)


@dataclass(frozen=True)
class SourceBackedScoreCache:
    """API-returned score arrays keyed by finish lie and fixed start state."""

    knots: dict[str, tuple[np.ndarray, np.ndarray]]
    api_evaluated_states: int
    baseline_sha256: str
    start_lie: str
    start_distance_yards: float

    def score(self, finish_lie: str, distance_yards: float) -> float:
        if not math.isfinite(distance_yards) or distance_yards < 0:
            raise ValueError("finish distance must be finite and nonnegative")
        if finish_lie not in self.knots:
            raise ValueError("finish lie is absent from source-backed cache")
        distances, values = self.knots[finish_lie]
        if not distances[0] <= distance_yards <= distances[-1]:
            raise ValueError("finish distance outside source-backed baseline support")
        return float(np.interp(distance_yards, distances, values))

    def score_many(self, finish_lie: str, distances_yards: np.ndarray) -> np.ndarray:
        distances = np.asarray(distances_yards, dtype=float)
        if not np.all(np.isfinite(distances)) or np.any(distances < 0):
            raise ValueError("finish distances must be finite and nonnegative")
        if finish_lie not in self.knots:
            raise ValueError("finish lie is absent from source-backed cache")
        knots, values = self.knots[finish_lie]
        if np.any(distances < knots[0]) or np.any(distances > knots[-1]):
            raise ValueError("finish distance outside source-backed baseline support")
        return np.interp(distances, knots, values)


def source_backed_score_cache(
    baseline: ExpectedStrokesBaselineV2,
    *,
    start_lie: str,
    start_distance_yards: float,
    finish_lies: tuple[str, ...],
) -> SourceBackedScoreCache:
    """Evaluate each finish-state knot through the canonical Tools API."""
    if not math.isfinite(start_distance_yards) or start_distance_yards < 0:
        raise ValueError("start distance must be finite and nonnegative")
    if not finish_lies or len(set(finish_lies)) != len(finish_lies):
        raise ValueError("finish_lies must contain distinct supported lies")
    points = sorted(
        (
            point
            for point in baseline.states
            if point.lie in finish_lies
            and point.context == "standard"
            and point.target == "hole"
        ),
        key=lambda point: (point.lie, point.distance_yards),
    )
    if {point.lie for point in points} != set(finish_lies):
        raise ValueError("requested finish lie absent from baseline")
    rows = [
        {
            "start_lie": start_lie,
            "finish_lie": point.lie,
            "context": "standard",
            "target": "hole",
            "start_distance": start_distance_yards,
            "finish_distance": point.distance_yards,
        }
        for point in points
    ]
    columns = {
        "context_column": "context",
        "target_column": "target",
        "distance_unit": "yd",
    }
    request = StrokesGainedRequestV1(
        start=CourseStateColumnsV1(
            lie_column="start_lie", distance_column="start_distance", **columns
        ),
        finish=CourseStateColumnsV1(
            lie_column="finish_lie", distance_column="finish_distance", **columns
        ),
        min_samples=1,
    )
    result = analyze_source_backed_strokes_gained(pd.DataFrame(rows), baseline, request)
    if result.exclusions.total_excluded or len(result.row_results) != len(points):
        raise ValueError("start or finish state outside source-backed baseline support")
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for lie in finish_lies:
        pairs = [
            (points[row.source_index].distance_yards, row.strokes_gained)
            for row in result.row_results
            if points[row.source_index].lie == lie
        ]
        arrays[lie] = (
            np.array([distance for distance, _ in pairs], dtype=float),
            np.array([score for _, score in pairs], dtype=float),
        )
    return SourceBackedScoreCache(
        knots=arrays,
        api_evaluated_states=len(result.row_results),
        baseline_sha256=baseline.table_sha256,
        start_lie=start_lie,
        start_distance_yards=start_distance_yards,
    )
