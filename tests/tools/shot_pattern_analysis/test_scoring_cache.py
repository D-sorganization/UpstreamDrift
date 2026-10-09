"""The public scoring API is authoritative at cached interpolation knots."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.tools.launch_monitor_model import (
    CourseStateColumnsV1,
    StrokesGainedRequestV1,
    analyze_source_backed_strokes_gained,
)
from src.tools.shot_pattern_analysis.scoring import build_broadie_approx_baseline
from src.tools.shot_pattern_analysis.scoring_cache import source_backed_score_cache

pytestmark = pytest.mark.unit


def _direct(lie: str, distance: float) -> float:
    baseline = build_broadie_approx_baseline()
    request = StrokesGainedRequestV1(
        start=CourseStateColumnsV1(
            lie_column="start_lie",
            context_column="context",
            target_column="target",
            distance_column="start_distance",
            distance_unit="yd",
        ),
        finish=CourseStateColumnsV1(
            lie_column="finish_lie",
            context_column="context",
            target_column="target",
            distance_column="finish_distance",
            distance_unit="yd",
        ),
        min_samples=1,
    )
    frame = pd.DataFrame(
        [
            {
                "start_lie": "fairway",
                "finish_lie": lie,
                "context": "standard",
                "target": "hole",
                "start_distance": 200.0,
                "finish_distance": distance,
            }
        ]
    )
    result = analyze_source_backed_strokes_gained(frame, baseline, request)
    assert result.exclusions.total_excluded == 0
    return result.row_results[0].strokes_gained


def test_cache_parity_at_knots_midpoints_and_random_interior() -> None:
    baseline = build_broadie_approx_baseline()
    cache = source_backed_score_cache(
        baseline,
        start_lie="fairway",
        start_distance_yards=200.0,
        finish_lies=("green", "rough"),
    )
    assert cache.api_evaluated_states > 20
    for lie, distances in (
        ("green", (0.0, 1 / 3, 5.5, 10.0, 19.37, 25.0)),
        ("rough", (10.0, 15.0, 20.0, 33.333, 75.0, 220.0)),
    ):
        for distance in distances:
            assert cache.score(lie, distance) == pytest.approx(
                _direct(lie, distance), abs=1e-12
            )


def test_cache_refuses_extrapolation_and_invalid_distance() -> None:
    cache = source_backed_score_cache(
        build_broadie_approx_baseline(),
        start_lie="fairway",
        start_distance_yards=200.0,
        finish_lies=("green", "rough"),
    )
    for distance in (-1.0, math.nan, math.inf, 700.0):
        with pytest.raises(ValueError):
            cache.score("rough", distance)
    with pytest.raises(ValueError):
        cache.score("sand", 20.0)
