"""Independent dense parity against the authoritative public scoring API."""

import numpy as np
import pandas as pd
import pytest

from src.tools.launch_monitor_model import (
    CourseStateColumnsV1,
    StrokesGainedRequestV1,
    analyze_source_backed_strokes_gained,
)
from src.tools.shot_pattern_analysis.scoring import (
    YARDS_PER_METRE,
    build_broadie_approx_baseline,
)
from src.tools.shot_pattern_analysis.scoring_cache import source_backed_score_cache

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "start_lie,start_m,finish_lies",
    [
        ("tee", 400.0, ("fairway", "rough")),
        ("fairway", 149.74812854111136, ("green", "rough")),
        ("fairway", 101.15986452790243, ("green", "rough")),
    ],
)
def test_every_knot_midpoint_and_100_random_states_match_public_api(
    start_lie, start_m, finish_lies
):
    baseline = build_broadie_approx_baseline()
    start_yd = start_m * YARDS_PER_METRE
    cache = source_backed_score_cache(
        baseline,
        start_lie=start_lie,
        start_distance_yards=start_yd,
        finish_lies=finish_lies,
    )
    rng = np.random.default_rng(20261008)
    rows, expected = [], []
    for lie in finish_lies:
        knots = np.array(
            sorted(
                point.distance_yards
                for point in baseline.states
                if point.lie == lie
                and point.context == "standard"
                and point.target == "hole"
            )
        )
        distances = np.concatenate(
            (knots, (knots[:-1] + knots[1:]) / 2, rng.uniform(knots[0], knots[-1], 100))
        )
        expected.extend(cache.score_many(lie, distances).tolist())
        for distance in distances:
            rows.append(
                {
                    "start_lie": start_lie,
                    "finish_lie": lie,
                    "context": "standard",
                    "target": "hole",
                    "start_distance": start_yd,
                    "finish_distance": distance,
                }
            )
        for outside in (
            np.nextafter(knots[0], -np.inf),
            np.nextafter(knots[-1], np.inf),
        ):
            with pytest.raises(ValueError):
                cache.score(lie, outside)
            with pytest.raises(ValueError):
                cache.score_many(lie, np.array([outside]))
    common = {
        "context_column": "context",
        "target_column": "target",
        "distance_unit": "yd",
    }
    request = StrokesGainedRequestV1(
        start=CourseStateColumnsV1(
            lie_column="start_lie", distance_column="start_distance", **common
        ),
        finish=CourseStateColumnsV1(
            lie_column="finish_lie", distance_column="finish_distance", **common
        ),
        min_samples=1,
    )
    result = analyze_source_backed_strokes_gained(pd.DataFrame(rows), baseline, request)
    assert result.exclusions.total_excluded == 0
    assert len(result.row_results) == len(rows)
    assert [row.source_index for row in result.row_results] == list(range(len(rows)))
    direct = [row.strokes_gained for row in result.row_results]
    np.testing.assert_allclose(expected, direct, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "bundle",
    ["results", "results_sd2", "results_large_curve_sd1", "results_large_curve_sd2"],
)
def test_cached_means_match_preserved_full_api_historical_bundles(bundle):
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    directory = root / "docs/research/shot_pattern_analysis" / bundle
    report = json.loads((directory / "strokes_gained.json").read_text())
    assert report["api_scored_shots"] == 30000
    samples = pd.read_csv(directory / "shots.csv")
    target = report["target_distance_m"]
    radius = report["green_radius_m"]
    cache = source_backed_score_cache(
        build_broadie_approx_baseline(),
        start_lie="fairway",
        start_distance_yards=target * YARDS_PER_METRE,
        finish_lies=("green", "rough"),
    )
    for pattern, expected in report["patterns"].items():
        group = samples[samples["pattern"] == pattern]
        assert len(group) == expected["n"]
        distance_m = np.hypot(
            group["aimed_x_m"].to_numpy() - target, group["aimed_y_m"].to_numpy()
        )
        green = distance_m <= radius
        scores = np.empty(len(group))
        for lie, mask in (("green", green), ("rough", ~green)):
            scores[mask] = cache.score_many(lie, distance_m[mask] * YARDS_PER_METRE)
        assert abs(float(np.mean(scores)) - expected["mean_strokes_gained"]) <= 1e-12
