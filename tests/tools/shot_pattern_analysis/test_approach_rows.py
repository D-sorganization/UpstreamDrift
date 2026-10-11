"""Approach course-state preparation preserves geometry and source ordering."""

import pytest

from src.tools.shot_pattern_analysis import scoring

pytestmark = pytest.mark.unit


def test_approach_rows_preserve_boundary_units_and_shot_order():
    endpoints = {
        name: [(7, 203.0, 4.0), (2, 200.0, 5.01)]
        for name in ("Fade", "Straight", "Draw")
    }
    rows, keys = scoring._approach_course_rows(endpoints, 200.0, 5.0)
    assert keys == [(name, i) for name in endpoints for i in (7, 2)]
    assert rows[0] == {
        "start_lie": "fairway",
        "start_context": "standard",
        "finish_lie": "green",
        "finish_context": "standard",
        "target": "hole",
        "start_distance_yd": 200.0 * scoring.YARDS_PER_METRE,
        "finish_distance_yd": 5.0 * scoring.YARDS_PER_METRE,
    }
    assert rows[1]["finish_lie"] == "rough"
    assert rows[1]["finish_distance_yd"] == 5.01 * scoring.YARDS_PER_METRE


@pytest.mark.parametrize(
    "endpoints,target,radius,error",
    [
        ({}, 0, 0, "target distance and green radius must be positive finite"),
        ({}, 700, 5, "start distance is outside the published fairway benchmark range"),
        ({}, 200, 5, "all three patterns are required"),
        (
            {name: [] for name in ("Straight", "Draw", "Fade")},
            200,
            5,
            "each pattern requires at least two scored endpoints",
        ),
        (
            {
                name: [(-1, 200, 0), (2, 200, 0)]
                for name in ("Straight", "Draw", "Fade")
            },
            200,
            5,
            "endpoints must have nonnegative indices and finite coordinates",
        ),
    ],
)
def test_approach_row_validation_order(endpoints, target, radius, error):
    with pytest.raises(ValueError) as caught:
        scoring._approach_course_rows(endpoints, target, radius)
    assert str(caught.value) == error


def test_approach_scores_match_original_public_api_exactly():
    import json
    from pathlib import Path

    reference = json.loads(
        (Path(__file__).parent / "fixtures/approach_scoring_reference.json").read_text()
    )
    for case in reference["cases"]:
        observed = scoring.score_approach_endpoints(
            case["endpoints"],
            target_x_m=200.0,
            green_radius_m=case["radius"],
            seed=42,
        )
        assert observed == case["output"]
