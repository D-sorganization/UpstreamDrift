"""Per-row source-backed scoring shares the canonical course geometry."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.shot_pattern_analysis.course_score_vectors import (
    score_course_endpoint_vectors,
)
from src.tools.shot_pattern_analysis.scenario_scoring import score_course_endpoints

pytestmark = pytest.mark.unit


def _endpoints() -> dict[str, list[tuple[int, float, float]]]:
    return {
        "Straight": [(0, 195.0, 0.0), (1, 202.0, 18.0), (2, 197.0, -11.0)],
        "Draw": [(0, 196.0, 5.0), (1, 201.0, 25.0), (2, 198.0, -12.0)],
        "Fade": [(0, 194.0, -5.0), (1, 203.0, 12.0), (2, 199.0, -23.0)],
    }


@pytest.mark.parametrize("club_id", ["driver", "seven_iron", "pitching_wedge"])
def test_source_backed_vectors_reproduce_canonical_means_and_lie_fractions(
    club_id: str,
) -> None:
    options = {
        "club_id": club_id,
        "target_x_m": 200.0,
        "hole_distance_m": 400.0,
        "fairway_half_width_m": 10.0,
        "green_radius_m": 12.0,
    }
    vectors = score_course_endpoint_vectors(_endpoints(), **options)
    report = score_course_endpoints(_endpoints(), **options)

    assert set(vectors.scores) == {"Straight", "Draw", "Fade"}
    assert vectors.api_evaluated_states == report["api_evaluated_states"]
    assert vectors.baseline == report["baseline"]
    assert vectors.start_lie == report["start_lie"]
    assert vectors.start_distance_m == report["start_distance_m"]
    assert vectors.scenario_id == report["scenario_id"]
    for pattern, scores in vectors.scores.items():
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))
        assert float(np.mean(scores)) == pytest.approx(
            report["patterns"][pattern]["mean_strokes_gained"], abs=1e-12
        )
        assert vectors.primary_fraction[pattern] == pytest.approx(
            report["patterns"][pattern][vectors.primary_fraction_name]
        )


def test_vector_scoring_preserves_row_order_and_rejects_unpaired_inputs() -> None:
    result = score_course_endpoint_vectors(
        _endpoints(), club_id="driver", target_x_m=200.0
    )
    assert result.scores["Straight"][0] != result.scores["Straight"][1]
    missing = _endpoints()
    missing["Fade"] = missing["Fade"][:-1]
    with pytest.raises(ValueError, match="share ordered shot indices"):
        score_course_endpoint_vectors(missing, club_id="driver", target_x_m=200.0)
