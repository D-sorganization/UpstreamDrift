"""Contracts for matched experiment comparisons."""

import pytest

from src.tools.shot_pattern_analysis.comparison import validate_comparison

pytestmark = pytest.mark.unit


def _summary(sd=1.0, scale=1.0):
    return {
        "config": {
            "seed": 7,
            "n_shots": 10000,
            "face_sd_deg": sd,
            "curve_scale": scale,
        },
        "target_x_m": 195.0,
        "patterns": {
            name: {
                "aimed_lateral_sd_m": 9.0,
                "aimed_target_hit_fraction": 0.87,
                "mean_carry_m": 194.0,
            }
            for name in ("Straight", "Draw", "Fade")
        },
    }


def test_matched_comparison_accepted():
    validate_comparison([_summary(), _summary(2, 2)])


@pytest.mark.parametrize("field,value", [("seed", 8), ("n_shots", 999)])
def test_mismatched_pairing_rejected(field, value):
    other = _summary()
    other["config"][field] = value
    with pytest.raises(ValueError, match="matched"):
        validate_comparison([_summary(), other])


def test_empty_comparison_rejected():
    with pytest.raises(ValueError):
        validate_comparison([])


@pytest.mark.parametrize(
    "field,value",
    [("club_id", "seven_iron"), ("club_speed_mps", 36.0), ("loft_deg", 24.0)],
)
def test_different_delivery_baselines_rejected(field, value):
    reference, other = _summary(), _summary()
    reference["config"].update(club_id="driver", club_speed_mps=45.0, loft_deg=12.8)
    other["config"].update(reference["config"])
    other["config"][field] = value
    with pytest.raises(ValueError, match="baseline"):
        validate_comparison([reference, other])


def test_scoring_comparison_rejects_partial_api_evaluation():
    from src.tools.shot_pattern_analysis.comparison import validate_scoring_comparison

    summary = _summary()
    summary["approach_scoring"] = {"api_scored_shots": 90}
    with pytest.raises(ValueError, match="every shot"):
        validate_scoring_comparison([summary])


def test_exact_public_api_cache_scoring_accepted():
    from src.tools.shot_pattern_analysis.comparison import validate_scoring_comparison

    summary = _summary()
    summary["course_scoring"] = {
        "status": "available",
        "scored_shots": 30000,
        "api_evaluated_states": 100,
        "interpolation_tolerance_strokes": 1e-12,
        "baseline": {"table_sha256": "verified-table"},
        "scenario_id": "tee_fairway_rough",
        "start_lie": "tee",
        "start_distance_m": 400.0,
        "fairway_half_width_m": 15.0,
        "green_radius_m": None,
    }
    validate_scoring_comparison([summary])
