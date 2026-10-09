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
