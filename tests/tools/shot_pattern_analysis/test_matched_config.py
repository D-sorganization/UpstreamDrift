"""Study inputs distinguish static equipment geometry from delivery."""

from dataclasses import replace
import pytest
from src.tools.shot_pattern_analysis.matched_config import MatchedConfig, matched_preset

pytestmark = pytest.mark.unit


def test_documented_static_lofts_and_lean_support():
    for club, loft in [("driver", 10), ("seven_iron", 34), ("pitching_wedge", 46)]:
        config = matched_preset(club)
        assert config.loft_deg == loft
        assert config.loft_source.startswith("https://www.titleist.com/")
        assert replace(config, shaft_lean_deg=10).shaft_lean_deg == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"club_speed_mps": 0},
        {"loft_deg": float("nan")},
        {"inertia_diagonal_kg_m2": (1, 1, 3)},
        {"cg_depth_m": -1},
        {"friction_coefficient": -1},
        {"cor": 1.1},
        {"dt_s": 0},
        {"face_half_height_m": 0},
        {"bulge_radius_m": -1},
    ],
)
def test_bad_physical_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        replace(MatchedConfig(), **kwargs)
