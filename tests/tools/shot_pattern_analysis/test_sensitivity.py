"""Deterministic diagnostics preserve the stated independent controls."""

import pytest

from src.tools.shot_pattern_analysis.core import AnalysisConfig
from src.tools.shot_pattern_analysis.physics import ShotOutcome
from src.tools.shot_pattern_analysis.sensitivity import (
    build_sensitivity,
    fixed_club_pitch,
)

pytestmark = pytest.mark.unit


class FakePhysics:
    def simulate(self, *, face_deg, path_deg, config, **kwargs):
        assert path_deg == 0
        return ShotOutcome(
            100.0,
            face_deg,
            100.0,
            face_deg,
            3000.0,
            0.0,
            launch_elevation_deg=10.0,
            ball_speed_mps=60.0,
        )


def test_sensitivity_grid_holds_nominal_loft_and_separates_pitch():
    config = AnalysisConfig(loft_deg=24.0, lie_deg=63.0)
    result = build_sensitivity({"test": config}, engine=FakePhysics())
    club = result["clubs"]["test"]
    assert len(club["face_error_axis_grid"]) == 30
    assert len(club["loft_only_grid"]) == 5
    assert len(club["fixed_club_pitch_grid"]) == 3
    for row in club["face_error_axis_grid"]:
        assert row["nominal_loft_deg"] == 24.0
        if row["face_error_deg"] == 0 or row["delivery_mode"] == "fixed_loft":
            assert row["dynamic_loft_deg"] == pytest.approx(24.0)
    pitched = club["fixed_club_pitch_grid"][-1]
    assert pitched["dynamic_loft_deg"] == pytest.approx(4.0)
    assert pitched["shaft_elevation_deg"] < 63.0


def test_fixed_club_pitch_is_rigid_and_preserves_face_shaft_angle():
    zero = fixed_club_pitch(base_loft_deg=24.0, lie_deg=63.0, pitch_deg=0.0)
    leaned = fixed_club_pitch(base_loft_deg=24.0, lie_deg=63.0, pitch_deg=20.0)
    assert leaned["normal_dot_shaft"] == pytest.approx(
        zero["normal_dot_shaft"], abs=1e-14
    )
    assert leaned["shaft_lean_deg"] == pytest.approx(20.0)
    assert leaned["face_angle_deg"] == pytest.approx(0.0)


@pytest.mark.parametrize("pitch", [float("nan"), -1.0, 45.0])
def test_pitch_outside_control_domain_is_rejected(pitch):
    with pytest.raises(ValueError):
        fixed_club_pitch(base_loft_deg=24.0, lie_deg=63.0, pitch_deg=pitch)


def test_empty_config_collection_is_rejected():
    with pytest.raises(ValueError):
        build_sensitivity({}, engine=FakePhysics())
