"""The wider PW nominal box is a real native-physics feasibility result."""

from __future__ import annotations

import pytest

pytest.importorskip("upstream_physics")

from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.core import PATTERNS
from src.tools.shot_pattern_analysis.matched_physics import MatchedShotPhysics
from src.tools.shot_pattern_analysis.matched_runner import (
    ExperimentConfig,
    calibrate_patterns,
)
from src.tools.shot_pattern_analysis.noise import NoiseSpec
from src.tools.shot_pattern_analysis.nominal_matching import (
    MatchBounds,
    match_nominal_landing,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_pw_strong_curve_matches_same_raw_landing_inside_reviewed_box():
    request = ExperimentConfig(
        matched_preset("pitching_wedge"),
        NoiseSpec((0,) * 7),
        n_shots=2,
        curve_scale=2,
        target_x_m=77.40247724866299,
        match_face_bounds_deg=(-6.0, 6.0),
    )
    target, nominals = calibrate_patterns(request, MatchedShotPhysics())
    assert target == pytest.approx(77.40247724866299)
    assert set(nominals) == {"Straight", "Draw", "Fade"}
    for name, nominal in nominals.items():
        assert nominal.residual_m < request.match_tolerance_m
        assert nominal.outcome.carry_x_m == pytest.approx(target, abs=0.05)
        assert nominal.outcome.carry_y_m == pytest.approx(0, abs=0.05)
        assert 24.0 <= nominal.speed_mps <= 40.0
        if name == "Draw":
            assert -0.4 < nominal.face_deg < -0.1
            assert nominal.path_deg > 2.0
        elif name == "Fade":
            assert 0.1 < nominal.face_deg < 0.4
            assert nominal.path_deg < -2.0


def test_default_bounds_reproduce_explicit_legacy_search_box_exactly():
    physics = MatchedShotPhysics()
    request = ExperimentConfig(
        matched_preset("pitching_wedge"),
        NoiseSpec((0,) * 7),
        n_shots=2,
        curve_scale=1,
        target_x_m=77.40247724866299,
    )
    _, new_nominals = calibrate_patterns(request, physics)
    for pattern in PATTERNS:
        reference = pattern.face_deg
        legacy = match_nominal_landing(
            physics,
            config=request.club,
            target_x_m=request.target_x_m,
            reference_face_deg=reference,
            face_to_path_deg=pattern.face_deg - pattern.path_deg,
            bounds=MatchBounds(
                reference - 3.0,
                reference + 3.0,
                0.75 * request.club.club_speed_mps,
                1.25 * request.club.club_speed_mps,
            ),
            tolerance_m=request.match_tolerance_m,
        )
        current = new_nominals[pattern.name]
        assert current.face_deg == legacy.face_deg
        assert current.path_deg == legacy.path_deg
        assert current.speed_mps == legacy.speed_mps
        assert current.residual_m == legacy.residual_m
