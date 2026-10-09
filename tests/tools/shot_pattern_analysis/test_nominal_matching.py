"""Nominal shapes must meet one raw target without postflight re-aiming."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from src.tools.shot_pattern_analysis.core import AnalysisConfig
from src.tools.shot_pattern_analysis.nominal_matching import (
    MatchBounds,
    match_nominal_landing,
)
from src.tools.shot_pattern_analysis.physics import ShotOutcome

pytestmark = pytest.mark.unit


class LinearPhysics:
    def __init__(self) -> None:
        self.calls: list[tuple[float, float, float, float]] = []

    def simulate(
        self,
        *,
        face_deg: float,
        path_deg: float,
        config: AnalysisConfig,
        sample_trajectory: bool = False,
        nominal_face_deg: float = 0.0,
    ) -> ShotOutcome:
        assert not sample_trajectory
        self.calls.append((face_deg, path_deg, config.club_speed_mps, nominal_face_deg))
        x = 4.0 * config.club_speed_mps + 2.0 * (path_deg - face_deg)
        y = face_deg - 0.4 * (path_deg - face_deg)
        return ShotOutcome(x, y, math.hypot(x, y), face_deg, 3000.0, 0.0)


@pytest.mark.parametrize(
    ("reference_face", "face_to_path"),
    [(0.0, 0.0), (1.5, -1.5), (-1.5, 1.5)],
)
def test_shapes_match_same_raw_landing_with_face_to_path_preserved(
    reference_face: float, face_to_path: float
) -> None:
    physics = LinearPhysics()
    config = replace(AnalysisConfig(), club_speed_mps=25.0)
    matched = match_nominal_landing(
        physics,
        config=config,
        target_x_m=100.0,
        reference_face_deg=reference_face,
        face_to_path_deg=face_to_path,
        bounds=MatchBounds(-3.0, 3.0, 20.0, 30.0),
    )
    assert matched.outcome.carry_x_m == pytest.approx(100.0, abs=1e-5)
    assert matched.outcome.carry_y_m == pytest.approx(0.0, abs=1e-5)
    assert matched.face_deg - matched.path_deg == pytest.approx(face_to_path)
    assert physics.calls[-1][3] == pytest.approx(matched.face_deg)
    assert matched.residual_m <= 1e-5


def test_unreachable_target_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unreachable"):
        match_nominal_landing(
            LinearPhysics(),
            config=replace(AnalysisConfig(), club_speed_mps=25.0),
            target_x_m=400.0,
            reference_face_deg=1.5,
            face_to_path_deg=-1.5,
            bounds=MatchBounds(0.0, 3.0, 20.0, 30.0),
        )


def test_invalid_bounds_and_target_rejected() -> None:
    with pytest.raises(ValueError, match="bounds"):
        MatchBounds(3.0, -3.0, 20.0, 30.0)
    with pytest.raises(ValueError, match="bounds"):
        MatchBounds(-3.0, 3.0, 0.0, 30.0)
    with pytest.raises(ValueError, match="target"):
        match_nominal_landing(
            LinearPhysics(),
            config=AnalysisConfig(),
            target_x_m=float("nan"),
            reference_face_deg=0.0,
            face_to_path_deg=0.0,
        )
