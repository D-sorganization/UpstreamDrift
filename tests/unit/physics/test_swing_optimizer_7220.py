"""Tests for inverse swing optimization core (#7220)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.physics.impact_model import ImpactParameters, PostImpactState
from src.shared.python.physics.swing_ball_flight_pipeline import (
    PipelineResult,
    SwingState,
)
from src.shared.python.physics.swing_optimizer import (
    ClubPreset,
    FlightTarget,
    OptimizationControls,
    SwingOptimizer,
)

pytestmark = pytest.mark.unit


class _AnalyticForwardPipeline:
    """Fast deterministic stand-in for SwingBallFlightPipeline."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, swing: SwingState) -> PipelineResult:
        self.calls += 1
        speed = float(np.linalg.norm(swing.clubhead_velocity))
        optimizer_params = swing.metadata["optimizer_parameters"]
        loft_deg = float(optimizer_params["loft_deg"])
        attack_deg = float(optimizer_params["attack_angle_deg"])
        face_deg = float(optimizer_params["face_to_path_deg"])

        carry_m = 4.0 * speed + 0.9 * loft_deg + 1.8 * attack_deg
        max_height_m = 0.12 * speed + 1.7 * loft_deg + 0.7 * attack_deg
        lateral_m = 2.4 * face_deg

        return PipelineResult(
            swing_state=swing,
            impact_state=PostImpactState(
                ball_velocity=np.array([carry_m / 5.0, lateral_m / 5.0, 10.0]),
                ball_angular_velocity=np.zeros(3),
                clubhead_velocity=np.zeros(3),
                clubhead_angular_velocity=np.zeros(3),
                contact_duration=0.0,
                energy_transfer=0.0,
                impact_location=np.zeros(2),
            ),
            launch_conditions=swing.metadata["launch_conditions"],
            trajectory=[],
            carry_m=carry_m,
            max_height_m=max_height_m,
            flight_time_s=5.0,
            landing_angle_deg=40.0,
            impact_params=ImpactParameters(),
            metadata={"terminal_lateral_m": lateral_m},
        )


def _target_from_result(result: PipelineResult) -> FlightTarget:
    return FlightTarget(
        carry_m=result.carry_m,
        max_height_m=result.max_height_m,
        lateral_m=result.metadata["terminal_lateral_m"],
    )


@pytest.mark.parametrize("club", [ClubPreset.driver(), ClubPreset.iron_7()])
def test_optimizer_roundtrips_reachable_target_within_carry_tolerance(
    club: ClubPreset,
) -> None:
    pipeline = _AnalyticForwardPipeline()
    optimizer = SwingOptimizer(pipeline=pipeline)
    expected_swing = optimizer.build_swing_state(
        speed_mps=club.initial_guess[0],
        loft_deg=club.initial_guess[1],
        attack_angle_deg=club.initial_guess[2],
        face_to_path_deg=club.initial_guess[3],
        club=club,
    )
    target = _target_from_result(pipeline.run(expected_swing))

    result = optimizer.solve(
        target,
        club,
        controls=OptimizationControls(
            max_iterations=40,
            timeout_s=2.0,
            carry_tolerance_fraction=0.02,
            absolute_tolerance_m=0.75,
        ),
    )

    assert result.diagnostics.converged
    assert not result.diagnostics.unreachable
    assert result.swing_state is not None
    assert result.pipeline_result is not None
    assert result.pipeline_result.carry_m == pytest.approx(target.carry_m, rel=0.02)
    assert result.diagnostics.evaluations <= 80


def test_optimizer_reports_non_achievable_target_without_returning_garbage() -> None:
    pipeline = _AnalyticForwardPipeline()
    optimizer = SwingOptimizer(pipeline=pipeline)
    target = FlightTarget(carry_m=2_000.0, max_height_m=220.0, lateral_m=0.0)

    result = optimizer.solve(
        target,
        ClubPreset.driver(),
        controls=OptimizationControls(max_iterations=20, timeout_s=2.0),
    )

    assert not result.diagnostics.converged
    assert result.diagnostics.unreachable
    assert result.swing_state is not None
    assert result.pipeline_result is not None
    assert "unreachable" in result.diagnostics.message.lower()
    assert abs(result.diagnostics.carry_error_m) > 1_000.0


def test_optimizer_timeout_returns_best_diagnostic() -> None:
    optimizer = SwingOptimizer(pipeline=_AnalyticForwardPipeline())

    result = optimizer.solve(
        FlightTarget(carry_m=180.0, max_height_m=35.0),
        ClubPreset.driver(),
        controls=OptimizationControls(max_iterations=200, timeout_s=0.0),
    )

    assert result.diagnostics.timed_out
    assert not result.diagnostics.converged
    assert result.swing_state is not None
    assert result.pipeline_result is not None
    assert result.diagnostics.evaluations >= 1
