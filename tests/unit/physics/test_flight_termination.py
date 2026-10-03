"""Unit tests for flight termination propagation and fail-closed landing gates (#11145).

Verifies:
1. Five distinct termination conditions:
   - TIME_LIMIT (time-cap while ascending)
   - LANDED (normal landing)
   - LANDED (negative launch angle ground contact)
   - SOLVER_FAILED (solver failure or negative status)
   - CANCELLED (cooperative cancellation)
2. Gating of landing metrics (carry_distance, landing_angle, lateral_deviation) to None
   for non-landed flights.
3. FlightResult.require_landing() raising IncompleteFlightError carrying the partial result.
4. FlightResult.__post_init__ contract validation enforcing None landing metrics for unlanded flights.
5. End-to-end provider-to-API-to-viewer contract test.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.api.routes._ball_flight_trajectory_import import (
    ImportedBallFlightTrajectory,
    ImportedTrajectorySample,
    summarize_imported_trajectory,
)
from src.api.routes.ball_flight import (
    BallFlightModelResult,
    BallFlightSimulationRequest,
    BallFlightSummary,
    _simulate_one,
)
from src.shared.python.physics.flight_models import (
    BallFlightModel,
    FlightModelRegistry,
    FlightModelType,
    FlightResult,
    FlightSimulationCancelled,
    FlightTermination,
    IncompleteFlightError,
    TrajectoryPoint,
    UnifiedLaunchConditions,
)

pytestmark = pytest.mark.unit


def _ascending_time_cap_launch() -> UnifiedLaunchConditions:
    return UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=160.0,
        launch_angle_deg=15.0,
        spin_rate_rpm=2500.0,
    )


def _normal_launch() -> UnifiedLaunchConditions:
    return UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=160.0,
        launch_angle_deg=11.0,
        spin_rate_rpm=2500.0,
    )


def _negative_angle_launch() -> UnifiedLaunchConditions:
    return UnifiedLaunchConditions.from_imperial(
        ball_speed_mph=100.0,
        launch_angle_deg=-5.0,
        spin_rate_rpm=2500.0,
    )


class TestFlightTerminationFixtures:
    """Distinct results for all five termination scenarios."""

    def test_time_limit_while_ascending_fixture(self) -> None:
        """Integration reaches max_time while ball is still in the air / ascending."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        result = model.simulate(_ascending_time_cap_launch(), max_time=0.1)

        assert result.termination is FlightTermination.TIME_LIMIT
        assert result.terminal_event is False
        assert result.landed is False
        assert result.flight_completed is False

        # Landing-derived metrics must be None
        assert result.carry_distance is None
        assert result.landing_angle is None
        assert result.lateral_deviation is None

        # Trajectory-derived quantities exist
        assert result.max_height > 0.0
        assert result.flight_time == pytest.approx(0.1, abs=0.02)
        assert result.actual_horizon == pytest.approx(0.1, abs=0.02)
        assert len(result.trajectory) > 1

        # Calling require_landing() must raise IncompleteFlightError
        with pytest.raises(IncompleteFlightError) as exc_info:
            result.require_landing()

        err = exc_info.value
        assert err.result is result
        assert err.termination is FlightTermination.TIME_LIMIT
        assert "termination=time_limit" in str(err)
        assert "Inspect .result for the partial trace" in str(err)

    def test_normal_landing_fixture(self) -> None:
        """Full simulation running to ground impact event."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        result = model.simulate(_normal_launch(), max_time=15.0)

        assert result.termination is FlightTermination.LANDED
        assert result.terminal_event is True
        assert result.landed is True
        assert result.flight_completed is True

        assert result.carry_distance is not None and result.carry_distance > 150.0
        assert result.landing_angle is not None and result.landing_angle > 0.0
        assert result.lateral_deviation is not None
        assert result.max_height > 10.0
        assert result.flight_time > 3.0
        assert result.actual_horizon > 3.0

        # require_landing() returns self
        assert result.require_landing() is result

    def test_negative_launch_angle_fixture(self) -> None:
        """Ball launched downward reaches the ground immediately."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        result = model.simulate(_negative_angle_launch(), max_time=5.0)

        assert result.termination is FlightTermination.LANDED
        assert result.terminal_event is True
        assert result.landed is True
        assert result.flight_completed is True

        assert result.actual_horizon == pytest.approx(0.0, abs=1e-5)
        assert result.carry_distance == pytest.approx(0.0, abs=1e-5)
        assert result.flight_time == pytest.approx(0.0, abs=1e-5)
        assert result.landing_angle is not None
        assert result.landing_angle == pytest.approx(5.0, abs=0.5)

    def test_solver_failed_fixture(self) -> None:
        """Solver failure produces SOLVER_FAILED with None landing metrics."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
        points = [
            TrajectoryPoint(0.0, np.array([0.0, 0.0, 1.0]), np.array([10.0, 0.0, 0.0])),
        ]
        result = model._compute_metrics(
            points,
            termination=FlightTermination.SOLVER_FAILED,
            terminal_event=False,
            actual_horizon=0.0,
        )

        assert result.termination is FlightTermination.SOLVER_FAILED
        assert result.terminal_event is False
        assert result.landed is False
        assert result.carry_distance is None
        assert result.landing_angle is None
        assert result.lateral_deviation is None

        with pytest.raises(IncompleteFlightError) as exc_info:
            result.require_landing()
        assert exc_info.value.termination is FlightTermination.SOLVER_FAILED

    def test_cancelled_fixture(self) -> None:
        """Cooperative cancellation raises FlightSimulationCancelled carrying partial result."""
        model = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)

        step_count = 0

        def should_cancel() -> bool:
            nonlocal step_count
            step_count += 1
            return step_count >= 3

        with pytest.raises(FlightSimulationCancelled) as exc_info:
            model.simulate(
                _normal_launch(),
                max_time=15.0,
                cancellation_requested=should_cancel,
            )

        err = exc_info.value
        assert err.result is not None
        partial = err.result
        assert partial.termination is FlightTermination.CANCELLED
        assert partial.terminal_event is False
        assert partial.landed is False
        assert partial.carry_distance is None
        assert partial.landing_angle is None
        assert partial.lateral_deviation is None
        assert partial.flight_time > 0.0


class TestFlightResultContractValidation:
    """Enforcement of landing metrics constraints and invariants."""

    def test_rejects_non_none_metrics_on_unlanded_flight(self) -> None:
        points = [
            TrajectoryPoint(0.0, np.array([0.0, 0.0, 1.0]), np.array([10.0, 0.0, 0.0])),
        ]
        with pytest.raises(ValueError, match="Landing metrics.*must be None"):
            FlightResult(
                trajectory=points,
                model_name="test",
                carry_distance=100.0,
                termination=FlightTermination.TIME_LIMIT,
            )

    def test_default_landed_flight_allows_numeric_metrics(self) -> None:
        points = [
            TrajectoryPoint(0.0, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])),
        ]
        result = FlightResult(
            trajectory=points,
            model_name="test",
            carry_distance=0.0,
            landing_angle=0.0,
            lateral_deviation=0.0,
            termination=FlightTermination.LANDED,
        )
        assert result.landed is True
        assert result.carry_distance == 0.0


class TestProviderToApiToViewerContract:
    """End-to-end integration across Provider -> API -> Viewer."""

    def test_api_simulation_propagates_unlanded_termination(self) -> None:
        request = BallFlightSimulationRequest.model_validate(
            {
                "ball_speed_mps": 70.0,
                "launch_angle_deg": 12.0,
                "max_time_s": 0.1,  # Incomplete flight
                "time_step_s": 0.05,
                "model_name": FlightModelType.WATERLOO_PENNER,
            }
        )
        res = _simulate_one(FlightModelType.WATERLOO_PENNER, request)
        assert res.summary.termination == "time_limit"
        assert res.summary.terminal_event is False
        assert res.summary.landed is False
        assert res.summary.carry_m is None
        assert res.summary.landing_angle_deg is None
        assert res.summary.lateral_deviation_m is None
        assert res.summary.flight_time_s == pytest.approx(0.1, abs=0.02)
        assert res.summary.actual_horizon_s == pytest.approx(0.1, abs=0.02)

    def test_api_simulation_propagates_landed_termination(self) -> None:
        request = BallFlightSimulationRequest.model_validate(
            {
                "ball_speed_mps": 70.0,
                "launch_angle_deg": 12.0,
                "max_time_s": 15.0,  # Full flight
                "time_step_s": 0.1,
                "model_name": FlightModelType.WATERLOO_PENNER,
            }
        )
        res = _simulate_one(FlightModelType.WATERLOO_PENNER, request)
        assert res.summary.termination == "landed"
        assert res.summary.terminal_event is True
        assert res.summary.landed is True
        assert res.summary.carry_m is not None and res.summary.carry_m > 150.0
        assert (
            res.summary.landing_angle_deg is not None
            and res.summary.landing_angle_deg > 0.0
        )
        assert res.summary.lateral_deviation_m is not None

    def test_imported_trajectory_unlanded_gates_metrics(self) -> None:
        """Imported trajectory ending above ground gates carry/landing metrics to None."""
        sample1 = ImportedTrajectorySample(0.0, (0.0, 0.0, 1.0), None)
        sample2 = ImportedTrajectorySample(0.5, (30.0, 0.0, 15.0), None)
        sample3 = ImportedTrajectorySample(
            1.0, (60.0, 0.0, 10.0), None
        )  # ends at z=10.0m
        traj = ImportedBallFlightTrajectory(
            source_id="test",
            model_family="test",
            model_name="test",
            parameter_digest="0" * 64,
            frame_id="flight_xfwd_yleft_zup",
            samples=(sample1, sample2, sample3),
        )
        summary = summarize_imported_trajectory(traj)
        assert summary.landed is False
        assert summary.terminal_event is False
        assert summary.termination == "time_limit"
        assert summary.carry_m is None
        assert summary.landing_angle_deg is None
        assert summary.lateral_deviation_m is None
        assert summary.actual_horizon_s == pytest.approx(1.0)

    def test_viewer_results_table_formatting_with_none_metrics(self) -> None:
        """Shot Tracer table formatting handles None carry and landing angle without TypeError."""
        # Unlanded result
        incomplete_points = [
            TrajectoryPoint(0.0, np.array([0.0, 0.0, 1.0]), np.array([10.0, 0.0, 0.0])),
        ]
        unlanded = FlightResult(
            trajectory=incomplete_points,
            model_name="Unlanded",
            carry_distance=None,
            max_height=12.5,
            flight_time=0.1,
            landing_angle=None,
            lateral_deviation=None,
            termination=FlightTermination.TIME_LIMIT,
            terminal_event=False,
            actual_horizon=0.1,
        )

        # Landed result
        landed_points = [
            TrajectoryPoint(0.0, np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])),
            TrajectoryPoint(
                5.0, np.array([200.0, 2.0, 0.0]), np.array([10.0, 0.0, -10.0])
            ),
        ]
        landed = FlightResult(
            trajectory=landed_points,
            model_name="Landed",
            carry_distance=200.0,
            max_height=30.0,
            flight_time=5.0,
            landing_angle=45.0,
            lateral_deviation=2.0,
            termination=FlightTermination.LANDED,
            terminal_event=True,
            actual_horizon=5.0,
        )

        # Mirror the formatting logic in _shot_tracer_gui._update_results_table
        results = {"Unlanded": unlanded, "Landed": landed}
        table_rows = []
        for model_name, res in results.items():
            carry_yd_str = (
                f"{res.carry_distance * 1.09361:.1f}"
                if res.carry_distance is not None
                else "N/A"
            )
            landing_str = (
                f"{res.landing_angle:.1f}" if res.landing_angle is not None else "N/A"
            )
            table_rows.append(
                (
                    model_name,
                    carry_yd_str,
                    f"{res.max_height:.1f}",
                    f"{res.flight_time:.2f}",
                    landing_str,
                )
            )

        assert table_rows[0] == ("Unlanded", "N/A", "12.5", "0.10", "N/A")
        assert table_rows[1] == ("Landed", "218.7", "30.0", "5.00", "45.0")
