"""Unit tests for the pure-Python parts of the H3 import module (#9352).

The vendored-reader path (frame resolution, unknown/missing-field
refusal, cross-family acceptance) is covered end-to-end through the API
in ``tests/api/test_routes_ball_flight.py``. This file covers the
logic that needs no vendored dependency: the summary derivation and
the dataclasses' own shape.
"""

from __future__ import annotations

import math

import pytest

from src.api.routes._ball_flight_trajectory_import import (
    ImportedBallFlightTrajectory,
    ImportedTrajectorySample,
    TrajectoryImportError,
    summarize_imported_trajectory,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _trajectory(
    samples: tuple[ImportedTrajectorySample, ...],
) -> ImportedBallFlightTrajectory:
    return ImportedBallFlightTrajectory(
        source_id="test:fixture",
        model_family="swing_sim.flight",
        model_name="Nathan",
        parameter_digest="a" * 64,
        frame_id="flight_xfwd_yleft_zup",
        samples=samples,
    )


class TestTrajectoryImportError:
    def test_is_a_value_error(self) -> None:
        """Composes with handle_api_errors' ValueError -> 400 mapping."""
        assert issubclass(TrajectoryImportError, ValueError)

    def test_message_is_preserved_verbatim(self) -> None:
        assert str(TrajectoryImportError("named reason")) == "named reason"


class TestSummarizeImportedTrajectory:
    def test_reports_final_position_as_carry_and_lateral(self) -> None:
        trajectory = _trajectory(
            (
                ImportedTrajectorySample(0.0, (0.0, 0.0, 0.0), None),
                ImportedTrajectorySample(1.0, (45.0, 2.0, 0.0), None),
            )
        )
        summary = summarize_imported_trajectory(trajectory)
        assert summary.carry_m == 45.0
        assert summary.lateral_deviation_m == 2.0
        assert summary.flight_time_s == 1.0

    def test_apex_is_the_maximum_retained_height(self) -> None:
        trajectory = _trajectory(
            (
                ImportedTrajectorySample(0.0, (0.0, 0.0, 0.0), None),
                ImportedTrajectorySample(0.5, (20.0, 0.0, 12.5), None),
                ImportedTrajectorySample(1.0, (45.0, 0.0, 0.0), None),
            )
        )
        summary = summarize_imported_trajectory(trajectory)
        assert summary.apex_m == 12.5

    def test_never_reads_the_optional_velocity_channel(self) -> None:
        """Works identically whether or not velocity_mps is present."""
        without_velocity = _trajectory(
            (
                ImportedTrajectorySample(0.0, (0.0, 0.0, 0.0), None),
                ImportedTrajectorySample(1.0, (45.0, 0.0, 0.0), None),
            )
        )
        with_velocity = _trajectory(
            (
                ImportedTrajectorySample(0.0, (0.0, 0.0, 0.0), (50.0, 0.0, 20.0)),
                ImportedTrajectorySample(1.0, (45.0, 0.0, 0.0), (44.0, 0.0, -8.0)),
            )
        )
        assert summarize_imported_trajectory(
            without_velocity
        ) == summarize_imported_trajectory(with_velocity)

    def test_landing_angle_is_zero_for_a_level_final_segment(self) -> None:
        trajectory = _trajectory(
            (
                ImportedTrajectorySample(0.0, (0.0, 0.0, 0.0), None),
                ImportedTrajectorySample(1.0, (45.0, 0.0, 0.0), None),
            )
        )
        summary = summarize_imported_trajectory(trajectory)
        assert summary.landing_angle_deg == 0.0

    def test_landing_angle_is_positive_for_a_descending_final_segment(self) -> None:
        trajectory = _trajectory(
            (
                ImportedTrajectorySample(0.0, (40.0, 0.0, 2.0), None),
                ImportedTrajectorySample(1.0, (45.0, 0.0, 0.0), None),
            )
        )
        summary = summarize_imported_trajectory(trajectory)
        assert summary.landing_angle_deg > 0.0
        assert math.isfinite(summary.landing_angle_deg)


class TestImportedTrajectorySample:
    def test_velocity_is_optional(self) -> None:
        sample = ImportedTrajectorySample(0.0, (1.0, 2.0, 3.0), None)
        assert sample.velocity_mps is None
