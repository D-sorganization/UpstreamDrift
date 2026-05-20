"""Integration tests for Putting Green Simulation.

These tests verify that all components work together correctly:
- TurfProperties + GreenSurface
- BallRollPhysics + GreenSurface
- PutterStroke + BallRollPhysics
- Full simulation end-to-end
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
    BallState,
)
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    PutterStroke,
    PutterType,
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.simulator import (
    PuttingGreenSimulator,
    SimulationConfig,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)


class TestEndToEndPutting:
    """End-to-end putting simulation tests."""

    @pytest.fixture
    def tournament_simulator(self) -> PuttingGreenSimulator:
        """Create tournament-level simulator."""
        turf = TurfProperties.create_preset("tournament_fast")
        green = GreenSurface(width=20.0, height=20.0, turf=turf)
        green.set_hole_position(np.array([15.0, 10.0]))
        config = SimulationConfig(timestep=0.001)
        return PuttingGreenSimulator(green=green, config=config)

    @pytest.fixture
    def sloped_simulator(self) -> PuttingGreenSimulator:
        """Create simulator with sloped green."""
        turf = TurfProperties.create_preset("tournament_fast")
        green = GreenSurface(width=20.0, height=20.0, turf=turf)
        green.add_slope_region(
            SlopeRegion(
                center=np.array([10.0, 10.0]),
                radius=8.0,
                slope_direction=np.array([0.0, 1.0]),  # Break left to right
                slope_magnitude=0.03,  # 3% slope
            )
        )
        green.set_hole_position(np.array([15.0, 10.0]))
        return PuttingGreenSimulator(green=green)

    def test_straight_putt_to_hole(
        self, tournament_simulator: PuttingGreenSimulator
    ) -> None:
        """Test a straight putt that should go in the hole."""
        tournament_simulator.set_ball_position(np.array([5.0, 10.0]))

        # Calculate required speed for 10m putt
        stroke = StrokeParameters.for_target_distance(
            distance=10.0,
            stimp_rating=tournament_simulator.green.turf.stimp_rating,
            direction=np.array([1.0, 0.0]),
        )

        result = tournament_simulator.simulate_putt(stroke)

        # Ball should reach near the hole (within reason for speed estimation)
        distance_from_hole = np.linalg.norm(
            result.final_position - np.array([15.0, 10.0])
        )
        assert (
            distance_from_hole < 1.0
        )  # Within 1 meter is reasonable, "Assertion failed: distance_from_hole < 1.0  # Within 1 meter is reasonable"

    def test_putt_with_break(self, sloped_simulator: PuttingGreenSimulator) -> None:
        """Test a putt on a sloped green has break."""
        sloped_simulator.set_ball_position(np.array([5.0, 10.0]))

        # Aim straight at hole (ignoring break)
        stroke = StrokeParameters(
            speed=2.5,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        result = sloped_simulator.simulate_putt(stroke)

        # Ball should have curved due to slope (y position changed)
        assert (
            result.final_position[1] != 10.0
        ), "Assertion failed: result.final_position[1] != 10.0"

    def test_ball_stops_eventually(
        self, tournament_simulator: PuttingGreenSimulator
    ) -> None:
        """Test that ball always stops within reasonable time."""
        tournament_simulator.set_ball_position(np.array([5.0, 10.0]))

        stroke = StrokeParameters(
            speed=4.0,  # Fast putt
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        result = tournament_simulator.simulate_putt(stroke)

        # Ball should have stopped (zero final velocity)
        final_speed = np.linalg.norm(result.velocities[-1])
        assert final_speed < 0.01, "Assertion failed: final_speed < 0.01"

        # Simulation should not have hit time limit
        assert result.duration < 20.0, "Assertion failed: result.duration < 20.0"

    def test_holed_putt_detection(
        self, tournament_simulator: PuttingGreenSimulator
    ) -> None:
        """Test detection of holed putt."""
        # Position close to hole
        tournament_simulator.set_ball_position(np.array([14.0, 10.0]))

        stroke = StrokeParameters(
            speed=0.8,  # Gentle tap
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        result = tournament_simulator.simulate_putt(stroke)

        # Should detect as holed
        assert result.holed, "Assertion failed: result.holed"

    def test_miss_putt_not_holed(
        self, tournament_simulator: PuttingGreenSimulator
    ) -> None:
        """Test that missed putt is not detected as holed."""
        tournament_simulator.set_ball_position(np.array([5.0, 5.0]))

        # Aim away from hole
        stroke = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        result = tournament_simulator.simulate_putt(stroke)

        assert not result.holed, "Assertion failed: not result.holed"
