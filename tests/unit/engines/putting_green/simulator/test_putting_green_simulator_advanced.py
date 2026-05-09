"""Unit tests for PuttingGreenSimulator module.

TDD Tests - These tests define the expected behavior of the main
putting green simulator engine that implements the PhysicsEngine protocol.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.simulator import (
    PuttingGreenSimulator,
    SimulationConfig,
    SimulationResult,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)


class TestPuttingGreenSimulatorAdvanced:
    """Advanced feature tests."""

    pytestmark = pytest.mark.slow

    @pytest.fixture
    def simulator(self) -> PuttingGreenSimulator:
        green = GreenSurface(
            width=20.0,
            height=20.0,
            turf=TurfProperties.create_preset("tournament_fast"),
        )
        return PuttingGreenSimulator(green=green)

    def test_multiple_ball_simulation(self, simulator: PuttingGreenSimulator) -> None:
        """Should support simulating multiple balls (scatter analysis)."""
        start_pos = np.array([5.0, 10.0])
        stroke_params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        results = simulator.simulate_scatter(
            start_pos,
            stroke_params,
            n_simulations=10,
            speed_variance=0.1,
            direction_variance_deg=2.0,
        )

        assert len(results) == 10
        # Should have variation in final positions
        final_positions = [r.final_position for r in results]
        positions_array = np.array(final_positions)
        variance = np.var(positions_array, axis=0)
        assert np.any(variance > 0)

    def test_scatter_is_deterministic_with_seed(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        """Scatter analysis should be reproducible with fixed RNG seed."""
        start_pos = np.array([5.0, 10.0])
        stroke_params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        seed = 123
        rng_one = np.random.default_rng(seed)
        rng_two = np.random.default_rng(seed)

        results_one = simulator.simulate_scatter(
            start_pos,
            stroke_params,
            n_simulations=5,
            speed_variance=0.1,
            direction_variance_deg=2.0,
            rng=rng_one,
        )
        results_two = simulator.simulate_scatter(
            start_pos,
            stroke_params,
            n_simulations=5,
            speed_variance=0.1,
            direction_variance_deg=2.0,
            rng=rng_two,
        )

        final_one = np.array([result.final_position for result in results_one])
        final_two = np.array([result.final_position for result in results_two])

        assert np.allclose(final_one, final_two)

    def test_aim_assist(self, simulator: PuttingGreenSimulator) -> None:
        """Should provide aim assist for breaking putts."""
        simulator.green.set_hole_position(np.array([15.0, 10.0]))
        simulator.green.add_slope_region(
            SlopeRegion(
                center=np.array([10.0, 10.0]),
                radius=8.0,
                slope_direction=np.array([0.0, 1.0]),  # Left-to-right break
                slope_magnitude=0.02,
            )
        )

        ball_pos = np.array([5.0, 10.0])
        aim_line = simulator.compute_aim_line(ball_pos)

        # Aim should be above the hole to account for break
        assert aim_line["aim_point"][1] < 10.0  # Aim left of hole

    def test_read_green(self, simulator: PuttingGreenSimulator) -> None:
        """Should provide green reading (slope analysis)."""
        ball_pos = np.array([5.0, 10.0])
        target = np.array([15.0, 10.0])

        reading = simulator.read_green(ball_pos, target)

        assert "total_break" in reading
        assert "recommended_speed" in reading
        assert "aim_point" in reading

    def test_practice_mode(self, simulator: PuttingGreenSimulator) -> None:
        """Should have practice mode with immediate feedback."""
        simulator.enable_practice_mode()
        simulator.set_ball_position(np.array([5.0, 10.0]))
        simulator.green.set_hole_position(np.array([15.0, 10.0]))

        stroke_params = StrokeParameters(
            speed=2.5,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        feedback = simulator.simulate_with_feedback(stroke_params)

        assert "distance_from_hole" in feedback
        assert "suggested_adjustment" in feedback

    def test_wind_effect(self, simulator: PuttingGreenSimulator) -> None:
        """Should optionally simulate wind effect."""
        simulator.set_wind(speed=5.0, direction=np.array([1.0, 0.0]))  # m/s
        simulator.set_ball_position(np.array([5.0, 10.0]))

        stroke_no_wind = StrokeParameters(
            speed=1.5,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        # Without wind
        simulator.set_wind(speed=0.0, direction=np.array([1.0, 0.0]))
        result_no_wind = simulator.simulate_putt(stroke_no_wind)

        # With headwind
        simulator.set_wind(speed=5.0, direction=np.array([-1.0, 0.0]))
        simulator.set_ball_position(np.array([5.0, 10.0]))
        result_headwind = simulator.simulate_putt(stroke_no_wind)

        # Headwind should result in shorter distance
        assert result_headwind.total_distance < result_no_wind.total_distance

    def test_replay_simulation(self, simulator: PuttingGreenSimulator) -> None:
        """Should be able to replay a simulation."""
        simulator.set_ball_position(np.array([5.0, 10.0]))
        stroke_params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        original_result = simulator.simulate_putt(stroke_params)

        # Replay should give same result
        simulator.set_ball_position(np.array([5.0, 10.0]))
        replay_result = simulator.simulate_putt(stroke_params)

        assert np.allclose(
            original_result.final_position, replay_result.final_position, atol=1e-6
        )
