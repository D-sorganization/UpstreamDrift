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


class TestPuttingGreenSimulator:
    """Tests for the main PuttingGreenSimulator class."""

    @pytest.fixture
    def simulator(self) -> PuttingGreenSimulator:
        """Create default simulator."""
        return PuttingGreenSimulator()

    @pytest.fixture
    def configured_simulator(self) -> PuttingGreenSimulator:
        """Create fully configured simulator."""
        config = SimulationConfig(timestep=0.001)
        turf = TurfProperties.create_preset("tournament_fast")
        green = GreenSurface(width=20.0, height=20.0, turf=turf)
        green.set_hole_position(np.array([15.0, 10.0]))

        return PuttingGreenSimulator(green=green, config=config)

    def test_simulator_creation(self, simulator: PuttingGreenSimulator) -> None:
        """Simulator should be created successfully."""
        assert simulator is not None

    def test_model_name_property(self, simulator: PuttingGreenSimulator) -> None:
        """Should return model name."""
        assert simulator.model_name == "putting_green"

    def test_simulator_reset_clears_state(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        """Reset should clear simulation state."""
        # Set some state
        simulator.set_ball_position(np.array([5.0, 5.0]))
        simulator.reset()

        # Time should be 0
        assert simulator.get_time() == 0.0

    def test_simulator_step_advances_time(
        self, configured_simulator: PuttingGreenSimulator
    ) -> None:
        """Step should advance simulation time."""
        configured_simulator.set_ball_position(np.array([5.0, 10.0]))
        configured_simulator.set_ball_velocity(np.array([2.0, 0.0]))

        initial_time = configured_simulator.get_time()
        configured_simulator.step()
        final_time = configured_simulator.get_time()

        assert final_time > initial_time

    def test_step_moves_ball(self, configured_simulator: PuttingGreenSimulator) -> None:
        """Step should move the ball."""
        configured_simulator.set_ball_position(np.array([5.0, 10.0]))
        configured_simulator.set_ball_velocity(np.array([2.0, 0.0]))

        initial_pos = configured_simulator.get_ball_position().copy()
        configured_simulator.step()
        final_pos = configured_simulator.get_ball_position()

        assert final_pos[0] > initial_pos[0]

    def test_forward_computes_kinematics(
        self, configured_simulator: PuttingGreenSimulator
    ) -> None:
        """Forward should compute kinematics without advancing time."""
        configured_simulator.set_ball_position(np.array([5.0, 10.0]))
        configured_simulator.set_ball_velocity(np.array([2.0, 0.0]))

        initial_time = configured_simulator.get_time()
        configured_simulator.forward()
        final_time = configured_simulator.get_time()

        # Time should not change
        assert final_time == initial_time
        assert configured_simulator.get_last_acceleration() is not None
        assert configured_simulator.get_last_roll_mode() is not None

    def test_get_state_returns_arrays(
        self, configured_simulator: PuttingGreenSimulator
    ) -> None:
        """get_state should return position and velocity arrays."""
        configured_simulator.set_ball_position(np.array([5.0, 10.0]))
        configured_simulator.set_ball_velocity(np.array([2.0, 1.0]))

        q, v = configured_simulator.get_state()

        assert isinstance(q, np.ndarray)
        assert isinstance(v, np.ndarray)
        assert q.shape == (2,)  # 2D position
        assert v.shape == (2,)  # 2D velocity

    def test_simulator_set_state(
        self, configured_simulator: PuttingGreenSimulator
    ) -> None:
        """set_state should update position and velocity."""
        q = np.array([8.0, 12.0])
        v = np.array([1.5, 0.5])

        configured_simulator.set_state(q, v)
        new_q, new_v = configured_simulator.get_state()

        assert np.allclose(new_q, q)
        assert np.allclose(new_v, v)

    def test_simulate_putt_returns_result(
        self, configured_simulator: PuttingGreenSimulator
    ) -> None:
        """simulate_putt should return SimulationResult."""
        configured_simulator.set_ball_position(np.array([5.0, 10.0]))

        stroke_params = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        result = configured_simulator.simulate_putt(stroke_params)

        assert isinstance(result, SimulationResult)
        assert len(result.positions) > 0

    def test_simulate_putt_ball_stops(
        self, configured_simulator: PuttingGreenSimulator
    ) -> None:
        """Ball should eventually stop."""
        configured_simulator.set_ball_position(np.array([5.0, 10.0]))

        stroke_params = StrokeParameters(
            speed=1.5,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        result = configured_simulator.simulate_putt(stroke_params)

        # Final velocity should be near zero
        final_vel = result.velocities[-1]
        assert np.linalg.norm(final_vel) < 0.01

    def test_detect_hole_in(self, configured_simulator: PuttingGreenSimulator) -> None:
        """Should detect when ball goes in hole."""
        # Ball position close to hole, aimed at hole
        configured_simulator.set_ball_position(np.array([14.0, 10.0]))

        stroke_params = StrokeParameters(
            speed=1.0,  # Gentle putt
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        result = configured_simulator.simulate_putt(stroke_params)

        assert result.holed

    def test_real_time_simulation_mode(
        self, configured_simulator: PuttingGreenSimulator
    ) -> None:
        """Should support real-time stepping mode."""
        configured_simulator.set_ball_position(np.array([5.0, 10.0]))
        configured_simulator.set_ball_velocity(np.array([2.0, 0.0]))

        # Enable real-time mode
        configured_simulator.set_real_time_mode(True)

        # Step should still work
        configured_simulator.step()
        pos = configured_simulator.get_ball_position()

        assert pos[0] > 5.0

    def test_get_trajectory_during_simulation(
        self, configured_simulator: PuttingGreenSimulator
    ) -> None:
        """Should be able to get partial trajectory during simulation."""
        configured_simulator.set_ball_position(np.array([5.0, 10.0]))
        configured_simulator.set_ball_velocity(np.array([2.0, 0.0]))

        # Run a few steps
        for _ in range(10):
            configured_simulator.step()

        trajectory = configured_simulator.get_current_trajectory()

        assert len(trajectory["positions"]) > 0
        assert len(trajectory["times"]) > 0
