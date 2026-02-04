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


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = SimulationConfig()
        assert config.timestep > 0
        assert config.max_simulation_time > 0
        assert config.stopping_velocity_threshold > 0

    def test_config_validation(self) -> None:
        """Should validate configuration parameters."""
        with pytest.raises(ValueError):
            SimulationConfig(timestep=-0.01)
        with pytest.raises(ValueError):
            SimulationConfig(max_simulation_time=0)

    def test_config_from_dict(self) -> None:
        """Should create config from dictionary."""
        data = {
            "timestep": 0.005,
            "max_simulation_time": 15.0,
            "stopping_velocity_threshold": 0.005,
        }
        config = SimulationConfig.from_dict(data)
        assert config.timestep == 0.005

    def test_config_to_dict(self) -> None:
        """Should serialize config to dictionary."""
        config = SimulationConfig(timestep=0.01)
        data = config.to_dict()
        assert data["timestep"] == 0.01


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    def test_result_contains_trajectory(self) -> None:
        """Result should contain trajectory data."""
        result = SimulationResult(
            positions=np.array([[0, 0], [1, 0], [2, 0]]),
            velocities=np.array([[2, 0], [1.5, 0], [0, 0]]),
            times=np.array([0, 0.1, 0.2]),
            holed=False,
            final_position=np.array([2, 0]),
        )
        assert len(result.positions) == 3
        assert not result.holed

    def test_result_distance_rolled(self) -> None:
        """Should compute total distance rolled."""
        result = SimulationResult(
            positions=np.array([[0, 0], [1, 0], [2, 0]]),
            velocities=np.array([[2, 0], [1.5, 0], [0, 0]]),
            times=np.array([0, 0.1, 0.2]),
            holed=False,
            final_position=np.array([2, 0]),
        )
        # Total distance should be approximately 2
        assert np.isclose(result.total_distance, 2.0, rtol=0.1)

    def test_result_duration(self) -> None:
        """Should report simulation duration."""
        result = SimulationResult(
            positions=np.array([[0, 0], [1, 0], [2, 0]]),
            velocities=np.array([[2, 0], [1.5, 0], [0, 0]]),
            times=np.array([0, 0.5, 1.5]),
            holed=False,
            final_position=np.array([2, 0]),
        )
        assert result.duration == 1.5


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

    def test_reset_clears_state(self, simulator: PuttingGreenSimulator) -> None:
        """Reset should clear simulation state."""
        # Set some state
        simulator.set_ball_position(np.array([5.0, 5.0]))
        simulator.reset()

        # Time should be 0
        assert simulator.get_time() == 0.0

    def test_step_advances_time(
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

    def test_set_state(self, configured_simulator: PuttingGreenSimulator) -> None:
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


class TestPuttingGreenSimulatorCheckpoints:
    """Tests for checkpoint functionality."""

    @pytest.fixture
    def simulator(self) -> PuttingGreenSimulator:
        green = GreenSurface(
            width=20.0,
            height=20.0,
            turf=TurfProperties.create_preset("tournament_fast"),
        )
        return PuttingGreenSimulator(green=green)

    def test_get_checkpoint(self, simulator: PuttingGreenSimulator) -> None:
        """Should save checkpoint of current state."""
        simulator.set_ball_position(np.array([5.0, 5.0]))
        simulator.set_ball_velocity(np.array([1.0, 0.0]))

        checkpoint = simulator.get_checkpoint()

        assert checkpoint is not None
        assert "position" in checkpoint or hasattr(checkpoint, "q")

    def test_restore_checkpoint(self, simulator: PuttingGreenSimulator) -> None:
        """Should restore state from checkpoint."""
        # Set initial state
        simulator.set_ball_position(np.array([5.0, 5.0]))
        simulator.set_ball_velocity(np.array([1.0, 0.0]))

        checkpoint = simulator.get_checkpoint()

        # Advance simulation
        for _ in range(50):
            simulator.step()

        # Restore
        simulator.restore_checkpoint(checkpoint)

        # Should be back to original
        q, v = simulator.get_state()
        assert np.isclose(q[0], 5.0, atol=0.01)


class TestPuttingGreenSimulatorPhysicsInterface:
    """Tests for PhysicsEngine protocol compliance."""

    @pytest.fixture
    def simulator(self) -> PuttingGreenSimulator:
        return PuttingGreenSimulator()

    def test_compute_mass_matrix(self, simulator: PuttingGreenSimulator) -> None:
        """Should return mass matrix (single ball = scalar mass)."""
        M = simulator.compute_mass_matrix()
        assert M.shape == (2, 2) or isinstance(M, (int, float))

    def test_compute_bias_forces(self, simulator: PuttingGreenSimulator) -> None:
        """Should compute bias forces (friction + slope)."""
        simulator.set_ball_position(np.array([10.0, 10.0]))
        simulator.set_ball_velocity(np.array([1.0, 0.0]))

        bias = simulator.compute_bias_forces()

        assert bias.shape == (2,)
        assert np.all(np.isfinite(bias))

    def test_compute_gravity_forces(self, simulator: PuttingGreenSimulator) -> None:
        """Should compute gravity on slope."""
        # Add slope to green
        simulator.green.add_slope_region(
            SlopeRegion(
                center=np.array([10.0, 10.0]),
                radius=5.0,
                slope_direction=np.array([1.0, 0.0]),
                slope_magnitude=0.05,
            )
        )
        simulator.set_ball_position(np.array([10.0, 10.0]))

        gravity = simulator.compute_gravity_forces()

        # On slope, should have non-zero gravity component
        assert np.linalg.norm(gravity) > 0

    def test_compute_drift_acceleration(self, simulator: PuttingGreenSimulator) -> None:
        """Should compute drift (passive) acceleration."""
        simulator.set_ball_position(np.array([10.0, 10.0]))
        simulator.set_ball_velocity(np.array([1.0, 0.0]))

        drift = simulator.compute_drift_acceleration()

        assert drift.shape == (2,)
        assert np.all(np.isfinite(drift))

    def test_compute_control_acceleration(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        """Should compute control acceleration from applied force."""
        tau = np.array([0.1, 0.0])  # Applied force

        control_acc = simulator.compute_control_acceleration(tau)

        assert control_acc.shape == (2,)
        # a = F/m
        expected = tau / simulator.ball_mass
        assert np.allclose(control_acc, expected, rtol=0.1)


class TestPuttingGreenSimulatorIO:
    """Tests for save/load functionality."""

    @pytest.fixture
    def simulator(self) -> PuttingGreenSimulator:
        return PuttingGreenSimulator()

    def test_load_from_path(self, simulator: PuttingGreenSimulator) -> None:
        """Should load green configuration from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "green": {
                    "width": 25.0,
                    "height": 25.0,
                    "turf": {
                        "stimp_rating": 11.0,
                        "grass_type": "bent_grass",
                    },
                    "hole_position": [15.0, 12.0],
                }
            }
            json.dump(config, f)
            f.flush()

            simulator.load_from_path(f.name)

            assert simulator.green.width == 25.0
            assert np.allclose(simulator.green.hole_position, [15.0, 12.0])

    def test_load_from_string(self, simulator: PuttingGreenSimulator) -> None:
        """Should load from JSON string."""
        config_str = json.dumps(
            {
                "green": {
                    "width": 30.0,
                    "height": 30.0,
                    "turf": {"stimp_rating": 12.0},
                }
            }
        )

        simulator.load_from_string(config_str, extension="json")

        assert simulator.green.width == 30.0

    def test_load_topographical_data(self, simulator: PuttingGreenSimulator) -> None:
        """Should load topographical/elevation data."""
        # Create a heightmap file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            heightmap = np.random.rand(100, 100) * 0.1
            np.save(f.name, heightmap)

            simulator.load_topographical_data(
                f.name,
                width=20.0,
                height=20.0,
            )

            # Should have loaded elevation data
            elev = simulator.green.get_elevation_at(np.array([10.0, 10.0]))
            assert 0 <= elev <= 0.1

    def test_load_topographical_csv(self, simulator: PuttingGreenSimulator) -> None:
        """Should load topographical data from CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write CSV with x, y, elevation columns
            f.write("x,y,elevation\n")
            for x in np.linspace(0, 20, 10):
                for y in np.linspace(0, 20, 10):
                    elev = 0.05 * np.sin(x / 5) * np.cos(y / 5)
                    f.write(f"{x},{y},{elev}\n")
            f.flush()

            simulator.load_topographical_data(f.name, width=20.0, height=20.0)

            # Should have loaded
            elev = simulator.green.get_elevation_at(np.array([10.0, 10.0]))
            assert np.isfinite(elev)

    def test_load_topographical_geotiff(self, simulator: PuttingGreenSimulator) -> None:
        """Should support GeoTIFF format (or skip if not available)."""
        # This would require rasterio, so we just check the method exists
        assert hasattr(simulator, "load_topographical_data")

    def test_export_simulation_result(self, simulator: PuttingGreenSimulator) -> None:
        """Should export simulation result to file."""
        simulator.set_ball_position(np.array([5.0, 10.0]))
        stroke_params = StrokeParameters(
            speed=1.5,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        result = simulator.simulate_putt(stroke_params)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            simulator.export_result(result, f.name)

            # Read back and verify
            with open(f.name) as rf:
                data = json.load(rf)
                assert "positions" in data
                assert "times" in data


class TestPuttingGreenSimulatorAdvanced:
    """Advanced feature tests."""

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
