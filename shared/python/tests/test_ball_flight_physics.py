import numpy as np

from shared.python.ball_flight_physics import (
    BallFlightSimulator,
    BallProperties,
    EnvironmentalConditions,
    LaunchConditions,
)


class TestBallFlightPhysics:
    def test_trajectory_simulation(self):
        """Test basic trajectory simulation."""
        simulator = BallFlightSimulator()
        launch = LaunchConditions(
            velocity=70.0,
            launch_angle=np.radians(12.0),
            spin_rate=3000.0,
            azimuth_angle=0.0,
        )

        trajectory = simulator.simulate_trajectory(launch)

        assert len(trajectory) > 0
        assert trajectory[0].time == 0.0
        assert np.allclose(trajectory[0].position, [0, 0, 0])

        # Check that it lands (height roughly 0 at end)
        assert abs(trajectory[-1].position[2]) < 0.5  # Approximate

        # Check metrics
        metrics = simulator.analyze_trajectory(trajectory)
        assert metrics["carry_distance"] > 100.0
        assert metrics["max_height"] > 10.0

    def test_no_spin_trajectory(self):
        """Test trajectory without spin."""
        simulator = BallFlightSimulator()
        launch = LaunchConditions(
            velocity=50.0, launch_angle=np.radians(45.0), spin_rate=0.0
        )

        trajectory = simulator.simulate_trajectory(launch)
        assert len(trajectory) > 0

    def test_spin_axis_effect(self):
        """Test that spin axis causes lateral movement."""
        simulator = BallFlightSimulator()

        launch = LaunchConditions(
            velocity=70.0,
            launch_angle=np.radians(10.0),
            spin_rate=3000.0,
            spin_axis=np.array([0.5, -1.0, 0.0]) / np.linalg.norm([0.5, -1.0, 0.0]),
        )

        trajectory = simulator.simulate_trajectory(launch)
        final_pos = trajectory[-1].position

        # Should have significant Y deviation
        assert abs(final_pos[1]) > 0.1

    def test_stop_condition(self):
        """Test that simulation stops at ground."""
        simulator = BallFlightSimulator()
        launch = LaunchConditions(velocity=10.0, launch_angle=np.radians(45.0))

        trajectory = simulator.simulate_trajectory(launch)
        # Z should be close to 0 at end
        assert trajectory[-1].position[2] <= 0.1

    def test_wind_effect(self):
        """Test that wind affects trajectory."""
        ball = BallProperties()

        # Headwind
        env_wind = EnvironmentalConditions(wind_velocity=np.array([-10.0, 0.0, 0.0]))
        sim_wind = BallFlightSimulator(ball, env_wind)

        # No wind
        env_calm = EnvironmentalConditions(wind_velocity=np.array([0.0, 0.0, 0.0]))
        sim_calm = BallFlightSimulator(ball, env_calm)

        launch = LaunchConditions(
            velocity=70.0, launch_angle=np.radians(10.0), spin_rate=2500.0
        )

        traj_wind = sim_wind.simulate_trajectory(launch)
        traj_calm = sim_calm.simulate_trajectory(launch)

        dist_wind = sim_wind.calculate_carry_distance(traj_wind)
        dist_calm = sim_calm.calculate_carry_distance(traj_calm)

        # Headwind should reduce distance
        assert dist_wind < dist_calm
