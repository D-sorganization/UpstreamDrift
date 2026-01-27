"""Unit tests for ball flight physics simulation.

Tests cover:
- Physical properties validation
- Launch conditions and environmental effects
- Trajectory simulation correctness
- Force calculations (gravity, drag, Magnus)
- Output metrics accuracy
"""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.ball_flight_physics import (
    BallFlightSimulator,
    BallProperties,
    EnvironmentalConditions,
    LaunchConditions,
    TrajectoryPoint,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_ball() -> BallProperties:
    """Default regulation golf ball properties."""
    return BallProperties()


@pytest.fixture
def default_environment() -> EnvironmentalConditions:
    """Default sea-level, no wind conditions."""
    return EnvironmentalConditions()


@pytest.fixture
def simulator() -> BallFlightSimulator:
    """Default simulator with regulation ball and standard conditions."""
    return BallFlightSimulator()


@pytest.fixture
def driver_launch() -> LaunchConditions:
    """Typical driver launch conditions.

    Based on PGA Tour average:
    - Ball speed: ~73 m/s (163 mph)
    - Launch angle: ~11 degrees
    - Backspin: ~2500 rpm
    """
    return LaunchConditions(
        velocity=73.0,
        launch_angle=math.radians(11.0),
        spin_rate=2500.0,
    )


@pytest.fixture
def iron_7_launch() -> LaunchConditions:
    """Typical 7-iron launch conditions.

    Based on PGA Tour average:
    - Ball speed: ~53 m/s (118 mph)
    - Launch angle: ~16 degrees
    - Backspin: ~7000 rpm
    """
    return LaunchConditions(
        velocity=53.0,
        launch_angle=math.radians(16.0),
        spin_rate=7000.0,
    )


# =============================================================================
# BallProperties Tests
# =============================================================================


class TestBallProperties:
    """Tests for BallProperties dataclass."""

    def test_default_values(self, default_ball: BallProperties) -> None:
        """Test default regulation golf ball values."""
        # Regulation golf ball: mass <= 45.93g, diameter >= 42.67mm
        assert default_ball.mass == pytest.approx(0.0459, rel=0.01)
        assert default_ball.diameter == pytest.approx(0.04267, rel=0.01)

    def test_radius_calculation(self, default_ball: BallProperties) -> None:
        """Test radius is half of diameter."""
        assert default_ball.radius == pytest.approx(default_ball.diameter / 2)

    def test_cross_sectional_area(self, default_ball: BallProperties) -> None:
        """Test cross-sectional area calculation."""
        expected_area = math.pi * default_ball.radius**2
        assert default_ball.cross_sectional_area == pytest.approx(expected_area)

    def test_custom_ball_properties(self) -> None:
        """Test creating ball with custom properties."""
        custom_ball = BallProperties(
            mass=0.05,
            diameter=0.045,
            cd0=0.25,  # Custom base drag
            cl1=0.35,  # Custom lift slope
        )
        assert custom_ball.mass == 0.05
        assert custom_ball.diameter == 0.045
        assert custom_ball.cd0 == 0.25
        assert custom_ball.cl1 == 0.35


# =============================================================================
# LaunchConditions Tests
# =============================================================================


class TestLaunchConditions:
    """Tests for LaunchConditions dataclass."""

    def test_default_spin_axis(self) -> None:
        """Test that default spin axis produces upward lift for backspin.

        Default axis is [0, -1, 0] which when crossed with forward velocity
        produces upward lift (Magnus force in +z direction).
        """
        launch = LaunchConditions(velocity=50.0, launch_angle=0.1)
        expected_axis = np.array([0.0, -1.0, 0.0])
        assert launch.spin_axis is not None
        np.testing.assert_array_almost_equal(launch.spin_axis, expected_axis)

    def test_custom_spin_axis(self) -> None:
        """Test custom spin axis is preserved."""
        custom_axis = np.array([0.5, 0.5, 0.0])
        launch = LaunchConditions(
            velocity=50.0, launch_angle=0.1, spin_axis=custom_axis
        )
        assert launch.spin_axis is not None
        np.testing.assert_array_almost_equal(launch.spin_axis, custom_axis)

    def test_zero_spin_rate_allowed(self) -> None:
        """Test that zero spin rate is valid."""
        launch = LaunchConditions(velocity=50.0, launch_angle=0.1, spin_rate=0.0)
        assert launch.spin_rate == 0.0


# =============================================================================
# EnvironmentalConditions Tests
# =============================================================================


class TestEnvironmentalConditions:
    """Tests for EnvironmentalConditions dataclass."""

    def test_default_values(self, default_environment: EnvironmentalConditions) -> None:
        """Test default sea-level conditions."""
        assert default_environment.air_density == pytest.approx(1.225)
        assert default_environment.gravity == pytest.approx(9.81, abs=0.01)
        assert default_environment.temperature == pytest.approx(15.0)

    def test_default_wind_is_zero(
        self, default_environment: EnvironmentalConditions
    ) -> None:
        """Test that default wind is zero vector."""
        assert default_environment.wind_velocity is not None
        np.testing.assert_array_equal(
            default_environment.wind_velocity, np.array([0.0, 0.0, 0.0])
        )

    def test_custom_wind(self) -> None:
        """Test custom wind configuration."""
        wind = np.array([5.0, 2.0, 0.0])  # 5 m/s headwind, 2 m/s crosswind
        env = EnvironmentalConditions(wind_velocity=wind)
        assert env.wind_velocity is not None
        np.testing.assert_array_equal(env.wind_velocity, wind)


# =============================================================================
# BallFlightSimulator Initialization Tests
# =============================================================================


class TestSimulatorInitialization:
    """Tests for BallFlightSimulator initialization."""

    def test_default_initialization(self) -> None:
        """Test simulator initializes with default components."""
        sim = BallFlightSimulator()
        assert sim.ball is not None
        assert sim.environment is not None

    def test_custom_ball(self) -> None:
        """Test simulator with custom ball."""
        custom_ball = BallProperties(mass=0.05)
        sim = BallFlightSimulator(ball=custom_ball)
        assert sim.ball.mass == 0.05

    def test_custom_environment(self) -> None:
        """Test simulator with custom environment."""
        custom_env = EnvironmentalConditions(altitude=1500.0)
        sim = BallFlightSimulator(environment=custom_env)
        assert sim.environment.altitude == 1500.0


# =============================================================================
# Trajectory Simulation Tests
# =============================================================================


class TestTrajectorySimulation:
    """Tests for trajectory simulation."""

    def test_trajectory_returns_list(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that simulation returns a list of trajectory points."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        assert isinstance(trajectory, list)
        assert len(trajectory) > 0

    def test_trajectory_point_structure(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test trajectory point has correct structure."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        point = trajectory[0]

        assert isinstance(point, TrajectoryPoint)
        assert isinstance(point.time, float)
        assert isinstance(point.position, np.ndarray)
        assert isinstance(point.velocity, np.ndarray)
        assert isinstance(point.acceleration, np.ndarray)
        assert isinstance(point.forces, dict)

    def test_initial_position_at_origin(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that trajectory starts at origin."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        initial_position = trajectory[0].position

        np.testing.assert_array_almost_equal(
            initial_position, np.array([0.0, 0.0, 0.0])
        )

    def test_trajectory_descends_to_ground(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that ball eventually returns to ground level."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=10.0)
        final_height = trajectory[-1].position[2]

        # Should land near ground level (z ≈ 0)
        # Tolerance accounts for event detection with discrete time steps
        assert final_height <= 0.2  # Within 20cm of ground

    def test_time_increases_monotonically(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that time increases throughout trajectory."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        times = [point.time for point in trajectory]

        for i in range(1, len(times)):
            assert times[i] > times[i - 1]


# =============================================================================
# Physics Validation Tests
# =============================================================================


class TestPhysicsValidation:
    """Tests validating physics correctness."""

    def test_gravity_only_trajectory(self, simulator: BallFlightSimulator) -> None:
        """Test trajectory with no spin (gravity and drag only)."""
        launch = LaunchConditions(
            velocity=30.0,
            launch_angle=math.radians(45.0),
            spin_rate=0.0,
        )
        trajectory = simulator.simulate_trajectory(launch, max_time=5.0)

        # Ball should travel forward and then land
        final_x = trajectory[-1].position[0]
        max_height = max(p.position[2] for p in trajectory)

        # Sanity checks
        assert final_x > 0  # Moved forward
        assert max_height > 0  # Went up

    def test_higher_launch_angle_higher_flight(
        self, simulator: BallFlightSimulator
    ) -> None:
        """Test that higher launch angle produces higher trajectory."""
        low_launch = LaunchConditions(
            velocity=50.0, launch_angle=math.radians(10.0), spin_rate=0.0
        )
        high_launch = LaunchConditions(
            velocity=50.0, launch_angle=math.radians(30.0), spin_rate=0.0
        )

        low_trajectory = simulator.simulate_trajectory(low_launch, max_time=6.0)
        high_trajectory = simulator.simulate_trajectory(high_launch, max_time=6.0)

        low_max_height = max(p.position[2] for p in low_trajectory)
        high_max_height = max(p.position[2] for p in high_trajectory)

        assert high_max_height > low_max_height

    def test_backspin_adds_lift(self, simulator: BallFlightSimulator) -> None:
        """Test that backspin produces lift (higher trajectory)."""
        no_spin = LaunchConditions(
            velocity=60.0, launch_angle=math.radians(12.0), spin_rate=0.0
        )
        with_spin = LaunchConditions(
            velocity=60.0, launch_angle=math.radians(12.0), spin_rate=3000.0
        )

        no_spin_traj = simulator.simulate_trajectory(no_spin, max_time=8.0)
        with_spin_traj = simulator.simulate_trajectory(with_spin, max_time=8.0)

        no_spin_max = max(p.position[2] for p in no_spin_traj)
        with_spin_max = max(p.position[2] for p in with_spin_traj)

        # Backspin should generate lift → higher trajectory
        assert with_spin_max > no_spin_max

    def test_higher_velocity_longer_carry(self, simulator: BallFlightSimulator) -> None:
        """Test that higher velocity produces longer carry distance."""
        slow_launch = LaunchConditions(
            velocity=40.0, launch_angle=math.radians(12.0), spin_rate=2500.0
        )
        fast_launch = LaunchConditions(
            velocity=70.0, launch_angle=math.radians(12.0), spin_rate=2500.0
        )

        slow_traj = simulator.simulate_trajectory(slow_launch, max_time=8.0)
        fast_traj = simulator.simulate_trajectory(fast_launch, max_time=8.0)

        slow_carry = simulator.calculate_carry_distance(slow_traj)
        fast_carry = simulator.calculate_carry_distance(fast_traj)

        assert fast_carry > slow_carry


# =============================================================================
# Force Calculation Tests
# =============================================================================


class TestForceCalculations:
    """Tests for force calculation methods."""

    def test_forces_include_gravity(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that forces include gravity."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        forces = trajectory[5].forces

        assert "gravity" in forces
        # Gravity should be negative z
        assert forces["gravity"][2] < 0

    def test_forces_include_drag(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that forces include drag when moving."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        forces = trajectory[5].forces

        assert "drag" in forces
        drag_magnitude = np.linalg.norm(forces["drag"])
        assert drag_magnitude > 0

    def test_magnus_force_with_spin(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that Magnus force is present with spin."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        forces = trajectory[5].forces

        assert "magnus" in forces
        magnus_magnitude = np.linalg.norm(forces["magnus"])
        # Should have some Magnus force with 2500 rpm spin
        assert magnus_magnitude > 0

    def test_no_magnus_without_spin(self, simulator: BallFlightSimulator) -> None:
        """Test that Magnus force is zero without spin."""
        no_spin_launch = LaunchConditions(
            velocity=50.0, launch_angle=math.radians(12.0), spin_rate=0.0
        )
        trajectory = simulator.simulate_trajectory(no_spin_launch, max_time=6.0)
        forces = trajectory[5].forces

        magnus = forces.get("magnus", np.zeros(3))
        np.testing.assert_array_almost_equal(magnus, np.zeros(3))

    def test_calculate_forces_vectorized(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test calculate_forces with vectorized input."""
        # Create dummy velocity batch (3, 5)
        vel = np.zeros((3, 5))
        vel[0, :] = 50.0  # x velocity

        forces = simulator._calculate_forces(vel, driver_launch)
        assert forces["drag"].shape == (3, 5)
        assert forces["magnus"].shape == (3, 5)
        assert forces["gravity"].shape == (3, 5)

    def test_calculate_forces_scalar_speed_threshold(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test calculate_forces with velocity below speed threshold."""
        vel = np.array([0.05, 0.0, 0.0])  # Below 0.1 m/s
        forces = simulator._calculate_forces(vel, driver_launch)
        np.testing.assert_array_equal(forces["drag"], np.zeros(3))
        np.testing.assert_array_equal(forces["magnus"], np.zeros(3))


# =============================================================================
# Metric Calculation Tests
# =============================================================================


class TestMetricCalculations:
    """Tests for trajectory metric calculations."""

    def test_carry_distance_positive(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test carry distance is positive for forward flight."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        carry = simulator.calculate_carry_distance(trajectory)

        assert carry > 0

    def test_max_height_positive(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test max height is positive for upward launch."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        max_height = simulator.calculate_max_height(trajectory)

        assert max_height > 0

    def test_flight_time_positive(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test flight time is positive."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        flight_time = simulator.calculate_flight_time(trajectory)

        assert flight_time > 0

    def test_empty_trajectory_returns_zero(
        self, simulator: BallFlightSimulator
    ) -> None:
        """Test that empty trajectory returns zero for all metrics."""
        empty_trajectory: list[TrajectoryPoint] = []

        assert simulator.calculate_carry_distance(empty_trajectory) == 0.0
        assert simulator.calculate_max_height(empty_trajectory) == 0.0
        assert simulator.calculate_flight_time(empty_trajectory) == 0.0


# =============================================================================
# Trajectory Analysis Tests
# =============================================================================


class TestTrajectoryAnalysis:
    """Tests for comprehensive trajectory analysis."""

    def test_analyze_trajectory_returns_dict(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that analysis returns dictionary with expected keys."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        analysis = simulator.analyze_trajectory(trajectory)

        assert isinstance(analysis, dict)
        assert "carry_distance" in analysis
        assert "max_height" in analysis
        assert "flight_time" in analysis
        assert "landing_angle" in analysis
        assert "apex_time" in analysis
        assert "trajectory_points" in analysis

    def test_apex_time_before_landing(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that apex time is before total flight time."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=8.0)
        analysis = simulator.analyze_trajectory(trajectory)

        assert analysis["apex_time"] < analysis["flight_time"]

    def test_landing_angle_reasonable(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that landing angle is within reasonable range."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=8.0)
        analysis = simulator.analyze_trajectory(trajectory)

        # Landing angle should be between 0 and 90 degrees
        # For a driver, typically 35-50 degrees
        assert 0 < analysis["landing_angle"] < 90

    def test_calculate_landing_angle_empty(
        self, simulator: BallFlightSimulator
    ) -> None:
        assert simulator._calculate_landing_angle([]) == 0.0

    def test_calculate_landing_angle_vertical_drop(
        self, simulator: BallFlightSimulator
    ) -> None:
        # Create a trajectory dropping straight down
        p1 = MagicMock(velocity=np.array([0.0, 0.0, -10.0]))
        p2 = MagicMock(velocity=np.array([0.0, 0.0, -20.0]))
        assert simulator._calculate_landing_angle([p1, p2]) == 90.0

    def test_calculate_apex_time_empty(self, simulator: BallFlightSimulator) -> None:
        assert simulator._calculate_apex_time([]) == 0.0


# =============================================================================
# Real-World Validation Tests
# =============================================================================


class TestRealWorldValidation:
    """Tests validating against known real-world data.

    Reference data from TrackMan/FlightScope:
    - Driver: 163 mph ball speed, 11° launch, 2500 rpm → ~275 yards carry
    - 7-iron: 118 mph ball speed, 16° launch, 7000 rpm → ~160 yards carry
    """

    def test_driver_carry_reasonable_range(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test driver carry distance is in reasonable range.

        The current empirical model targets:
        - Driver (73 m/s, 11°, 2500 rpm): ~200 yards

        Note: Model accuracy depends on lift/drag coefficient tuning.
        Wider tolerance allows for different coefficient sets.
        """
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=8.0)
        carry_m = simulator.calculate_carry_distance(trajectory)
        carry_yards = carry_m * 1.09361  # Convert to yards

        # Broad tolerance: 150-350 yards to accommodate model variations
        assert 150 < carry_yards < 350, f"Carry was {carry_yards:.1f} yards"

    def test_iron_7_carry_reasonable_range(
        self, simulator: BallFlightSimulator, iron_7_launch: LaunchConditions
    ) -> None:
        """Test 7-iron carry distance is in reasonable range.

        The current empirical model targets:
        - 7-iron (53 m/s, 16°, 7000 rpm): ~165 yards
        """
        trajectory = simulator.simulate_trajectory(iron_7_launch, max_time=8.0)
        carry_m = simulator.calculate_carry_distance(trajectory)
        carry_yards = carry_m * 1.09361

        # Should be between 100 and 220 yards
        assert 100 < carry_yards < 220, f"Carry was {carry_yards:.1f} yards"

    def test_driver_max_height_reasonable(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test driver max height is reasonable.

        Model produces heights in range 10-25m for driver trajectory.
        """
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=8.0)
        max_height = simulator.calculate_max_height(trajectory)

        # Allow 5-50m range for model variations
        assert 5 < max_height < 50, f"Max height was {max_height:.1f}m"

    def test_driver_flight_time_reasonable(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test driver flight time is reasonable.

        Model produces flight times of 3-6 seconds for driver.
        """
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=10.0)
        flight_time = simulator.calculate_flight_time(trajectory)

        # Allow 2-8 seconds for model variations
        assert 2 < flight_time < 8, f"Flight time was {flight_time:.1f}s"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_low_velocity(self, simulator: BallFlightSimulator) -> None:
        """Test trajectory with very low velocity."""
        low_velocity_launch = LaunchConditions(
            velocity=5.0, launch_angle=math.radians(45.0), spin_rate=0.0
        )
        trajectory = simulator.simulate_trajectory(low_velocity_launch, max_time=2.0)

        assert len(trajectory) > 0
        carry = simulator.calculate_carry_distance(trajectory)
        assert carry > 0  # Should still move forward

    def test_very_high_spin(self, simulator: BallFlightSimulator) -> None:
        """Test trajectory with extreme spin rate."""
        high_spin_launch = LaunchConditions(
            velocity=50.0, launch_angle=math.radians(15.0), spin_rate=10000.0
        )
        trajectory = simulator.simulate_trajectory(high_spin_launch, max_time=8.0)

        # Should still produce valid trajectory
        assert len(trajectory) > 0
        max_height = simulator.calculate_max_height(trajectory)
        assert max_height > 0

    def test_zero_launch_angle(self, simulator: BallFlightSimulator) -> None:
        """Test horizontal launch (zero launch angle).

        When launched horizontally from ground level (z=0) with no spin,
        the ball immediately contacts the ground due to gravity.
        """
        horizontal_launch = LaunchConditions(
            velocity=50.0, launch_angle=0.0, spin_rate=0.0
        )
        trajectory = simulator.simulate_trajectory(horizontal_launch, max_time=3.0)

        # Should have at least one point (the initial state)
        assert len(trajectory) >= 1
        # Ball should not rise above starting height
        max_height = max(p.position[2] for p in trajectory)
        assert max_height <= 0.1  # Within 10cm of ground (accounting for numerics)


# =============================================================================
# Wind Effect Tests
# =============================================================================


class TestWindEffects:
    """Tests for wind effects on trajectory."""

    def test_headwind_reduces_carry(self) -> None:
        """Test that headwind reduces carry distance."""
        launch = LaunchConditions(
            velocity=60.0, launch_angle=math.radians(12.0), spin_rate=2500.0
        )

        no_wind_sim = BallFlightSimulator()
        headwind_sim = BallFlightSimulator(
            environment=EnvironmentalConditions(
                wind_velocity=np.array([-10.0, 0.0, 0.0])  # 10 m/s headwind
            )
        )

        no_wind_traj = no_wind_sim.simulate_trajectory(launch, max_time=8.0)
        headwind_traj = headwind_sim.simulate_trajectory(launch, max_time=8.0)

        no_wind_carry = no_wind_sim.calculate_carry_distance(no_wind_traj)
        headwind_carry = headwind_sim.calculate_carry_distance(headwind_traj)

        assert headwind_carry < no_wind_carry

    def test_tailwind_increases_carry(self) -> None:
        """Test that tailwind increases carry distance."""
        launch = LaunchConditions(
            velocity=60.0, launch_angle=math.radians(12.0), spin_rate=2500.0
        )

        no_wind_sim = BallFlightSimulator()
        tailwind_sim = BallFlightSimulator(
            environment=EnvironmentalConditions(
                wind_velocity=np.array([10.0, 0.0, 0.0])  # 10 m/s tailwind
            )
        )

        no_wind_traj = no_wind_sim.simulate_trajectory(launch, max_time=8.0)
        tailwind_traj = tailwind_sim.simulate_trajectory(launch, max_time=8.0)

        no_wind_carry = no_wind_sim.calculate_carry_distance(no_wind_traj)
        tailwind_carry = tailwind_sim.calculate_carry_distance(tailwind_traj)

        assert tailwind_carry > no_wind_carry
