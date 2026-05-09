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
from src.shared.python.physics.rust_kernel import is_rust_available

pytestmark = pytest.mark.skipif(
    not is_rust_available(),
    reason="upstream-physics Rust kernel not installed (pip install upstream-drift[rust])",
)

from src.shared.python.physics.ball_flight_physics import (  # noqa: E402
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


# =============================================================================
# LaunchConditions Tests
# =============================================================================


# =============================================================================
# EnvironmentalConditions Tests
# =============================================================================


# =============================================================================
# BallFlightSimulator Initialization Tests
# =============================================================================


# =============================================================================
# Trajectory Simulation Tests
# =============================================================================


# =============================================================================
# Physics Validation Tests
# =============================================================================


# =============================================================================
# Force Calculation Tests
# =============================================================================


class TestForceCalculations:
    """Tests for force calculation methods."""

    @pytest.mark.parametrize(
        "force_name",
        ["gravity", "drag", "magnus"],
    )
    def test_forces_present_in_trajectory(
        self,
        simulator: BallFlightSimulator,
        driver_launch: LaunchConditions,
        force_name: str,
    ) -> None:
        """Test that expected force components are present in trajectory."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        forces = trajectory[5].forces
        assert force_name in forces
        magnitude = np.linalg.norm(forces[force_name])
        assert magnitude > 0

    def test_gravity_direction(
        self, simulator: BallFlightSimulator, driver_launch: LaunchConditions
    ) -> None:
        """Test that gravity force is in negative z direction."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        forces = trajectory[5].forces
        assert forces["gravity"][2] < 0

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


# =============================================================================
# Trajectory Analysis Tests
# =============================================================================


# =============================================================================
# Real-World Validation Tests
# =============================================================================


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


# =============================================================================
# Wind Effect Tests
# =============================================================================


# =============================================================================
# Spin Decay Tests
# =============================================================================
