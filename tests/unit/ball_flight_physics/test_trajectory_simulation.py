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


# =============================================================================
# Force Calculation Tests
# =============================================================================


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
