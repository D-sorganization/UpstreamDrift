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


# =============================================================================
# Metric Calculation Tests
# =============================================================================


# =============================================================================
# Trajectory Analysis Tests
# =============================================================================


class TestTrajectoryAnalysis:
    """Tests for comprehensive trajectory analysis."""

    @pytest.mark.parametrize(
        "expected_key",
        [
            "carry_distance",
            "max_height",
            "flight_time",
            "landing_angle",
            "apex_time",
            "trajectory_points",
        ],
    )
    def test_analyze_trajectory_contains_key(
        self,
        simulator: BallFlightSimulator,
        driver_launch: LaunchConditions,
        expected_key: str,
    ) -> None:
        """Test that analysis returns dictionary with expected key."""
        trajectory = simulator.simulate_trajectory(driver_launch, max_time=6.0)
        analysis = simulator.analyze_trajectory(trajectory)
        assert isinstance(analysis, dict)
        assert expected_key in analysis

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


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


# =============================================================================
# Wind Effect Tests
# =============================================================================


# =============================================================================
# Spin Decay Tests
# =============================================================================
