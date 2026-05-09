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
