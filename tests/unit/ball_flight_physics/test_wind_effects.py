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


# =============================================================================
# Real-World Validation Tests
# =============================================================================


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


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


# =============================================================================
# Spin Decay Tests
# =============================================================================
