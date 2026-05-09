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


# =============================================================================
# Spin Decay Tests
# =============================================================================


class TestSpinDecay:
    """Tests for spin decay model in BallFlightSimulator."""

    def test_default_spin_decay_rate(self) -> None:
        """Test BallProperties has default spin decay rate from physics constants."""
        ball = BallProperties()
        assert ball.spin_decay_rate == pytest.approx(0.05, rel=0.01)

    def test_custom_spin_decay_rate(self) -> None:
        """Test custom spin decay rate can be set."""
        ball = BallProperties(spin_decay_rate=0.1)
        assert ball.spin_decay_rate == pytest.approx(0.1)

    def test_zero_spin_decay_rate(self) -> None:
        """Test spin decay can be disabled by setting rate to zero."""
        ball = BallProperties(spin_decay_rate=0.0)
        assert ball.spin_decay_rate == 0.0

    def test_spin_decay_reduces_carry(self) -> None:
        """Spin decay should reduce carry distance vs constant spin.

        With spin decaying, Magnus lift decreases over time, leading
        to a shorter carry than if spin were constant.
        """
        launch = LaunchConditions(
            velocity=73.0,
            launch_angle=math.radians(11.0),
            spin_rate=2500.0,
        )

        # Simulator with no spin decay (constant spin)
        no_decay = BallFlightSimulator(ball=BallProperties(spin_decay_rate=0.0))
        # Simulator with default spin decay
        with_decay = BallFlightSimulator(ball=BallProperties(spin_decay_rate=0.05))

        traj_no_decay = no_decay.simulate_trajectory(launch, max_time=8.0)
        traj_with_decay = with_decay.simulate_trajectory(launch, max_time=8.0)

        carry_no_decay = no_decay.calculate_carry_distance(traj_no_decay)
        carry_with_decay = with_decay.calculate_carry_distance(traj_with_decay)

        # Spin decay reduces lift, so carry should be shorter
        assert carry_with_decay < carry_no_decay

    def test_higher_decay_rate_less_carry(self) -> None:
        """Higher spin decay rate should produce shorter carry."""
        launch = LaunchConditions(
            velocity=53.0,
            launch_angle=math.radians(16.0),
            spin_rate=7000.0,
        )

        slow_decay = BallFlightSimulator(ball=BallProperties(spin_decay_rate=0.02))
        fast_decay = BallFlightSimulator(ball=BallProperties(spin_decay_rate=0.2))

        traj_slow = slow_decay.simulate_trajectory(launch, max_time=8.0)
        traj_fast = fast_decay.simulate_trajectory(launch, max_time=8.0)

        carry_slow = slow_decay.calculate_carry_distance(traj_slow)
        carry_fast = fast_decay.calculate_carry_distance(traj_fast)

        assert carry_fast < carry_slow

    def test_no_spin_no_decay_effect(self) -> None:
        """With zero initial spin, decay rate should not affect trajectory."""
        launch = LaunchConditions(
            velocity=50.0,
            launch_angle=math.radians(15.0),
            spin_rate=0.0,
        )

        no_decay = BallFlightSimulator(ball=BallProperties(spin_decay_rate=0.0))
        with_decay = BallFlightSimulator(ball=BallProperties(spin_decay_rate=0.1))

        traj_no = no_decay.simulate_trajectory(launch, max_time=6.0)
        traj_wd = with_decay.simulate_trajectory(launch, max_time=6.0)

        carry_no = no_decay.calculate_carry_distance(traj_no)
        carry_wd = with_decay.calculate_carry_distance(traj_wd)

        assert carry_no == pytest.approx(carry_wd, rel=1e-6)
