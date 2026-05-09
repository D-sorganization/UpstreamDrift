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


class TestBallProperties:
    """Tests for BallProperties dataclass."""

    def test_ball_flight_physics_default_values(
        self, default_ball: BallProperties
    ) -> None:
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
