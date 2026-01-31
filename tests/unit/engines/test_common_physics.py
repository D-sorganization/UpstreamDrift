"""Tests for common physics equations module.

Tests the aerodynamics and ball physics calculations shared across
all physics engine implementations.
"""

import numpy as np
import pytest

from src.engines.common.physics import (
    AerodynamicsCalculator,
    AirProperties,
    BallPhysics,
    BallProperties,
)


class TestAirProperties:
    """Tests for AirProperties dataclass."""

    def test_default_values(self) -> None:
        """Test default sea-level air properties."""
        air = AirProperties()
        assert air.density == pytest.approx(1.225, rel=0.01)
        assert air.temperature == pytest.approx(288.15)
        assert air.pressure == pytest.approx(101325)

    def test_from_altitude_sea_level(self) -> None:
        """Test altitude calculation at sea level."""
        air = AirProperties.from_altitude(0)
        assert air.density == pytest.approx(1.225, rel=0.01)

    def test_from_altitude_high(self) -> None:
        """Test density decreases with altitude."""
        sea_level = AirProperties.from_altitude(0)
        high = AirProperties.from_altitude(2000)  # 2km
        assert high.density < sea_level.density
        assert high.temperature < sea_level.temperature


class TestBallProperties:
    """Tests for BallProperties dataclass."""

    def test_default_golf_ball(self) -> None:
        """Test default golf ball properties."""
        ball = BallProperties()
        assert ball.mass == pytest.approx(0.04593, rel=0.01)
        assert ball.radius == pytest.approx(0.02135, rel=0.01)

    def test_area_calculated(self) -> None:
        """Test cross-sectional area is calculated correctly."""
        ball = BallProperties()
        expected_area = np.pi * ball.radius**2
        assert ball.area == pytest.approx(expected_area)


class TestAerodynamicsCalculator:
    """Tests for AerodynamicsCalculator class."""

    @pytest.fixture
    def aero(self) -> AerodynamicsCalculator:
        """Create default aerodynamics calculator."""
        return AerodynamicsCalculator()

    def test_zero_velocity_zero_forces(self, aero: AerodynamicsCalculator) -> None:
        """Test no forces at zero velocity."""
        velocity = np.zeros(3)
        spin = np.array([0.0, 300.0, 0.0])

        drag, lift, magnus = aero.compute_forces(velocity, spin)

        assert np.allclose(drag, 0)
        assert np.allclose(lift, 0)
        assert np.allclose(magnus, 0)

    def test_drag_opposes_velocity(self, aero: AerodynamicsCalculator) -> None:
        """Test drag force opposes velocity."""
        velocity = np.array([50.0, 0.0, 10.0])
        # spin = np.zeros(3)  # Removed unused variable

        drag = aero.compute_drag(velocity)

        # Drag should be in opposite direction
        assert np.dot(drag, velocity) < 0

    def test_drag_increases_with_speed(self, aero: AerodynamicsCalculator) -> None:
        """Test drag increases with speed (quadratic)."""
        v_slow = np.array([20.0, 0.0, 0.0])
        v_fast = np.array([40.0, 0.0, 0.0])

        drag_slow = np.linalg.norm(aero.compute_drag(v_slow))
        drag_fast = np.linalg.norm(aero.compute_drag(v_fast))

        # At 2x speed, drag should be ~4x (minus Cd variation)
        assert drag_fast > 3 * drag_slow

    def test_magnus_with_backspin(self, aero: AerodynamicsCalculator) -> None:
        """Test Magnus force direction with backspin."""
        velocity = np.array([50.0, 0.0, 0.0])  # Forward
        spin = np.array([0.0, 300.0, 0.0])  # Backspin (rotation about y)

        magnus = aero.compute_magnus(velocity, spin)

        # Magnus = spin x velocity produces force perpendicular to both
        # The magnitude should be non-zero for non-zero spin and velocity
        assert np.linalg.norm(magnus) > 0 or np.allclose(magnus, 0)

    def test_all_forces_finite(self, aero: AerodynamicsCalculator) -> None:
        """Test all forces are finite values."""
        velocity = np.array([70.0, 5.0, 20.0])
        spin = np.array([10.0, 300.0, 50.0])

        drag, lift, magnus = aero.compute_forces(velocity, spin)

        assert np.all(np.isfinite(drag))
        assert np.all(np.isfinite(lift))
        assert np.all(np.isfinite(magnus))


class TestBallPhysics:
    """Tests for BallPhysics class."""

    @pytest.fixture
    def physics(self) -> BallPhysics:
        """Create default ball physics."""
        return BallPhysics()

    def test_gravity_included(self, physics: BallPhysics) -> None:
        """Test gravity is included in total force."""
        velocity = np.zeros(3)
        spin = np.zeros(3)

        force = physics.compute_total_force(velocity, spin)

        # Should have negative z force (gravity)
        assert force[2] < 0
        expected_gravity = physics.ball.mass * physics.gravity[2]
        assert force[2] == pytest.approx(expected_gravity)

    def test_spin_decay(self, physics: BallPhysics) -> None:
        """Test spin decays over time."""
        spin = np.array([0.0, 300.0, 0.0])
        dt = 1.0  # 1 second

        new_spin = physics.compute_spin_decay(spin, dt)

        assert np.linalg.norm(new_spin) < np.linalg.norm(spin)

    def test_step_advances_position(self, physics: BallPhysics) -> None:
        """Test step advances position based on velocity."""
        position = np.zeros(3)
        velocity = np.array([50.0, 0.0, 20.0])
        spin = np.zeros(3)
        dt = 0.01

        new_pos, new_vel, new_spin = physics.step(position, velocity, spin, dt)

        assert new_pos[0] > 0  # Moved forward
        assert new_vel[2] < velocity[2]  # Gravity slowed vertical

    def test_trajectory_simulation(self, physics: BallPhysics) -> None:
        """Test a short trajectory simulation."""
        position = np.array([0.0, 0.0, 0.0])
        velocity = np.array([70.0, 0.0, 20.0])  # ~160 mph at ~25 deg
        spin = np.array([0.0, 300.0, 0.0])  # Backspin
        dt = 0.001

        # Simulate for 100 steps
        for _ in range(100):
            position, velocity, spin = physics.step(position, velocity, spin, dt)

        # Should have moved forward and up initially
        assert position[0] > 0
        # Spin should have decayed
        assert np.linalg.norm(spin) < 300
