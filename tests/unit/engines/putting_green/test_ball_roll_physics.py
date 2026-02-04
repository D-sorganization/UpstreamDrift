"""Unit tests for BallRollPhysics module.

TDD Tests - These tests define the expected behavior of the ball rolling
physics including sliding, rolling, spin decay, and energy conservation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
    BallState,
    RollMode,
)
from src.engines.physics_engines.putting_green.python.green_surface import GreenSurface
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)


class TestRollMode:
    """Tests for RollMode enumeration."""

    def test_roll_modes_exist(self) -> None:
        """Verify all roll modes are defined."""
        assert hasattr(RollMode, "SLIDING")
        assert hasattr(RollMode, "ROLLING")
        assert hasattr(RollMode, "STOPPED")


class TestBallState:
    """Tests for BallState dataclass."""

    def test_ball_state_creation(self) -> None:
        """BallState should store position, velocity, spin."""
        state = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.array([0.0, 0.0, 100.0]),  # Backspin about x-axis
        )
        assert np.allclose(state.position, [5.0, 5.0])
        assert np.allclose(state.velocity, [2.0, 0.0])
        assert state.spin[2] == 100.0

    def test_ball_state_speed_property(self) -> None:
        """Speed should be magnitude of velocity."""
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([3.0, 4.0]),
            spin=np.zeros(3),
        )
        assert np.isclose(state.speed, 5.0)

    def test_ball_state_is_moving(self) -> None:
        """Should detect if ball is moving."""
        moving = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
            spin=np.zeros(3),
        )
        stopped = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            spin=np.zeros(3),
        )
        assert moving.is_moving
        assert not stopped.is_moving

    def test_ball_state_copy(self) -> None:
        """Should create independent copy."""
        original = BallState(
            position=np.array([1.0, 1.0]),
            velocity=np.array([2.0, 2.0]),
            spin=np.array([0.0, 0.0, 50.0]),
        )
        copy = original.copy()

        # Modify copy
        copy.position[0] = 999.0

        # Original should be unchanged
        assert original.position[0] == 1.0

    def test_ball_state_direction(self) -> None:
        """Should compute unit direction vector."""
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([3.0, 4.0]),
            spin=np.zeros(3),
        )
        direction = state.direction
        assert np.isclose(np.linalg.norm(direction), 1.0)


class TestBallRollPhysics:
    """Tests for BallRollPhysics class."""

    @pytest.fixture
    def physics(self) -> BallRollPhysics:
        """Create default ball roll physics."""
        return BallRollPhysics()

    @pytest.fixture
    def physics_with_turf(self) -> BallRollPhysics:
        """Create physics with specific turf."""
        turf = TurfProperties.create_preset("tournament_fast")
        return BallRollPhysics(turf=turf)

    @pytest.fixture
    def physics_with_green(self) -> BallRollPhysics:
        """Create physics with full green surface."""
        green = GreenSurface(
            width=20.0,
            height=20.0,
            turf=TurfProperties.create_preset("tournament_fast"),
        )
        return BallRollPhysics(green=green)

    def test_default_ball_properties(self, physics: BallRollPhysics) -> None:
        """Should have standard golf ball properties."""
        assert np.isclose(physics.ball_mass, 0.04593, atol=0.001)  # kg
        assert np.isclose(physics.ball_radius, 0.02135, atol=0.001)  # m

    def test_initial_roll_mode_determination(self, physics: BallRollPhysics) -> None:
        """Should determine correct initial roll mode."""
        # High spin relative to velocity = sliding
        sliding_state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.array([0.0, 500.0, 0.0]),  # High spin
        )
        assert physics.determine_roll_mode(sliding_state) == RollMode.SLIDING

        # Matched spin for pure rolling
        # For pure roll: ω = v / r, so v=2, r=0.02135 → ω ≈ 93.7
        rolling_state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.array([0.0, -93.7, 0.0]),  # Negative for forward roll
        )
        assert physics.determine_roll_mode(rolling_state) == RollMode.ROLLING

    def test_rolling_friction_force(self, physics: BallRollPhysics) -> None:
        """Rolling friction should oppose motion."""
        state = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.zeros(3),
        )
        friction = physics.compute_rolling_friction(state)

        # Should oppose velocity
        assert np.dot(friction, state.velocity) < 0
        # Should be finite
        assert np.all(np.isfinite(friction))

    def test_sliding_friction_force(self, physics: BallRollPhysics) -> None:
        """Sliding friction should be higher than rolling."""
        state = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.array([0.0, 500.0, 0.0]),  # High spin
        )

        sliding_friction = physics.compute_sliding_friction(state)
        rolling_friction = physics.compute_rolling_friction(state)

        # Sliding friction magnitude should be greater
        assert np.linalg.norm(sliding_friction) > np.linalg.norm(rolling_friction)

    def test_spin_decay_during_slide(self, physics: BallRollPhysics) -> None:
        """Spin should decay faster during sliding."""
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.array([0.0, 500.0, 0.0]),
        )

        new_spin = physics.compute_spin_decay(state, dt=0.1, mode=RollMode.SLIDING)

        # Spin should decrease
        assert np.linalg.norm(new_spin) < np.linalg.norm(state.spin)

    def test_spin_converges_to_rolling(self, physics: BallRollPhysics) -> None:
        """Spin should converge to pure rolling condition."""
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.array([0.0, 500.0, 0.0]),  # Excess backspin
        )

        # Simulate for a while
        dt = 0.01
        for _ in range(100):
            state = physics.step(state, dt)

        # Should be in rolling mode now (if still moving)
        if state.is_moving:
            mode = physics.determine_roll_mode(state)
            # Should have transitioned to rolling or very close
            assert mode == RollMode.ROLLING or state.speed < 0.5

    def test_step_advances_position(self, physics: BallRollPhysics) -> None:
        """Step should advance ball position based on velocity."""
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.zeros(3),
        )

        new_state = physics.step(state, dt=0.1)

        # Position should have advanced
        assert new_state.position[0] > state.position[0]

    def test_ball_decelerates_to_stop(self, physics: BallRollPhysics) -> None:
        """Ball should eventually stop due to friction."""
        state = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([1.0, 0.0]),
            spin=np.zeros(3),
        )

        dt = 0.01
        max_steps = 1000
        for _ in range(max_steps):
            state = physics.step(state, dt)
            if not state.is_moving:
                break

        # Should have stopped
        assert not state.is_moving
        assert physics.determine_roll_mode(state) == RollMode.STOPPED

    def test_slope_accelerates_ball(self, physics_with_green: BallRollPhysics) -> None:
        """Ball on slope should accelerate downhill."""
        # Add a slope to the green
        from src.engines.physics_engines.putting_green.python.green_surface import (
            SlopeRegion,
        )

        physics_with_green.green.add_slope_region(
            SlopeRegion(
                center=np.array([10.0, 10.0]),
                radius=5.0,
                slope_direction=np.array([1.0, 0.0]),
                slope_magnitude=0.05,  # 5% slope
            )
        )

        # Ball at rest on slope
        state = BallState(
            position=np.array([10.0, 10.0]),
            velocity=np.array([0.0, 0.0]),
            spin=np.zeros(3),
        )

        # After a step, ball should start moving downhill
        new_state = physics_with_green.step(state, dt=0.1)

        # Should have gained velocity (downhill = negative x direction)
        assert new_state.velocity[0] < 0

    def test_energy_dissipation(self, physics: BallRollPhysics) -> None:
        """Total energy should decrease due to friction."""
        state = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([3.0, 0.0]),
            spin=np.zeros(3),
        )

        initial_ke = physics.compute_kinetic_energy(state)

        # Simulate for a bit
        dt = 0.01
        for _ in range(50):
            state = physics.step(state, dt)

        final_ke = physics.compute_kinetic_energy(state)

        # Energy should have decreased
        assert final_ke < initial_ke

    def test_kinetic_energy_calculation(self, physics: BallRollPhysics) -> None:
        """Should compute kinetic energy correctly."""
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([3.0, 4.0]),
            spin=np.zeros(3),
        )

        ke = physics.compute_kinetic_energy(state)

        # KE = 0.5 * m * v^2
        expected_ke = 0.5 * physics.ball_mass * (5.0**2)
        assert np.isclose(ke, expected_ke, rtol=0.01)

    def test_simulate_putt(self, physics_with_green: BallRollPhysics) -> None:
        """Should simulate complete putt trajectory."""
        initial_state = BallState(
            position=np.array([5.0, 10.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.zeros(3),
        )

        trajectory = physics_with_green.simulate_putt(
            initial_state, max_time=10.0, dt=0.01
        )

        # Should have trajectory data
        assert len(trajectory["positions"]) > 1
        assert len(trajectory["velocities"]) > 1
        assert len(trajectory["times"]) > 1

        # Ball should have stopped
        final_vel = trajectory["velocities"][-1]
        assert np.linalg.norm(final_vel) < 0.01

    def test_simulate_with_hole(self, physics_with_green: BallRollPhysics) -> None:
        """Should detect when ball goes in hole."""
        physics_with_green.green.set_hole_position(np.array([15.0, 10.0]))

        # Putt straight at hole
        initial_state = BallState(
            position=np.array([5.0, 10.0]),
            velocity=np.array([3.0, 0.0]),  # Toward hole
            spin=np.zeros(3),
        )

        result = physics_with_green.simulate_putt(initial_state, max_time=10.0)

        # Should have "holed" flag
        assert result["holed"] is True

    def test_sidespin_curves_ball(self, physics: BallRollPhysics) -> None:
        """Sidespin should cause ball to curve."""
        # Ball with sidespin
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.array([0.0, 0.0, 100.0]),  # Sidespin about z-axis
        )

        # Simulate
        trajectory_positions = []
        dt = 0.01
        for _ in range(100):
            state = physics.step(state, dt)
            trajectory_positions.append(state.position.copy())

        # Ball should have curved (y-position should change)
        final_y = state.position[1]
        assert abs(final_y) > 0.001

    def test_backspin_check_effect(self, physics: BallRollPhysics) -> None:
        """Backspin should check (slow down) ball on impact."""
        # Initial sliding with backspin
        with_backspin = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([3.0, 0.0]),
            spin=np.array([0.0, 300.0, 0.0]),  # Backspin
        )
        without_backspin = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([3.0, 0.0]),
            spin=np.array([0.0, 0.0, 0.0]),
        )

        # Simulate both
        dt = 0.01
        steps = 20
        for _ in range(steps):
            with_backspin = physics.step(with_backspin, dt)
            without_backspin = physics.step(without_backspin, dt)

        # Ball with backspin should have traveled less
        assert with_backspin.position[0] < without_backspin.position[0]


class TestBallRollPhysicsEdgeCases:
    """Edge case tests for BallRollPhysics."""

    @pytest.fixture
    def physics(self) -> BallRollPhysics:
        return BallRollPhysics()

    def test_zero_velocity_step(self, physics: BallRollPhysics) -> None:
        """Step with zero velocity should not move ball."""
        state = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([0.0, 0.0]),
            spin=np.zeros(3),
        )

        new_state = physics.step(state, dt=0.1)

        # Position should be unchanged
        assert np.allclose(new_state.position, state.position)

    def test_very_small_velocity_stops(self, physics: BallRollPhysics) -> None:
        """Very small velocity should snap to zero (stopping threshold)."""
        state = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([0.001, 0.0]),  # Very slow
            spin=np.zeros(3),
        )

        new_state = physics.step(state, dt=0.1)

        # Should stop
        assert not new_state.is_moving

    def test_large_dt_stability(self, physics: BallRollPhysics) -> None:
        """Large time steps should not cause numerical instability."""
        state = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.zeros(3),
        )

        # Large time step
        new_state = physics.step(state, dt=1.0)

        # Should be finite
        assert np.all(np.isfinite(new_state.position))
        assert np.all(np.isfinite(new_state.velocity))

    def test_negative_velocity(self, physics: BallRollPhysics) -> None:
        """Negative velocity should work correctly."""
        state = BallState(
            position=np.array([10.0, 10.0]),
            velocity=np.array([-2.0, -1.0]),
            spin=np.zeros(3),
        )

        new_state = physics.step(state, dt=0.1)

        # Should move in negative direction
        assert new_state.position[0] < state.position[0]
        assert new_state.position[1] < state.position[1]

    def test_extreme_spin(self, physics: BallRollPhysics) -> None:
        """Extreme spin values should be handled."""
        state = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.array([0.0, 10000.0, 0.0]),  # Extreme spin
        )

        # Should not crash
        new_state = physics.step(state, dt=0.01)
        assert np.all(np.isfinite(new_state.velocity))

    def test_timestep_independence(self, physics: BallRollPhysics) -> None:
        """Results should be similar regardless of timestep size (within reason)."""
        initial = BallState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.zeros(3),
        )

        # Simulate with small timesteps
        state_small = initial.copy()
        for _ in range(100):
            state_small = physics.step(state_small, dt=0.001)

        # Simulate with larger timesteps
        state_large = initial.copy()
        for _ in range(10):
            state_large = physics.step(state_large, dt=0.01)

        # Should be reasonably close (not exact due to numerical differences)
        assert np.allclose(state_small.position, state_large.position, rtol=0.1)


class TestBallRollPhysicsIntegrators:
    """Tests for different integration methods."""

    def test_euler_integration(self) -> None:
        """Euler integration should work."""
        physics = BallRollPhysics(integrator="euler")
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.zeros(3),
        )

        new_state = physics.step(state, dt=0.01)
        assert np.all(np.isfinite(new_state.position))

    def test_rk4_integration(self) -> None:
        """RK4 integration should be more accurate."""
        physics = BallRollPhysics(integrator="rk4")
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.zeros(3),
        )

        new_state = physics.step(state, dt=0.01)
        assert np.all(np.isfinite(new_state.position))

    def test_verlet_integration(self) -> None:
        """Verlet integration for better energy conservation."""
        physics = BallRollPhysics(integrator="verlet")
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([2.0, 0.0]),
            spin=np.zeros(3),
        )

        new_state = physics.step(state, dt=0.01)
        assert np.all(np.isfinite(new_state.position))
