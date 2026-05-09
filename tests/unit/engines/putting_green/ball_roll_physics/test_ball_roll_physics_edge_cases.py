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
