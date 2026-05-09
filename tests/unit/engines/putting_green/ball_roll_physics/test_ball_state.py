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

    def test_ball_state_column_vector_velocity_is_supported(self) -> None:
        """Column-vector velocity input should be normalized to 1D."""
        state = BallState(
            position=np.array([[0.0], [0.0]]),
            velocity=np.array([[3.0], [4.0]]),
            spin=np.array([[0.0], [0.0], [0.0]]),
        )
        assert state.velocity.shape == (2,)
        assert np.isclose(state.speed, 5.0)
        assert np.allclose(state.direction, np.array([0.6, 0.8]))

    def test_ball_state_row_vector_velocity_is_supported(self) -> None:
        """Row-vector velocity input should be normalized to 1D."""
        state = BallState(
            position=np.array([[0.0, 0.0]]),
            velocity=np.array([[3.0, 4.0]]),
            spin=np.array([[0.0, 0.0, 0.0]]),
        )
        assert state.velocity.shape == (2,)
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
