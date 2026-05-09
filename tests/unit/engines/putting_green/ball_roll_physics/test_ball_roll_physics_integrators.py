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
