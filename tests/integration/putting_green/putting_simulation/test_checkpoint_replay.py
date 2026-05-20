"""Integration tests for Putting Green Simulation.

These tests verify that all components work together correctly:
- TurfProperties + GreenSurface
- BallRollPhysics + GreenSurface
- PutterStroke + BallRollPhysics
- Full simulation end-to-end
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    BallRollPhysics,
    BallState,
)
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    PutterStroke,
    PutterType,
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.simulator import (
    PuttingGreenSimulator,
    SimulationConfig,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)


class TestCheckpointReplay:
    """Tests for checkpoint and replay functionality."""

    def test_checkpoint_and_restore(self) -> None:
        """Test saving and restoring simulation state."""
        sim = PuttingGreenSimulator()
        sim.set_ball_position(np.array([5.0, 10.0]))
        sim.set_ball_velocity(np.array([2.0, 0.0]))

        checkpoint = sim.get_checkpoint()

        # Advance simulation
        for _ in range(100):
            sim.step()

        # Position should have changed
        pos_after = sim.get_ball_position()
        assert pos_after[0] > 5.0, "Assertion failed: pos_after[0] > 5.0"

        # Restore checkpoint
        sim.restore_checkpoint(checkpoint)

        # Position should be back to original
        pos_restored = sim.get_ball_position()
        assert np.isclose(
            pos_restored[0], 5.0, atol=0.01
        ), "Assertion failed: np.isclose(pos_restored[0], 5.0, atol=0.01)"

    def test_deterministic_replay(self) -> None:
        """Test that simulation is deterministic (same inputs = same outputs)."""
        turf = TurfProperties.create_preset("tournament_fast")
        green = GreenSurface(width=20.0, height=20.0, turf=turf)

        config = SimulationConfig(timestep=0.001)
        sim1 = PuttingGreenSimulator(green=green, config=config)
        sim2 = PuttingGreenSimulator(green=green, config=config)

        stroke = StrokeParameters(
            speed=2.5,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=-2.0,
        )

        sim1.set_ball_position(np.array([5.0, 10.0]))
        sim2.set_ball_position(np.array([5.0, 10.0]))

        result1 = sim1.simulate_putt(stroke)
        result2 = sim2.simulate_putt(stroke)

        # Results should be identical
        assert np.allclose(
            result1.final_position, result2.final_position
        ), "Assertion failed: np.allclose(result1.final_position, result2.final_position)"
        assert np.allclose(
            result1.positions, result2.positions
        ), "Assertion failed: np.allclose(result1.positions, result2.positions)"
