"""Unit tests for PuttingGreenSimulator module.

TDD Tests - These tests define the expected behavior of the main
putting green simulator engine that implements the PhysicsEngine protocol.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pytest
from src.engines.physics_engines.putting_green.python.green_surface import (
    GreenSurface,
    SlopeRegion,
)
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.simulator import (
    PuttingGreenSimulator,
    SimulationConfig,
    SimulationResult,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)


class TestPuttingGreenSimulatorCheckpoints:
    """Tests for checkpoint functionality."""

    @pytest.fixture
    def simulator(self) -> PuttingGreenSimulator:
        green = GreenSurface(
            width=20.0,
            height=20.0,
            turf=TurfProperties.create_preset("tournament_fast"),
        )
        return PuttingGreenSimulator(green=green)

    def test_get_checkpoint(self, simulator: PuttingGreenSimulator) -> None:
        """Should save checkpoint of current state."""
        simulator.set_ball_position(np.array([5.0, 5.0]))
        simulator.set_ball_velocity(np.array([1.0, 0.0]))

        checkpoint = simulator.get_checkpoint()

        assert checkpoint is not None
        assert "position" in checkpoint or hasattr(checkpoint, "q")

    def test_simulator_restore_checkpoint(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        """Should restore state from checkpoint."""
        # Set initial state
        simulator.set_ball_position(np.array([5.0, 5.0]))
        simulator.set_ball_velocity(np.array([1.0, 0.0]))

        checkpoint = simulator.get_checkpoint()

        # Advance simulation
        for _ in range(50):
            simulator.step()

        # Restore
        simulator.restore_checkpoint(checkpoint)

        # Should be back to original
        q, v = simulator.get_state()
        assert np.isclose(q[0], 5.0, atol=0.01)
