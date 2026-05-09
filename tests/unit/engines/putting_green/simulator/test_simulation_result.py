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


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    def test_result_contains_trajectory(self) -> None:
        """Result should contain trajectory data."""
        result = SimulationResult(
            positions=np.array([[0, 0], [1, 0], [2, 0]]),
            velocities=np.array([[2, 0], [1.5, 0], [0, 0]]),
            times=np.array([0, 0.1, 0.2]),
            holed=False,
            final_position=np.array([2, 0]),
        )
        assert len(result.positions) == 3
        assert not result.holed

    def test_result_distance_rolled(self) -> None:
        """Should compute total distance rolled."""
        result = SimulationResult(
            positions=np.array([[0, 0], [1, 0], [2, 0]]),
            velocities=np.array([[2, 0], [1.5, 0], [0, 0]]),
            times=np.array([0, 0.1, 0.2]),
            holed=False,
            final_position=np.array([2, 0]),
        )
        # Total distance should be approximately 2
        assert np.isclose(result.total_distance, 2.0, rtol=0.1)

    def test_result_duration(self) -> None:
        """Should report simulation duration."""
        result = SimulationResult(
            positions=np.array([[0, 0], [1, 0], [2, 0]]),
            velocities=np.array([[2, 0], [1.5, 0], [0, 0]]),
            times=np.array([0, 0.5, 1.5]),
            holed=False,
            final_position=np.array([2, 0]),
        )
        assert result.duration == 1.5
