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


class TestGreenReading:
    """Tests for green reading functionality."""

    def test_aim_line_calculation(self) -> None:
        """Test aim line calculation for breaking putt."""
        turf = TurfProperties.create_preset("tournament_fast")
        green = GreenSurface(width=20.0, height=20.0, turf=turf)
        green.add_slope_region(
            SlopeRegion(
                center=np.array([10.0, 10.0]),
                radius=10.0,
                slope_direction=np.array([0.0, 1.0]),  # Right to left break
                slope_magnitude=0.03,
            )
        )
        green.set_hole_position(np.array([15.0, 10.0]))

        sim = PuttingGreenSimulator(green=green)

        aim_info = sim.compute_aim_line(np.array([5.0, 10.0]))

        # Aim point should be to the left of the hole to compensate for break
        assert "aim_point" in aim_info, "Assertion failed: aim_point in aim_info"
        assert (
            aim_info["break"] > 0
        )  # Should have detected break, "Assertion failed: aim_info[break] > 0  # Should have detected break"

    def test_putt_line_reading(self) -> None:
        """Test reading putt line for elevations and slopes."""
        turf = TurfProperties.create_preset("tournament_fast")
        green = GreenSurface(width=20.0, height=20.0, turf=turf)
        green.add_slope_region(
            SlopeRegion(
                center=np.array([10.0, 10.0]),
                radius=8.0,
                slope_direction=np.array([1.0, 0.0]),
                slope_magnitude=0.02,
            )
        )
        green.set_hole_position(np.array([15.0, 10.0]))

        sim = PuttingGreenSimulator(green=green)

        reading = sim.read_green(np.array([5.0, 10.0]), np.array([15.0, 10.0]))

        assert "positions" in reading, "Assertion failed: positions in reading"
        assert "slopes" in reading, "Assertion failed: slopes in reading"
        assert "recommended_speed" in reading, (
            "Assertion failed: recommended_speed in reading"
        )
        assert reading["distance"] > 0, "Assertion failed: reading[distance] > 0"
