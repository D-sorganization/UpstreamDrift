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


class TestScatterAnalysis:
    """Tests for scatter/dispersion analysis."""

    def test_scatter_produces_variation(self) -> None:
        """Scatter analysis should produce varied results."""
        turf = TurfProperties.create_preset("tournament_fast")
        green = GreenSurface(width=20.0, height=20.0, turf=turf)
        sim = PuttingGreenSimulator(green=green)

        stroke = StrokeParameters(
            speed=2.0,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        results = sim.simulate_scatter(
            start_position=np.array([5.0, 10.0]),
            stroke_params=stroke,
            n_simulations=20,
            speed_variance=0.15,
            direction_variance_deg=3.0,
        )

        final_positions = np.array([r.final_position for r in results])

        # Check variance in final positions
        std_x = np.std(final_positions[:, 0])
        std_y = np.std(final_positions[:, 1])

        # Should have some spread
        assert std_x > 0.1, "Assertion failed: std_x > 0.1"
        assert (
            std_y > 0.01
        )  # Less spread in y for straight putt, "Assertion failed: std_y > 0.01  # Less spread in y for straight putt"
