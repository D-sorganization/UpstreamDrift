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


class TestTopographyLoading:
    """Tests for loading topographical data."""

    def test_load_numpy_heightmap(self) -> None:
        """Test loading heightmap from NumPy file."""
        sim = PuttingGreenSimulator()

        # Create test heightmap
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            heightmap = np.zeros((50, 50))
            # Add a hill
            for i in range(50):
                for j in range(50):
                    dist = np.sqrt((i - 25) ** 2 + (j - 25) ** 2)
                    heightmap[i, j] = max(0, 0.05 - dist * 0.002)
            np.save(f.name, heightmap)

            sim.load_topographical_data(f.name, width=20.0, height=20.0)

        # Check elevation at center (should be higher)
        center_elev = sim.green.get_elevation_at(np.array([10.0, 10.0]))
        edge_elev = sim.green.get_elevation_at(np.array([1.0, 1.0]))

        assert center_elev > edge_elev, "Assertion failed: center_elev > edge_elev"

    def test_load_csv_contours(self) -> None:
        """Test loading elevation from CSV."""
        sim = PuttingGreenSimulator()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("x,y,elevation\n")
            f.write("5,5,0.02\n")
            f.write("15,5,0.01\n")
            f.write("10,10,0.03\n")
            f.write("5,15,0.015\n")
            f.write("15,15,0.005\n")
            f.flush()

            sim.load_topographical_data(f.name, width=20.0, height=20.0)

        # Check that interpolation gives reasonable values
        elev = sim.green.get_elevation_at(np.array([10.0, 10.0]))
        assert 0 < elev < 0.05, "Assertion failed: 0 < elev < 0.05"

    def test_load_json_config(self) -> None:
        """Test loading green config from JSON."""
        sim = PuttingGreenSimulator()

        config = {
            "green": {
                "width": 25.0,
                "height": 30.0,
                "turf": {
                    "stimp_rating": 11.5,
                    "grass_type": "bent_grass",
                },
                "hole_position": [20.0, 15.0],
                "slopes": [
                    {
                        "center": [12.5, 15.0],
                        "radius": 10.0,
                        "direction": [0.707, 0.707],
                        "magnitude": 0.025,
                    }
                ],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()

            sim.load_from_path(f.name)

        assert sim.green.width == 25.0, "Assertion failed: sim.green.width == 25.0"
        assert sim.green.height == 30.0, "Assertion failed: sim.green.height == 30.0"
        assert np.allclose(
            sim.green.hole_position, [20.0, 15.0]
        ), "Assertion failed: np.allclose(sim.green.hole_position, [20.0, 15.0])"
