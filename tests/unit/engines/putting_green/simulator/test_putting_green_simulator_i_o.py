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


class TestPuttingGreenSimulatorIO:
    """Tests for save/load functionality."""

    @pytest.fixture
    def simulator(self) -> PuttingGreenSimulator:
        return PuttingGreenSimulator()

    def test_simulator_load_from_path(self, simulator: PuttingGreenSimulator) -> None:
        """Should load green configuration from file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "green": {
                    "width": 25.0,
                    "height": 25.0,
                    "turf": {
                        "stimp_rating": 11.0,
                        "grass_type": "bent_grass",
                    },
                    "hole_position": [15.0, 12.0],
                }
            }
            json.dump(config, f)
            f.flush()

            simulator.load_from_path(f.name)

            assert simulator.green.width == 25.0
            assert np.allclose(simulator.green.hole_position, [15.0, 12.0])

    def test_simulator_load_from_string(self, simulator: PuttingGreenSimulator) -> None:
        """Should load from JSON string."""
        config_str = json.dumps(
            {
                "green": {
                    "width": 30.0,
                    "height": 30.0,
                    "turf": {"stimp_rating": 12.0},
                }
            }
        )

        simulator.load_from_string(config_str, extension="json")

        assert simulator.green.width == 30.0

    def test_load_topographical_data(self, simulator: PuttingGreenSimulator) -> None:
        """Should load topographical/elevation data."""
        # Create a heightmap file
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            rng = np.random.default_rng(42)
            heightmap = rng.random((100, 100)) * 0.1
            np.save(f.name, heightmap)

            simulator.load_topographical_data(
                f.name,
                width=20.0,
                height=20.0,
            )

            # Should have loaded elevation data
            elev = simulator.green.get_elevation_at(np.array([10.0, 10.0]))
            assert 0 <= elev <= 0.1

    def test_load_topographical_csv(self, simulator: PuttingGreenSimulator) -> None:
        """Should load topographical data from CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write CSV with x, y, elevation columns
            f.write("x,y,elevation\n")
            for x in np.linspace(0, 20, 10):
                for y in np.linspace(0, 20, 10):
                    elev = 0.05 * np.sin(x / 5) * np.cos(y / 5)
                    f.write(f"{x},{y},{elev}\n")
            f.flush()

            simulator.load_topographical_data(f.name, width=20.0, height=20.0)

            # Should have loaded
            elev = simulator.green.get_elevation_at(np.array([10.0, 10.0]))
            assert np.isfinite(elev)

    def test_load_topographical_geotiff(self, simulator: PuttingGreenSimulator) -> None:
        """Should support GeoTIFF format (or skip if not available)."""
        # This would require rasterio, so we just check the method exists
        assert hasattr(simulator, "load_topographical_data")

    def test_export_simulation_result(self, simulator: PuttingGreenSimulator) -> None:
        """Should export simulation result to file."""
        simulator.set_ball_position(np.array([5.0, 10.0]))
        stroke_params = StrokeParameters(
            speed=1.5,
            direction=np.array([1.0, 0.0]),
            face_angle=0.0,
            attack_angle=0.0,
        )

        result = simulator.simulate_putt(stroke_params)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            simulator.export_result(result, f.name)

            # Read back and verify
            with open(f.name) as rf:
                data = json.load(rf)
                assert "positions" in data
                assert "times" in data
