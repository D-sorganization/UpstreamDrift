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


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_simulator_default_config(self) -> None:
        """Default config should have sensible values."""
        config = SimulationConfig()
        assert config.timestep > 0
        assert config.max_simulation_time > 0
        assert config.stopping_velocity_threshold > 0

    def test_simulator_config_validation(self) -> None:
        """Should validate configuration parameters."""
        with pytest.raises(ValueError):
            SimulationConfig(timestep=-0.01)
        with pytest.raises(ValueError):
            SimulationConfig(max_simulation_time=0)

    def test_config_from_dict(self) -> None:
        """Should create config from dictionary."""
        data = {
            "timestep": 0.005,
            "max_simulation_time": 15.0,
            "stopping_velocity_threshold": 0.005,
        }
        config = SimulationConfig.from_dict(data)
        assert config.timestep == 0.005

    def test_config_to_dict(self) -> None:
        """Should serialize config to dictionary."""
        config = SimulationConfig(timestep=0.01)
        data = config.to_dict()
        assert data["timestep"] == 0.01
