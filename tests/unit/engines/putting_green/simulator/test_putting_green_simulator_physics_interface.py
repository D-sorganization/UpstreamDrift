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


class TestPuttingGreenSimulatorPhysicsInterface:
    """Tests for PhysicsEngine protocol compliance."""

    @pytest.fixture
    def simulator(self) -> PuttingGreenSimulator:
        return PuttingGreenSimulator()

    def test_simulator_compute_mass_matrix(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        """Should return mass matrix (single ball = scalar mass)."""
        M = simulator.compute_mass_matrix()
        assert M.shape == (2, 2) or isinstance(M, int | float)

    def test_compute_bias_forces(self, simulator: PuttingGreenSimulator) -> None:
        """Should compute bias forces (friction + slope)."""
        simulator.set_ball_position(np.array([10.0, 10.0]))
        simulator.set_ball_velocity(np.array([1.0, 0.0]))

        bias = simulator.compute_bias_forces()

        assert bias.shape == (2,)
        assert np.all(np.isfinite(bias))

    def test_simulator_compute_gravity_forces(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        """Should compute gravity on slope."""
        # Add slope to green
        simulator.green.add_slope_region(
            SlopeRegion(
                center=np.array([10.0, 10.0]),
                radius=5.0,
                slope_direction=np.array([1.0, 0.0]),
                slope_magnitude=0.05,
            )
        )
        simulator.set_ball_position(np.array([10.0, 10.0]))

        gravity = simulator.compute_gravity_forces()

        # On slope, should have non-zero gravity component
        assert np.linalg.norm(gravity) > 0

    def test_compute_drift_acceleration(self, simulator: PuttingGreenSimulator) -> None:
        """Should compute drift (passive) acceleration."""
        simulator.set_ball_position(np.array([10.0, 10.0]))
        simulator.set_ball_velocity(np.array([1.0, 0.0]))

        drift = simulator.compute_drift_acceleration()

        assert drift.shape == (2,)
        assert np.all(np.isfinite(drift))

    def test_compute_control_acceleration(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        """Should compute control acceleration from applied force."""
        tau = np.array([0.1, 0.0])  # Applied force

        control_acc = simulator.compute_control_acceleration(tau)

        assert control_acc.shape == (2,)
        # a = F/m
        expected = tau / simulator.ball_mass
        assert np.allclose(control_acc, expected, rtol=0.1)
