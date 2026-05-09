"""Tests for energy conservation monitoring.

Tests the energy drift detection system that implements Guideline O3
for ensuring physical validity of conservative system integrations.
"""

from __future__ import annotations

from typing import NoReturn

import numpy as np
import pytest
from src.shared.python.core.contracts import StateError
from src.shared.python.physics.energy_monitor import (
    ENERGY_DRIFT_CRITICAL_PCT,
    ENERGY_DRIFT_TOLERANCE_PCT,
    ConservationMonitor,
    EnergySnapshot,
    IntegrationFailureError,
)
from src.shared.python.tests.mock_physics_engine import (
    MockPhysicsEngine,
    as_physics_engine,
)


class TestEstimateMaxStableTimestep:
    """Test estimate_max_stable_timestep() method."""

    def test_slow_motion_recommendation(self) -> None:
        """Test timestep recommendation for slow motion."""
        engine = MockPhysicsEngine()
        engine.set_state(q=np.array([0.0, 0.0]), v=np.array([0.1, 0.2]))  # ||v|| < 1.0

        monitor = ConservationMonitor(as_physics_engine(engine))
        dt_max = monitor.estimate_max_stable_timestep()

        # For slow motion, should recommend dt = 0.01s
        assert dt_max == 0.01

    def test_normal_motion_recommendation(self) -> None:
        """Test timestep recommendation for normal motion."""
        engine = MockPhysicsEngine()
        engine.set_state(
            q=np.array([0.0, 0.0]),
            v=np.array([3.0, 4.0]),  # ||v|| = 5.0, in [1, 10)
        )

        monitor = ConservationMonitor(as_physics_engine(engine))
        dt_max = monitor.estimate_max_stable_timestep()

        # For normal motion, should recommend dt = 0.001s
        assert dt_max == 0.001

    def test_high_speed_motion_recommendation(self) -> None:
        """Test timestep recommendation for high-speed motion."""
        engine = MockPhysicsEngine()
        engine.set_state(
            q=np.array([0.0, 0.0, 0.0]),
            v=np.array([50.0, 50.0, 50.0]),  # ||v|| ~ 86.6, >> 10
        )

        monitor = ConservationMonitor(as_physics_engine(engine))
        dt_max = monitor.estimate_max_stable_timestep()

        # For high-speed motion, should recommend dt = 0.0001s
        assert dt_max == 0.0001

    def test_zero_velocity(self) -> None:
        """Test timestep recommendation with zero velocity."""
        engine = MockPhysicsEngine()
        engine.set_state(q=np.array([0.0, 0.0]), v=np.array([0.0, 0.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        dt_max = monitor.estimate_max_stable_timestep()

        # Zero velocity -> slow motion regime
        assert dt_max == 0.01
