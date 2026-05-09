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


class TestMonitorInitialize:
    """Test ConservationMonitor.initialize() method."""

    def test_initialize_sets_initial_energy(self) -> None:
        """Test that initialize() sets E_initial."""
        from src.shared.python.core.constants import GRAVITY_M_S2

        engine = MockPhysicsEngine()
        engine.set_state(q=np.array([1.0, 2.0]), v=np.array([0.5, 0.5]))
        engine.set_mass_matrix(np.eye(2))
        engine.set_gravity_forces(np.array([0.0, -GRAVITY_M_S2]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        assert monitor.E_initial is not None
        assert isinstance(monitor.E_initial, float)

    def test_initialize_clears_drift_history(self) -> None:
        """Test that initialize() clears drift history."""
        engine = MockPhysicsEngine()
        engine.set_state(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))

        # Add some fake history
        monitor.drift_history.append((0.0, 0.5))
        monitor.drift_history.append((1.0, 1.0))

        # Initialize should clear it
        monitor.initialize()
        assert len(monitor.drift_history) == 0

    def test_initialize_computes_correct_energy(self) -> None:
        """Test that initialize() computes energy correctly."""
        engine = MockPhysicsEngine()

        # Set up simple state: KE = 0.5 * m * v^2
        m = 2.0
        v_val = 3.0
        engine.set_state(q=np.array([0.0]), v=np.array([v_val]))
        engine.set_mass_matrix(np.array([[m]]))
        engine.set_gravity_forces(np.array([0.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # KE = 0.5 * m * v^2 = 0.5 * 2.0 * 3.0^2 = 9.0 J
        expected_KE = 0.5 * m * v_val**2
        assert monitor.E_initial is not None
        np.testing.assert_allclose(monitor.E_initial, expected_KE, rtol=1e-10)
