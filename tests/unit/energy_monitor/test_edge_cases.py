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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_initial_energy(self) -> None:
        """Test behavior with zero initial energy.

        Note: This is a pathological case. Division by zero occurs when
        calculating drift percentage. This test documents the current behavior.
        """
        engine = MockPhysicsEngine()

        # Zero energy state
        engine.set_state(q=np.array([0.0]), v=np.array([0.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Drift calculation with zero denominator causes division by zero
        # Perturb slightly
        engine.set_state(q=np.array([0.0]), v=np.array([0.01]))

        # Current implementation raises ZeroDivisionError for zero initial energy
        # This is acceptable as it's a pathological case (no energy to conserve)
        with pytest.raises(ZeroDivisionError):
            monitor.check_and_warn()

    def test_very_large_energy_drift(self) -> None:
        """Test behavior with extremely large drift."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create 1000% drift
        v_new = np.sqrt(10.0)  # 10x energy
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        # Should raise critical error
        with pytest.raises(IntegrationFailureError):
            monitor.check_and_warn()

    def test_negative_energy_total(self) -> None:
        """Test with negative total energy (PE dominates)."""
        engine = MockPhysicsEngine()

        # Large negative potential, small kinetic
        # KE = 0.5, PE = -10.0, Total = -9.5
        engine.set_state(q=np.array([10.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.array([-1.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Drift calculation should work with negative energy
        drift_pct = monitor.check_and_warn()

        assert np.isfinite(drift_pct)

    def test_multiple_initializations(self) -> None:
        """Test that re-initialization resets the monitor."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()
        E_first = monitor.E_initial

        # Add some drift history
        monitor.check_and_warn()
        assert len(monitor.drift_history) == 1

        # Change state and re-initialize
        engine.set_state(q=np.array([0.0]), v=np.array([2.0]))
        monitor.initialize()
        E_second = monitor.E_initial

        # Energy should be different
        assert E_second != E_first
        # History should be cleared
        assert len(monitor.drift_history) == 0
