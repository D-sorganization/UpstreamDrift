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


class TestPhysicalRealism:
    """Test physical realism of energy monitoring."""

    def test_conservative_system_maintains_energy(self) -> None:
        """Test that a true conservative system shows zero drift."""
        engine = MockPhysicsEngine()

        # Set up conservative system (no external forces, no damping)
        engine.set_state(q=np.array([1.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Simulate conservation (no actual dynamics, just checking)
        # In real simulation, energy would be conserved
        drift_pct = monitor.check_and_warn()

        # Should have zero drift
        np.testing.assert_allclose(drift_pct, 0.0, atol=1e-10)

    def test_drift_detection_sensitivity(self) -> None:
        """Test that monitor detects small energy changes."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()  # E = 0.5 J

        # Create tiny 0.1% drift
        v_new = np.sqrt(1.001)
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        drift_pct = monitor.check_and_warn()

        # Should detect this small drift
        assert 0.09 < drift_pct < 0.11  # ~0.1%
