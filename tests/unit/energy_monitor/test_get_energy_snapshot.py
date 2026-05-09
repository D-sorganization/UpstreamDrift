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


class TestGetEnergySnapshot:
    """Test ConservationMonitor.get_energy_snapshot() method."""

    def test_snapshot_captures_current_time(self) -> None:
        """Test that snapshot captures current simulation time."""
        engine = MockPhysicsEngine()
        engine.time = 5.5
        engine.set_state(np.array([0.0, 0.0]), np.array([0.0, 0.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        snapshot = monitor.get_energy_snapshot()

        assert snapshot.time == 5.5

    def test_snapshot_computes_kinetic_energy(self) -> None:
        """Test kinetic energy computation: KE = 0.5 * v^T * M * v."""
        engine = MockPhysicsEngine()

        # Simple case: 1D, m=2.0, v=3.0 -> KE = 0.5 * 2.0 * 3.0^2 = 9.0
        engine.set_state(q=np.array([0.0]), v=np.array([3.0]))
        engine.set_mass_matrix(np.array([[2.0]]))
        engine.set_gravity_forces(np.array([0.0]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        snapshot = monitor.get_energy_snapshot()

        expected_KE = 0.5 * 2.0 * 3.0**2
        np.testing.assert_allclose(snapshot.kinetic, expected_KE, rtol=1e-10)

    def test_snapshot_computes_potential_energy(self) -> None:
        """Test potential energy computation: PE = -q^T * g."""
        from src.shared.python.core.constants import GRAVITY_M_S2

        engine = MockPhysicsEngine()

        # q = [1.0], g = [-GRAVITY_M_S2] -> PE = -1.0 * (-GRAVITY_M_S2) = GRAVITY_M_S2
        engine.set_state(q=np.array([1.0]), v=np.array([0.0]))
        engine.set_mass_matrix(np.array([[1.0]]))
        engine.set_gravity_forces(np.array([-GRAVITY_M_S2]))

        monitor = ConservationMonitor(as_physics_engine(engine))
        snapshot = monitor.get_energy_snapshot()

        expected_PE = -1.0 * (-GRAVITY_M_S2)
        np.testing.assert_allclose(snapshot.potential, expected_PE, rtol=1e-10)

    def test_snapshot_with_multidof_system(self) -> None:
        """Test energy computation for multi-DOF system."""
        from src.shared.python.core.constants import GRAVITY_M_S2

        engine = MockPhysicsEngine(n_dof=3)

        q = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, 1.0, 1.5])
        M = np.diag([1.0, 2.0, 3.0])
        g = np.array([-GRAVITY_M_S2, -GRAVITY_M_S2, -GRAVITY_M_S2])

        engine.set_state(q, v)
        engine.set_mass_matrix(M)
        engine.set_gravity_forces(g)

        monitor = ConservationMonitor(as_physics_engine(engine))
        snapshot = monitor.get_energy_snapshot()

        # KE = 0.5 * v^T * M * v
        # = 0.5 * (0.5^2 * 1.0 + 1.0^2 * 2.0 + 1.5^2 * 3.0)
        # = 0.5 * (0.25 + 2.0 + 6.75) = 0.5 * 9.0 = 4.5
        expected_KE = 0.5 * (v * M.diagonal() * v).sum()

        # PE = -q^T * g = -(1.0 * -GRAVITY_M_S2 + 2.0 * -GRAVITY_M_S2 + 3.0 * -GRAVITY_M_S2)
        # = -(-GRAVITY_M_S2 - 2*GRAVITY_M_S2 - 3*GRAVITY_M_S2) = 6*GRAVITY_M_S2
        expected_PE = -np.dot(q, g)

        np.testing.assert_allclose(snapshot.kinetic, expected_KE, rtol=1e-10)
        np.testing.assert_allclose(snapshot.potential, expected_PE, rtol=1e-10)
