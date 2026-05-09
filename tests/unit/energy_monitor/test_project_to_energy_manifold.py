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


class TestProjectToEnergyManifold:
    """Test project_to_energy_manifold() method."""

    def test_requires_initialization(self) -> None:
        """Test that projection requires initialization first."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(as_physics_engine(engine))

        with pytest.raises(StateError, match="not initialized"):
            monitor.project_to_energy_manifold()

    def test_projection_scales_velocity(self) -> None:
        """Test that projection scales velocity to restore energy."""
        engine = MockPhysicsEngine()

        # Initial: v = 1.0, E = 0.5
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()  # E_initial = 0.5

        # Perturb: v = 1.5, E = 1.125 (2.25x increase)
        engine.set_state(q=np.array([0.0]), v=np.array([1.5]))

        # Project back
        monitor.project_to_energy_manifold()

        # Check that energy is restored
        q, v = engine.get_state()
        E_restored = 0.5 * v[0] ** 2

        np.testing.assert_allclose(E_restored, 0.5, rtol=1e-6)

    def test_projection_does_not_change_position(self) -> None:
        """Test that projection only changes velocity, not position."""
        engine = MockPhysicsEngine()

        q_initial = np.array([1.5, 2.5])
        v_initial = np.array([1.0, 1.0])

        engine.set_state(q_initial, v_initial)
        engine.set_mass_matrix(np.eye(2))
        engine.set_gravity_forces(np.zeros(2))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Perturb velocity
        engine.set_state(q_initial, v_initial * 1.2)

        # Project
        monitor.project_to_energy_manifold()

        # Check position unchanged
        q, v = engine.get_state()
        np.testing.assert_allclose(q, q_initial, rtol=1e-10)

    def test_projection_with_near_zero_energy(self, caplog) -> None:
        """Test projection behavior when current energy is near zero."""
        engine = MockPhysicsEngine()

        # Initial energy
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Set energy to near zero
        engine.set_state(q=np.array([0.0]), v=np.array([1e-10]))

        with caplog.at_level("WARNING"):
            monitor.project_to_energy_manifold()

        # Should warn about inability to project
        assert "Cannot project to energy manifold" in caplog.text

    def test_projection_preserves_direction(self) -> None:
        """Test that projection preserves velocity direction (only scales magnitude)."""
        engine = MockPhysicsEngine(n_dof=2)

        v_initial = np.array([3.0, 4.0])  # ||v|| = 5.0
        engine.set_state(q=np.zeros(2), v=v_initial)
        engine.set_mass_matrix(np.eye(2))
        engine.set_gravity_forces(np.zeros(2))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Perturb magnitude
        engine.set_state(q=np.zeros(2), v=v_initial * 1.3)

        # Project
        monitor.project_to_energy_manifold()

        # Check direction preserved
        q, v_projected = engine.get_state()
        v_initial_normalized = v_initial / np.linalg.norm(v_initial)
        v_projected_normalized = v_projected / np.linalg.norm(v_projected)

        np.testing.assert_allclose(
            v_projected_normalized,
            v_initial_normalized,
            rtol=1e-6,
            err_msg="Projection should preserve velocity direction",
        )
