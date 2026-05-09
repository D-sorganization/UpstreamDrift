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


class TestCheckAndWarn:
    """Test ConservationMonitor.check_and_warn() method."""

    def test_requires_initialization(self) -> None:
        """Test that check_and_warn() requires initialization first."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(as_physics_engine(engine))

        with pytest.raises(StateError, match="not initialized"):
            monitor.check_and_warn()

    def test_zero_drift_returns_zero(self) -> None:
        """Test that zero drift returns 0.0%."""
        engine = MockPhysicsEngine()
        engine.set_state(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        engine.set_mass_matrix(np.eye(2))
        engine.set_gravity_forces(np.zeros(2))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # No change in state
        drift_pct = monitor.check_and_warn()

        np.testing.assert_allclose(drift_pct, 0.0, atol=1e-10)

    def test_drift_calculation_positive(self) -> None:
        """Test drift calculation when energy increases."""
        engine = MockPhysicsEngine()

        # Initial: v = 1.0, KE = 0.5 * 1.0^2 = 0.5 J
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()  # E_initial = 0.5 J

        # Change: v = 1.02, KE = 0.5 * 1.02^2 = 0.5202 J (small increase, < 5%)
        v_new = 1.02
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        drift_pct = monitor.check_and_warn()

        # Drift = (0.5202 - 0.5) / 0.5 * 100 = 4.04%
        E_new = 0.5 * v_new**2
        expected_drift = (E_new - 0.5) / 0.5 * 100
        np.testing.assert_allclose(drift_pct, expected_drift, rtol=1e-6)

    def test_drift_calculation_negative(self) -> None:
        """Test drift calculation when energy decreases."""
        engine = MockPhysicsEngine()

        # Initial: v = 2.0, KE = 0.5 * 2.0^2 = 2.0 J
        engine.set_state(q=np.array([0.0]), v=np.array([2.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Change: v = 1.96, KE = 0.5 * 1.96^2 = 1.9208 J (small decrease, < 5%)
        v_new = 1.96
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        drift_pct = monitor.check_and_warn()

        # Drift = (1.9208 - 2.0) / 2.0 * 100 = -3.96%
        E_new = 0.5 * v_new**2
        expected_drift = (E_new - 2.0) / 2.0 * 100
        np.testing.assert_allclose(drift_pct, expected_drift, rtol=1e-6)

    def test_drift_history_accumulation(self) -> None:
        """Test that drift history is accumulated."""
        engine = MockPhysicsEngine()
        engine.set_state(np.array([0.0]), np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Multiple checks
        engine.advance_time(1.0)
        monitor.check_and_warn()

        engine.advance_time(1.0)
        monitor.check_and_warn()

        engine.advance_time(1.0)
        monitor.check_and_warn()

        # Should have 3 entries
        assert len(monitor.drift_history) == 3
        assert monitor.drift_history[0][0] == 1.0  # First time
        assert monitor.drift_history[1][0] == 2.0  # Second time
        assert monitor.drift_history[2][0] == 3.0  # Third time

    def test_warning_at_tolerance_threshold(self, caplog) -> None:
        """Test that warning is logged at 1% drift threshold."""
        engine = MockPhysicsEngine()

        # Initial: KE = 0.5 J
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create exactly 1.1% drift (just above threshold)
        # E_new = E_initial * 1.011 = 0.5 * 1.011 = 0.5055 J
        # v_new^2 = 2 * E_new = 1.011 -> v_new = sqrt(1.011) ≈ 1.00548
        v_new = np.sqrt(1.011)
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        with caplog.at_level("WARNING"):
            drift_pct = monitor.check_and_warn()

        # Should warn because drift > 1%
        assert drift_pct > 1.0
        assert (
            "Energy conservation violated" in caplog.text
            or "conservation" in caplog.text.lower()
        )

    def test_critical_error_at_5_percent_drift(self) -> None:
        """Test that IntegrationFailureError is raised at 5% drift."""
        engine = MockPhysicsEngine()

        # Initial: KE = 0.5 J
        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create 6% drift (above critical threshold)
        # E_new = E_initial * 1.06 = 0.5 * 1.06 = 0.53 J
        # v_new = sqrt(2 * 0.53) = sqrt(1.06)
        v_new = np.sqrt(1.06)
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        with pytest.raises(IntegrationFailureError, match="INTEGRATION FAILURE"):
            monitor.check_and_warn()

    def test_no_warning_below_tolerance(self, caplog) -> None:
        """Test that no warning is logged below 1% drift."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create 0.5% drift (below threshold)
        v_new = np.sqrt(1.005)
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        with caplog.at_level("WARNING"):
            drift_pct = monitor.check_and_warn()

        # Should not warn
        assert drift_pct < 1.0
        # Check no warning in logs (might have other logs, so check specifically)
        energy_warnings = [
            record
            for record in caplog.records
            if "energy conservation violated" in record.message.lower()
        ]
        assert len(energy_warnings) == 0

    def test_negative_drift_triggers_warning(self, caplog) -> None:
        """Test that negative drift (energy loss) also triggers warning."""
        engine = MockPhysicsEngine()

        engine.set_state(q=np.array([0.0]), v=np.array([1.0]))
        engine.set_mass_matrix(np.eye(1))
        engine.set_gravity_forces(np.zeros(1))

        monitor = ConservationMonitor(as_physics_engine(engine))
        monitor.initialize()

        # Create -1.5% drift
        v_new = np.sqrt(0.985)  # E = 0.985 * E_initial
        engine.set_state(q=np.array([0.0]), v=np.array([v_new]))

        with caplog.at_level("WARNING"):
            drift_pct = monitor.check_and_warn()

        assert drift_pct < -1.0
        assert (
            "Energy conservation violated" in caplog.text
            or "conservation" in caplog.text.lower()
        )
