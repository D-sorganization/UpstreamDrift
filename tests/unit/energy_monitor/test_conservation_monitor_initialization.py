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


class TestConservationMonitorInitialization:
    """Test ConservationMonitor initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic monitor initialization."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(as_physics_engine(engine))

        assert monitor.engine is engine
        assert monitor.E_initial is None
        assert len(monitor.drift_history) == 0
        assert monitor.max_drift_pct == ENERGY_DRIFT_TOLERANCE_PCT
        assert monitor.critical_drift_pct == ENERGY_DRIFT_CRITICAL_PCT

    def test_custom_drift_thresholds(self) -> None:
        """Test initialization with custom drift thresholds."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(
            as_physics_engine(engine),
            max_drift_pct=0.5,
            critical_drift_pct=2.0,
        )

        assert monitor.max_drift_pct == 0.5
        assert monitor.critical_drift_pct == 2.0

    def test_default_tolerance_values(self) -> None:
        """Test that default tolerance values match Guideline O3."""
        engine = MockPhysicsEngine()
        monitor = ConservationMonitor(as_physics_engine(engine))

        # Per Guideline O3
        assert monitor.max_drift_pct == 1.0  # 1% warning threshold
        assert monitor.critical_drift_pct == 5.0  # 5% critical threshold
