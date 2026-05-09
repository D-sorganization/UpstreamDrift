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


class TestEnergySnapshot:
    """Test EnergySnapshot dataclass."""

    def test_energy_monitor_initialization(self) -> None:
        """Test basic initialization."""
        snapshot = EnergySnapshot(time=1.0, kinetic=5.0, potential=10.0)
        assert snapshot.time == 1.0
        assert snapshot.kinetic == 5.0
        assert snapshot.potential == 10.0

    @pytest.mark.parametrize(
        "kinetic, potential, expected_total",
        [
            (3.0, 7.0, 10.0),
            (15.0, -5.0, 10.0),
            (0.0, 0.0, 0.0),
            (5.0, 5.0, 10.0),
            (100.0, 0.0, 100.0),
            (0.0, 100.0, 100.0),
        ],
        ids=[
            "positive_both",
            "negative_potential",
            "zero_energy",
            "equal_components",
            "kinetic_only",
            "potential_only",
        ],
    )
    def test_total_energy(self, kinetic, potential, expected_total) -> None:
        """Test that total property returns KE + PE for various inputs."""
        snapshot = EnergySnapshot(time=0.0, kinetic=kinetic, potential=potential)
        assert snapshot.total == expected_total

    def test_total_is_computed_property(self) -> None:
        """Test that total is a computed property, not stored."""
        snapshot = EnergySnapshot(time=0.0, kinetic=5.0, potential=5.0)
        assert snapshot.total == 10.0

        # Modify kinetic energy
        snapshot.kinetic = 8.0
        # Total should update automatically
        assert snapshot.total == 13.0
