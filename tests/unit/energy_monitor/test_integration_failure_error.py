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


class TestIntegrationFailureError:
    """Test IntegrationFailureError exception."""

    def test_exception_is_raised(self) -> NoReturn:
        """Test that IntegrationFailureError can be raised."""
        with pytest.raises(IntegrationFailureError):
            raise IntegrationFailureError("Test error")

    def test_exception_inherits_from_exception(self) -> None:
        """Test that IntegrationFailureError is an Exception."""
        assert issubclass(IntegrationFailureError, Exception)

    def test_exception_message(self) -> None:
        """Test that exception message is preserved."""
        msg = "Critical energy drift detected"
        try:
            raise IntegrationFailureError(msg)
        except IntegrationFailureError as e:
            assert msg in str(e)
