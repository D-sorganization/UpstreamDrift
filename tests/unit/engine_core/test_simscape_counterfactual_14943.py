"""Simscape skeleton satisfies extended DynamicsInterface (#14943 slice 3)."""

from __future__ import annotations

import pytest

from src.engines.simscape import SimscapeAdapter
from src.shared.python.engine_core.interfaces import PhysicsEngine

pytestmark = pytest.mark.unit


def test_simscape_adapter_satisfies_physics_engine_protocol() -> None:
    adapter = SimscapeAdapter()
    assert isinstance(adapter, PhysicsEngine)
    assert hasattr(adapter, "compute_gravity_only_acceleration")
    assert hasattr(adapter, "compute_control_acceleration_at_state")
