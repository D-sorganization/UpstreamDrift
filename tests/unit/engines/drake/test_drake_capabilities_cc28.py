"""Capability declaration tests for the Drake physics engine (CC-28)."""

from __future__ import annotations

import pytest

from src.engines.physics_engines.drake.python.drake_physics_engine import (
    DrakePhysicsEngine,
)
from src.shared.python.engine_core.capabilities import CapabilityLevel

pytestmark = pytest.mark.unit


def test_drake_physics_engine_reports_autodiff_contact_capabilities() -> None:
    engine = DrakePhysicsEngine.__new__(DrakePhysicsEngine)

    caps = engine.get_capabilities()

    assert caps.engine_name == "Drake"
    assert caps.forward_sim == CapabilityLevel.FULL
    assert caps.inverse_dynamics == CapabilityLevel.FULL
    assert caps.contact_forces == CapabilityLevel.FULL
    assert caps.contact_step == CapabilityLevel.FULL
    assert caps.state_control_gradients == CapabilityLevel.FULL
    assert caps.parameter_gradients == CapabilityLevel.PARTIAL
    assert caps.trajectory_opt == CapabilityLevel.FULL
    assert caps.extra["gradient_scalar"] == "AutoDiffXd"
    assert caps.extra["contact_model"] == "hydroelastic_or_point_contact"
