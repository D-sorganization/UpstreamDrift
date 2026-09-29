"""Capability declaration tests for MuJoCo + OpenSim engines (#7050).

Mirrors ``tests/unit/engines/drake/test_drake_capabilities_cc28.py``: every
engine adapter must report an :class:`EngineCapabilities` with accurate
``CapabilityLevel`` values. The OpenSim adapter imports its backend lazily so
its capability declaration is testable without the wheel; the MuJoCo adapter
imports ``mujoco`` at module load, so that test is gated on availability.
"""

from __future__ import annotations

import pytest

from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)
from src.shared.python.engine_core.engine_availability import (
    is_engine_available,
)

pytestmark = pytest.mark.unit


def test_opensim_reports_capabilities() -> None:
    """OpenSim declares analytic dynamics + muscle support (#7050)."""
    from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
        OpenSimPhysicsEngine,
    )

    engine = OpenSimPhysicsEngine.__new__(OpenSimPhysicsEngine)
    caps = engine.get_capabilities()

    assert isinstance(caps, EngineCapabilities)
    assert caps.engine_name == "OpenSim"
    assert caps.mass_matrix == CapabilityLevel.FULL
    assert caps.jacobian == CapabilityLevel.FULL
    assert caps.inverse_dynamics == CapabilityLevel.FULL
    assert caps.muscles == CapabilityLevel.FULL
    assert caps.forward_sim == CapabilityLevel.FULL
    # OpenSim has no first-class joint-torque contact reporting.
    assert caps.contact_forces == CapabilityLevel.NONE
    assert caps.extra["jacobian_method"] == "simbody_calcStationJacobian"


@pytest.mark.skipif(not is_engine_available("mujoco"), reason="mujoco not installed")
def test_mujoco_reports_capabilities() -> None:
    """MuJoCo declares analytic dynamics + contact support (#7050)."""
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine import (  # noqa: E501
        MuJoCoPhysicsEngine,
    )

    engine = MuJoCoPhysicsEngine.__new__(MuJoCoPhysicsEngine)
    caps = engine.get_capabilities()

    assert isinstance(caps, EngineCapabilities)
    assert caps.engine_name == "MuJoCo"
    assert caps.mass_matrix == CapabilityLevel.FULL
    assert caps.jacobian == CapabilityLevel.FULL
    assert caps.contact_forces == CapabilityLevel.FULL
    assert caps.inverse_dynamics == CapabilityLevel.FULL
    assert caps.drift_acceleration == CapabilityLevel.FULL
    assert caps.forward_sim == CapabilityLevel.FULL
    assert caps.contact_step == CapabilityLevel.FULL
    assert caps.extra["zvcf"] == "supported"
