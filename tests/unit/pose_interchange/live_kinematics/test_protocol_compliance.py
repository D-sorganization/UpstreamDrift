"""Every registered service must satisfy :class:`LiveKinematicsService`."""

from __future__ import annotations

import pytest

from src.shared.python.pose_interchange.live_kinematics import (
    LiveKinematicsService,
)
from src.shared.python.pose_interchange.services import (
    KINEMATICS_SERVICE_REGISTRY,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("engine_name", sorted(KINEMATICS_SERVICE_REGISTRY))
def test_registry_factory_returns_protocol_compliant_service(
    engine_name: str,
) -> None:
    """Each registered factory yields an instance that satisfies the Protocol.

    ``LiveKinematicsService`` is a runtime-checkable Protocol, so this
    test catches accidental missing methods or wrong attribute names in
    any per-engine module.
    """
    factory = KINEMATICS_SERVICE_REGISTRY[engine_name]
    service = factory()
    assert isinstance(service, LiveKinematicsService), (
        f"Factory for engine_name={engine_name!r} returned "
        f"{type(service).__name__} which does not satisfy "
        "LiveKinematicsService."
    )
    # Protocol attributes that must be present and match the engine name.
    assert hasattr(service, "engine_name")
    assert service.engine_name == engine_name


def test_registry_covers_all_first_class_engines() -> None:
    """Every first-class engine (per CROSS_ENGINE_PARITY_SPEC) is registered."""
    expected = {"drake", "mujoco", "pinocchio", "opensim", "simscape"}
    assert set(KINEMATICS_SERVICE_REGISTRY) == expected
