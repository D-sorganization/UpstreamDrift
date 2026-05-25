"""Registry-fallback contract: missing wheel -> :class:`MockKinematicsService`.

When an engine wheel is not importable in the test environment, the
registry factory must return a :class:`MockKinematicsService` whose
``engine_name`` matches the registry key.  We exercise this by
forcing the per-engine ``_*_is_importable`` probe to return ``False``
via :func:`unittest.mock.patch` so the test works in any environment,
whether or not real wheels are installed.
"""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import patch

import pytest

from src.shared.python.pose_interchange.services import (
    KINEMATICS_SERVICE_REGISTRY,
)
from src.shared.python.pose_interchange.services._mock import (
    MockKinematicsService,
)

pytestmark = pytest.mark.unit


# (engine_name, dotted module path, name of the importable-probe symbol).
_FALLBACK_PROBES: Iterable[tuple[str, str, str]] = (
    (
        "drake",
        "src.shared.python.pose_interchange.services.drake",
        "_drake_is_importable",
    ),
    (
        "mujoco",
        "src.shared.python.pose_interchange.services.mujoco",
        "_mujoco_is_importable",
    ),
    (
        "myosuite",
        "src.shared.python.pose_interchange.services.myosuite",
        "_myosuite_is_importable",
    ),
    (
        "pinocchio",
        "src.shared.python.pose_interchange.services.pinocchio",
        "_pinocchio_is_importable",
    ),
    (
        "opensim",
        "src.shared.python.pose_interchange.services.opensim",
        "_opensim_is_importable",
    ),
    (
        "simscape",
        "src.shared.python.pose_interchange.services.simscape",
        "_matlab_engine_is_importable",
    ),
)


@pytest.mark.parametrize(("engine_name", "module_path", "probe_name"), _FALLBACK_PROBES)
def test_registry_falls_back_to_mock_when_wheel_absent(
    engine_name: str,
    module_path: str,
    probe_name: str,
) -> None:
    """Force the wheel probe to ``False`` and assert mock-fallback contract."""
    factory = KINEMATICS_SERVICE_REGISTRY[engine_name]
    with patch(f"{module_path}.{probe_name}", return_value=False):
        service = factory()
    assert isinstance(service, MockKinematicsService), (
        f"Expected MockKinematicsService fallback for {engine_name!r}, "
        f"got {type(service).__name__}."
    )
    assert service.engine_name == engine_name


@pytest.mark.parametrize("engine_name", sorted(KINEMATICS_SERVICE_REGISTRY))
def test_registry_factory_resolvable(engine_name: str) -> None:
    """Each registry entry resolves without raising (mock or real)."""
    factory = KINEMATICS_SERVICE_REGISTRY[engine_name]
    service = factory()
    assert hasattr(service, "engine_name")
    assert service.engine_name == engine_name
