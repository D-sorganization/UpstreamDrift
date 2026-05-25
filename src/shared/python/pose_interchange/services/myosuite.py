"""MyoSuite :class:`LiveKinematicsService` implementation.

Lazily imports :mod:`myosuite` (or its shim) and loads a model. If the wheel is
unavailable, :func:`create_myosuite_service` falls back to a
:class:`MockKinematicsService` configured with ``engine_name="myosuite"``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.pose_interchange.live_kinematics import (
    LiveKinematicsService,
    ServiceCapabilities,
)
from src.shared.python.pose_interchange.services._mock import (
    MockKinematicsService,
)

logger = get_logger(__name__)

ENGINE_NAME = "myosuite"


def _myosuite_is_importable() -> bool:
    """Check if the myosuite engine wheel is present."""
    # Depending on how the codebase manages it, we mock it.
    return importlib.util.find_spec("myosuite") is not None


class MyoSuiteKinematicsService(LiveKinematicsService):
    """Real MyoSuite kinematics service (placeholder wiring)."""

    engine_name: str = ENGINE_NAME
    _capabilities: ServiceCapabilities = ServiceCapabilities(
        supports_dynamics_step=False,
        supports_collision_query=False,
        supports_realtime=False,
    )

    def __init__(self) -> None:
        pass

    def capabilities(self) -> ServiceCapabilities:
        return self._capabilities

    def load(self, model_path: str | Path) -> None:
        raise NotImplementedError("Real MyoSuite implementation pending.")

    def set_pose(self, pose: CanonicalPose) -> None:
        raise NotImplementedError("Real MyoSuite implementation pending.")

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        raise NotImplementedError("Real MyoSuite implementation pending.")

    def step(self, dt: float) -> None:
        raise NotImplementedError("Real MyoSuite implementation pending.")

    def reset(self) -> None:
        raise NotImplementedError("Real MyoSuite implementation pending.")


def create_myosuite_service() -> LiveKinematicsService:
    """Return a real MyoSuite service if available, else a mock."""
    if _myosuite_is_importable():
        return MyoSuiteKinematicsService()
    logger.debug(
        "myosuite not found, falling back to MockKinematicsService(engine_name=%r).",
        ENGINE_NAME,
    )
    mock: LiveKinematicsService = MockKinematicsService(engine_name=ENGINE_NAME)
    return mock
