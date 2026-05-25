"""OpenSim :class:`LiveKinematicsService` implementation (Subtask 3 of #4895).

Lazily imports :mod:`opensim` and loads a ``.osim`` file via
:class:`opensim.Model`.  If the wheel is unavailable,
:func:`create_opensim_service` falls back to a
:class:`MockKinematicsService` configured with
``engine_name="opensim"``.


"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

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

ENGINE_NAME = "opensim"


_OPENSIM_CAPABILITIES = ServiceCapabilities(
    supports_dynamics_step=True,
    supports_collision_query=False,
    supports_realtime=False,
)


def _opensim_is_importable() -> bool:
    """Return ``True`` if :mod:`opensim` can be imported.

    Resilient to test conftests that stub the module without setting
    ``__spec__`` (``find_spec`` raises ``ValueError`` in that case).
    """
    try:
        return importlib.util.find_spec("opensim") is not None
    except (ImportError, ValueError):
        return False


class OpenSimKinematicsService:
    """Real-engine :class:`LiveKinematicsService` backed by :mod:`opensim`."""

    engine_name: str = ENGINE_NAME

    def __init__(self) -> None:
        self._model: Any = None
        self._state: Any = None
        self._pose: CanonicalPose | None = None

    def load(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            raise TypeError(
                f"model_path must be a pathlib.Path, got {type(model_path).__name__}"
            )
        import opensim  # type: ignore[import-not-found]

        self._model = opensim.Model(str(model_path))
        self._state = self._model.initSystem()
        self._default_state = opensim.State(self._state)

        logger.debug(
            "OpenSimKinematicsService loaded model_path=%s",
            model_path,
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        if (
            getattr(self, "_model", None) is None
            or getattr(self, "_state", None) is None
        ):
            raise RuntimeError(
                "OpenSimKinematicsService.set_pose: model not loaded; call load() first."
            )

        self._pose = pose
        from src.shared.python.pose_interchange.adapters.opensim import OpenSimAdapter

        adapter = OpenSimAdapter()
        layout = adapter.joint_layout(self._model)
        q = adapter.from_canonical(pose, model=self._model)

        coord_set = self._model.updCoordinateSet()
        for slot in layout.values():
            coordinate_name = slot.engine_name
            if coord_set.contains(coordinate_name):
                coord = coord_set.get(coordinate_name)
                coord.setValue(self._state, float(q[slot.start_index]))

        self._model.realizePosition(self._state)

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        if (
            getattr(self, "_model", None) is None
            or getattr(self, "_state", None) is None
        ):
            raise RuntimeError(
                "OpenSimKinematicsService.get_link_transforms: model not loaded; call load() first."
            )

        transforms: dict[str, npt.NDArray[np.float64]] = {}
        body_set = self._model.getBodySet()
        for i in range(body_set.getSize()):
            body = body_set.get(i)
            transform_in_ground = body.getTransformInGround(self._state)

            R = transform_in_ground.R()
            p = transform_in_ground.p()

            mat = np.eye(4, dtype=np.float64)
            for r in range(3):
                for c in range(3):
                    mat[r, c] = R.get(r, c)
            mat[0, 3] = p.get(0)
            mat[1, 3] = p.get(1)
            mat[2, 3] = p.get(2)

            transforms[body.getName()] = mat

        return transforms

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        if (
            getattr(self, "_model", None) is None
            or getattr(self, "_state", None) is None
        ):
            raise RuntimeError(
                "OpenSimKinematicsService.step: model not loaded; call load() first."
            )

        import opensim  # type: ignore[import-not-found]

        manager = opensim.Manager(self._model)
        manager.initialize(self._state)
        current_time = self._state.getTime()
        manager.integrate(current_time + dt)
        self._state = manager.getState()

    def reset(self) -> None:
        self._pose = None
        if getattr(self, "_model", None) is None:
            return

        self._state = self._model.initSystem()

    def capabilities(self) -> ServiceCapabilities:
        return _OPENSIM_CAPABILITIES


def create_opensim_service() -> LiveKinematicsService:
    """Return an OpenSim service if the wheel is installed, else mock."""
    if _opensim_is_importable():
        logger.debug(
            "create_opensim_service: opensim importable, returning real service."
        )
        return OpenSimKinematicsService()
    logger.info(
        "create_opensim_service: opensim not importable, "
        "falling back to MockKinematicsService(engine_name=%r).",
        ENGINE_NAME,
    )
    mock: LiveKinematicsService = MockKinematicsService(engine_name=ENGINE_NAME)
    return mock


__all__ = [
    "ENGINE_NAME",
    "OpenSimKinematicsService",
    "create_opensim_service",
]
