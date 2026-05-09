"""MuJoCo :class:`LiveKinematicsService` implementation (Subtask 3 of #4895).

Lazily imports :mod:`mujoco` and loads an MJCF file via
:func:`mujoco.MjModel.from_xml_path`.  If the wheel is unavailable,
:func:`create_mujoco_service` falls back to a
:class:`MockKinematicsService` configured with ``engine_name="mujoco"``.

Method bodies that require non-trivial MuJoCo wiring currently raise
:class:`NotImplementedError`; full body-transform queries land in a
follow-up issue against the EPIC #4895 Pose Studio engine bridge.
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

ENGINE_NAME = "mujoco"


_MUJOCO_CAPABILITIES = ServiceCapabilities(
    supports_dynamics_step=True,
    supports_collision_query=True,
    supports_realtime=True,
)


def _mujoco_is_importable() -> bool:
    """Return ``True`` if :mod:`mujoco` can be imported.

    Resilient to test conftests that stub the module without setting
    ``__spec__`` (``find_spec`` raises ``ValueError`` in that case).
    """
    try:
        return importlib.util.find_spec("mujoco") is not None
    except (ImportError, ValueError):
        return False


class MuJoCoKinematicsService:
    """Real-engine :class:`LiveKinematicsService` backed by :mod:`mujoco`."""

    engine_name: str = ENGINE_NAME

    def __init__(self) -> None:
        self._model: object | None = None
        self._data: object | None = None
        self._pose: CanonicalPose | None = None

    def load(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            raise TypeError(
                f"model_path must be a pathlib.Path, got {type(model_path).__name__}"
            )
        # TODO(#4963): mujoco.MjModel.from_xml_path(str(path))
        # then mujoco.MjData(model).
        raise NotImplementedError(
            "MuJoCoKinematicsService.load is not yet wired; tracked by "
            "the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        self._pose = pose
        # TODO(#4963): adapter -> data.qpos[:] = q; mj_forward.
        raise NotImplementedError(
            "MuJoCoKinematicsService.set_pose is not yet wired; tracked by "
            "the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        # TODO(#4963): iterate model.nbody and read
        # data.xpos[i] / data.xmat[i].reshape(3, 3) into 4x4 SE(3).
        raise NotImplementedError(
            "MuJoCoKinematicsService.get_link_transforms is not yet wired; "
            "tracked by the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        # TODO(#4963): set model.opt.timestep and call mj_step.
        raise NotImplementedError(
            "MuJoCoKinematicsService.step is not yet wired; tracked by "
            "the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def reset(self) -> None:
        self._pose = None
        # TODO(#4963): mj_resetData(model, data).

    def capabilities(self) -> ServiceCapabilities:
        return _MUJOCO_CAPABILITIES


def create_mujoco_service() -> LiveKinematicsService:
    """Return a MuJoCo service if :mod:`mujoco` is installed, else mock."""
    if _mujoco_is_importable():
        logger.debug(
            "create_mujoco_service: mujoco importable, returning real service."
        )
        return MuJoCoKinematicsService()
    logger.info(
        "create_mujoco_service: mujoco not importable, "
        "falling back to MockKinematicsService(engine_name=%r).",
        ENGINE_NAME,
    )
    mock: LiveKinematicsService = MockKinematicsService(engine_name=ENGINE_NAME)
    return mock


__all__ = [
    "ENGINE_NAME",
    "MuJoCoKinematicsService",
    "create_mujoco_service",
]
