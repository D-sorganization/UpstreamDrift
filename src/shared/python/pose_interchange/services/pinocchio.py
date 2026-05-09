"""Pinocchio :class:`LiveKinematicsService` implementation (Subtask 3 of #4895).

Lazily imports :mod:`pinocchio` and loads a URDF via
:func:`pinocchio.buildModelFromUrdf`.  If the wheel is unavailable,
:func:`create_pinocchio_service` falls back to a
:class:`MockKinematicsService` configured with
``engine_name="pinocchio"``.

Method bodies that require non-trivial Pinocchio wiring currently
raise :class:`NotImplementedError` with a TODO tied to follow-up #4963
issue against the EPIC #4895 Pose Studio engine bridge.
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

ENGINE_NAME = "pinocchio"


_PINOCCHIO_CAPABILITIES = ServiceCapabilities(
    supports_dynamics_step=True,
    supports_collision_query=False,
    supports_realtime=True,
)


def _pinocchio_is_importable() -> bool:
    """Return ``True`` if :mod:`pinocchio` can be imported.

    Resilient to test conftests that stub the module without setting
    ``__spec__`` (``find_spec`` raises ``ValueError`` in that case).
    """
    try:
        return importlib.util.find_spec("pinocchio") is not None
    except (ImportError, ValueError):
        return False


class PinocchioKinematicsService:
    """Real-engine :class:`LiveKinematicsService` backed by :mod:`pinocchio`."""

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
        # TODO(#4963): pinocchio.buildModelFromUrdf(str(path))
        # then pinocchio.Data(model).
        raise NotImplementedError(
            "PinocchioKinematicsService.load is not yet wired; tracked by "
            "the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        self._pose = pose
        # TODO(#4963): adapter -> q vector;
        # pinocchio.forwardKinematics(model, data, q).
        raise NotImplementedError(
            "PinocchioKinematicsService.set_pose is not yet wired; tracked by "
            "the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        # TODO(#4963): iterate model.frames and pull
        # data.oMf[i] / data.oMi[joint_id] into 4x4 SE(3).
        raise NotImplementedError(
            "PinocchioKinematicsService.get_link_transforms is not yet wired; "
            "tracked by the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        # TODO(#4963): aba/forwardDynamics + symplectic Euler.
        raise NotImplementedError(
            "PinocchioKinematicsService.step is not yet wired; tracked by "
            "the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def reset(self) -> None:
        self._pose = None
        # TODO(#4963): zero q, v back to neutral.

    def capabilities(self) -> ServiceCapabilities:
        return _PINOCCHIO_CAPABILITIES


def create_pinocchio_service() -> LiveKinematicsService:
    """Return a Pinocchio service if the wheel is installed, else mock."""
    if _pinocchio_is_importable():
        logger.debug(
            "create_pinocchio_service: pinocchio importable, returning real service."
        )
        return PinocchioKinematicsService()
    logger.info(
        "create_pinocchio_service: pinocchio not importable, "
        "falling back to MockKinematicsService(engine_name=%r).",
        ENGINE_NAME,
    )
    mock: LiveKinematicsService = MockKinematicsService(engine_name=ENGINE_NAME)
    return mock


__all__ = [
    "ENGINE_NAME",
    "PinocchioKinematicsService",
    "create_pinocchio_service",
]
