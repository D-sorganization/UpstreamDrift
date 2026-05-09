"""Drake :class:`LiveKinematicsService` implementation (Subtask 3 of #4895).

The real implementation lazily imports :mod:`pydrake` and loads a URDF
via the multibody plant's :class:`pydrake.multibody.parsing.Parser`.
If :mod:`pydrake` is unavailable, :func:`create_drake_service` falls
back to a :class:`MockKinematicsService` configured with
``engine_name="drake"``.

Method bodies that require non-trivial Drake plumbing currently raise
:class:`NotImplementedError` with a TODO tied to follow-up #4963;
this PR only commits the wiring scaffold so downstream code can target
``LiveKinematicsService`` without waiting on the full Drake bridge.
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

ENGINE_NAME = "drake"


_DRAKE_CAPABILITIES = ServiceCapabilities(
    supports_dynamics_step=True,
    supports_collision_query=True,
    supports_realtime=False,
)


def _drake_is_importable() -> bool:
    """Return ``True`` if :mod:`pydrake` can be imported in this env.

    Some test conftests stub ``pydrake`` into ``sys.modules`` without
    a ``__spec__``; ``find_spec`` raises ``ValueError`` in that case
    and we conservatively report ``False`` (no real engine wheel).
    """
    try:
        return importlib.util.find_spec("pydrake") is not None
    except (ImportError, ValueError):
        return False


class DrakeKinematicsService:
    """Real-engine :class:`LiveKinematicsService` backed by :mod:`pydrake`.

    Skeletal: full body-transform queries are wired up but require a
    follow-up issue to land alongside the Pose Studio engine bridge.
    """

    engine_name: str = ENGINE_NAME

    def __init__(self) -> None:
        self._plant: object | None = None
        self._context: object | None = None
        self._pose: CanonicalPose | None = None

    def load(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            raise TypeError(
                f"model_path must be a pathlib.Path, got {type(model_path).__name__}"
            )
        # TODO(#4963): build MultibodyPlant + Parser, parse
        # the URDF, finalize, and stash the plant + a default context.
        raise NotImplementedError(
            "DrakeKinematicsService.load is not yet wired; tracked by "
            "the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        self._pose = pose
        # TODO(#4963): use the Drake adapter (Subtask 2) to
        # encode the canonical pose into the plant's q vector and write
        # it into the cached context.
        raise NotImplementedError(
            "DrakeKinematicsService.set_pose is not yet wired; tracked by "
            "the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        # TODO(#4963): iterate plant.GetBodyIndices() and call
        # plant.EvalBodyPoseInWorld(context, body) for each to build the
        # SE(3) dict.
        raise NotImplementedError(
            "DrakeKinematicsService.get_link_transforms is not yet wired; "
            "tracked by the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        # TODO(#4963): advance a Simulator bound to the plant.
        raise NotImplementedError(
            "DrakeKinematicsService.step is not yet wired; tracked by "
            "the EPIC #4895 Pose Studio engine-bridge follow-up."
        )

    def reset(self) -> None:
        self._pose = None
        # TODO(#4963): restore the cached default context.

    def capabilities(self) -> ServiceCapabilities:
        return _DRAKE_CAPABILITIES


def create_drake_service() -> LiveKinematicsService:
    """Return a Drake service if :mod:`pydrake` is installed, else mock."""
    if _drake_is_importable():
        logger.debug(
            "create_drake_service: pydrake importable, returning real service."
        )
        return DrakeKinematicsService()
    logger.info(
        "create_drake_service: pydrake not importable, "
        "falling back to MockKinematicsService(engine_name=%r).",
        ENGINE_NAME,
    )
    # MockKinematicsService satisfies the LiveKinematicsService Protocol
    # via structural typing.
    mock: LiveKinematicsService = MockKinematicsService(engine_name=ENGINE_NAME)
    return mock


__all__ = [
    "ENGINE_NAME",
    "DrakeKinematicsService",
    "create_drake_service",
]
