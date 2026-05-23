"""Drake :class:`LiveKinematicsService` implementation (Subtask 3 of #4895).

The real implementation lazily imports :mod:`pydrake` and loads a URDF
via the multibody plant's :class:`pydrake.multibody.parsing.Parser`.
If :mod:`pydrake` is unavailable, :func:`create_drake_service` falls
back to a :class:`MockKinematicsService` configured with
``engine_name="drake"``.

Method bodies that require non-trivial Drake plumbing currently raise
:class:`NotImplementedError` with a TODO(#4963) tied to follow-up;
this PR only commits the wiring scaffold so downstream code can target
``LiveKinematicsService`` without waiting on the full Drake bridge.
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
        self._plant: Any = None
        self._context: Any = None
        self._pose: CanonicalPose | None = None

    def load(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            raise TypeError(
                f"model_path must be a pathlib.Path, got {type(model_path).__name__}"
            )
        import pydrake.multibody.parsing as parsing  # type: ignore[import-not-found]
        import pydrake.multibody.plant as plant  # type: ignore[import-not-found]
        import pydrake.systems.framework as framework  # type: ignore[import-not-found]
        import pydrake.systems.analysis as analysis  # type: ignore[import-not-found]

        self._builder = framework.DiagramBuilder()
        self._plant, self._scene_graph = plant.AddMultibodyPlantSceneGraph(
            self._builder, time_step=0.001
        )
        parsing.Parser(self._plant).AddModels(model_path)
        self._plant.Finalize()
        self._diagram = self._builder.Build()

        self._context = self._diagram.CreateDefaultContext()
        self._plant_context = self._plant.GetMyContextFromRoot(self._context)

        self._neutral_context = self._diagram.CreateDefaultContext()
        self._simulator = analysis.Simulator(self._diagram, self._context)
        self._simulator.Initialize()

        logger.debug(
            "DrakeKinematicsService loaded model_path=%s num_positions=%d",
            model_path,
            self._plant.num_positions(),
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        if (
            getattr(self, "_plant", None) is None
            or getattr(self, "_context", None) is None
        ):
            raise RuntimeError(
                "DrakeKinematicsService.set_pose: model not loaded; call load() first."
            )
        self._pose = pose
        from src.shared.python.pose_interchange.adapters.drake import DrakeAdapter

        adapter = DrakeAdapter()
        q_full = adapter.from_canonical(pose, model=self._plant)

        nq = int(self._plant.num_positions())
        q = np.zeros(nq, dtype=np.float64)
        n = min(nq, q_full.shape[0])
        q[:n] = q_full[:n]
        self._plant.SetPositions(self._plant_context, q)

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        if (
            getattr(self, "_plant", None) is None
            or getattr(self, "_context", None) is None
        ):
            raise RuntimeError(
                "DrakeKinematicsService.get_link_transforms: model not loaded; call load() first."
            )

        transforms: dict[str, npt.NDArray[np.float64]] = {}
        for body_index in self._plant.GetBodyIndices(
            self._plant.world_model_instance()
        ):
            body = self._plant.get_body(body_index)
            name = body.name()
            pose_in_world = self._plant.EvalBodyPoseInWorld(self._plant_context, body)
            transforms[name] = pose_in_world.GetAsMatrix4()
        return transforms

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        if (
            getattr(self, "_context", None) is None
            or getattr(self, "_simulator", None) is None
        ):
            raise RuntimeError(
                "DrakeKinematicsService.step: model not loaded; call load() first."
            )
        current_time = self._context.get_time()
        self._simulator.AdvanceTo(current_time + dt)

    def reset(self) -> None:
        self._pose = None
        if (
            getattr(self, "_context", None) is None
            or getattr(self, "_neutral_context", None) is None
        ):
            return

        self._context.SetTimeStateAndParametersFrom(self._neutral_context)
        if getattr(self, "_simulator", None) is not None:
            self._simulator.Initialize()

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
