"""Drake LiveKinematicsService.

Loads a MultibodyPlant via `Parser`, maps :class:`CanonicalPose` to ``q`` using
the :class:`DrakeAdapter`, and returns world-frame transforms
from ``plant.EvalBodyPoseInWorld``. ``step()`` advances a Drake :class:`Simulator`;
``reset()`` restores a stored default :class:`Context`.

Falls back to :class:`MockKinematicsService` when the Drake wheel is not installed.
"""

from __future__ import annotations

from collections.abc import Mapping
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
    """Real-engine :class:`LiveKinematicsService` backed by :mod:`pydrake`."""

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
        self._plant.SetPositions(self._plant_context, self._positions_for(pose))

    def _positions_for(self, pose: CanonicalPose) -> npt.NDArray[np.float64]:
        """Build a valid ``q`` for the loaded plant from a canonical pose.

        ``DrakeAdapter.from_canonical`` only understands a ``joint_layout``
        mapping, not a ``MultibodyPlant``, and its free-flyer ``[xyz, rpy]``
        prefix is not Drake's ``[quat_wxyz, xyz]`` floating-base layout.
        Start from the plant's current (valid-quaternion) positions, place
        the floating base through ``SetFreeBodyPose``, then write any
        single-DOF joint whose name is a canonical joint field.
        """
        from pydrake.math import (  # type: ignore[import-not-found]
            RigidTransform,
            RollPitchYaw,
        )
        from pydrake.multibody.tree import BodyIndex  # type: ignore[import-not-found]

        plant = self._plant
        floating = sorted(int(i) for i in plant.GetFloatingBaseBodies())
        if floating:
            plant.SetFreeBodyPose(
                self._plant_context,
                plant.get_body(BodyIndex(floating[0])),
                RigidTransform(
                    RollPitchYaw(np.radians(pose.pelvis_rotation_xyz_deg)),
                    np.asarray(pose.pelvis_translation_m, dtype=np.float64),
                ),
            )
        q = np.array(plant.GetPositions(self._plant_context), dtype=np.float64)
        for joint_index in plant.GetJointIndices():
            joint = plant.get_joint(joint_index)
            angle_deg = pose.joint_angles_deg.get(joint.name())
            if angle_deg is not None and joint.num_positions() == 1:
                q[joint.position_start()] = np.radians(angle_deg)
        return q

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        if (
            getattr(self, "_plant", None) is None
            or getattr(self, "_context", None) is None
        ):
            raise RuntimeError(
                "DrakeKinematicsService.get_link_transforms: model not loaded; call load() first."
            )

        from pydrake.multibody.tree import BodyIndex  # type: ignore[import-not-found]

        transforms: dict[str, npt.NDArray[np.float64]] = {}
        for body_index in range(self._plant.num_bodies()):
            body = self._plant.get_body(BodyIndex(body_index))
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

    def joint_limits(self) -> Mapping[str, tuple[float, float]]:
        """Return an empty mapping: no anatomical joint-limit data is
        wired for this engine yet (issue #8887)."""
        return {}


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
