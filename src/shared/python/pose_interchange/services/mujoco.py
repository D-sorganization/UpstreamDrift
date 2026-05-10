"""MuJoCo :class:`LiveKinematicsService` implementation.

Lazily imports :mod:`mujoco` and loads an MJCF file via
:func:`mujoco.MjModel.from_xml_path`.  If the wheel is unavailable,
:func:`create_mujoco_service` falls back to a
:class:`MockKinematicsService` configured with ``engine_name="mujoco"``.

The real bridge wires:

- :meth:`load` -> ``MjModel.from_xml_path`` + ``MjData(model)``.
- :meth:`set_pose` -> build a model-aware ``qpos`` from the canonical
  pose using the MuJoCo joint-address table, then ``mj_forward`` to
  update derived quantities (``xpos`` / ``xmat``).
- :meth:`get_link_transforms` -> iterate ``model.nbody`` and read
  ``data.xpos[i]`` / ``data.xmat[i].reshape(3, 3)`` into 4x4 SE(3).
- :meth:`step` -> set ``model.opt.timestep`` and call ``mj_step``.
- :meth:`reset` -> ``mj_resetData`` then re-apply the last pose.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_interchange.adapters._base import (
    build_default_joint_layout,
    encode_joint_angles,
    euler_xyz_deg_to_quat_wxyz,
)
from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.pose_interchange.live_kinematics import (
    LiveKinematicsService,
    ServiceCapabilities,
)
from src.shared.python.pose_interchange.protocol import JointSlot
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


def _configured_joint_layout(model: Any) -> Mapping[str, JointSlot] | None:
    if hasattr(model, "joint_layout") and isinstance(model.joint_layout, Mapping):
        return model.joint_layout
    if isinstance(model, Mapping) and isinstance(model.get("joint_layout"), Mapping):
        return model["joint_layout"]
    return None


def _discover_joint_layout(
    model: Any,
    mujoco_module: Any,
) -> Mapping[str, JointSlot]:
    configured_layout = _configured_joint_layout(model)
    if configured_layout is not None:
        return configured_layout

    layout_by_engine_name = {
        slot.engine_name: canonical_name
        for canonical_name, slot in build_default_joint_layout(
            base_offset=0, units="rad", sign=1, name_prefix="mj_"
        ).items()
    }
    layout_by_engine_name.update(
        {
            canonical_name: canonical_name
            for canonical_name in layout_by_engine_name.values()
        }
    )

    discovered: dict[str, JointSlot] = {}
    njnt = int(getattr(model, "njnt", 0))
    jnt_qposadr = getattr(model, "jnt_qposadr", ())
    jnt_type = getattr(model, "jnt_type", ())
    free_joint_kind = int(mujoco_module.mjtJoint.mjJNT_FREE)
    for joint_index in range(njnt):
        joint_name = mujoco_module.mj_id2name(
            model,
            mujoco_module.mjtObj.mjOBJ_JOINT,
            joint_index,
        )
        if not joint_name:
            continue
        canonical_name = layout_by_engine_name.get(str(joint_name))
        if canonical_name is None:
            continue
        if int(jnt_type[joint_index]) == free_joint_kind:
            continue
        discovered[canonical_name] = JointSlot(
            canonical_name=canonical_name,
            engine_name=str(joint_name),
            start_index=int(jnt_qposadr[joint_index]),
            length=1,
            units="rad",
            sign=1,
        )
    return discovered


def _free_joint_qpos_address(model: Any, mujoco_module: Any) -> int | None:
    njnt = int(getattr(model, "njnt", 0))
    jnt_qposadr = getattr(model, "jnt_qposadr", ())
    jnt_type = getattr(model, "jnt_type", ())
    free_joint_kind = int(mujoco_module.mjtJoint.mjJNT_FREE)
    for joint_index in range(njnt):
        if int(jnt_type[joint_index]) == free_joint_kind:
            return int(jnt_qposadr[joint_index])
    return None


def _canonical_pose_to_qpos(
    model: Any,
    pose: CanonicalPose,
    mujoco_module: Any,
) -> npt.NDArray[np.float64]:
    nq = int(model.nq)
    qpos = np.zeros(nq, dtype=np.float64)

    free_joint_qpos_address = _free_joint_qpos_address(model, mujoco_module)
    if free_joint_qpos_address is not None:
        free_joint_stop = free_joint_qpos_address + 7
        if free_joint_stop > nq:
            raise RuntimeError(
                "MuJoCoKinematicsService.set_pose: free joint overruns qpos "
                f"(start={free_joint_qpos_address}, nq={nq})."
            )
        qpos[free_joint_qpos_address : free_joint_qpos_address + 3] = (
            pose.pelvis_translation_m
        )
        qpos[free_joint_qpos_address + 3 : free_joint_stop] = (
            euler_xyz_deg_to_quat_wxyz(pose.pelvis_rotation_xyz_deg)
        )
    elif np.any(pose.pelvis_translation_m) or np.any(pose.pelvis_rotation_xyz_deg):
        logger.debug(
            "MuJoCoKinematicsService.set_pose: model has no free joint; "
            "ignoring canonical pelvis pose."
        )

    joint_layout = _discover_joint_layout(model, mujoco_module)
    if joint_layout:
        encode_joint_angles(pose.joint_angles_deg, joint_layout, qpos)
    return qpos


class MuJoCoKinematicsService:
    """Real-engine :class:`LiveKinematicsService` backed by :mod:`mujoco`."""

    engine_name: str = ENGINE_NAME

    def __init__(self) -> None:
        self._model: Any = None
        self._data: Any = None
        self._pose: CanonicalPose | None = None

    def _require_loaded(self, op: str) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError(
                f"MuJoCoKinematicsService.{op}: model not loaded; call load() first."
            )

    def load(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            raise TypeError(
                f"model_path must be a pathlib.Path, got {type(model_path).__name__}"
            )
        import mujoco  # type: ignore[import-not-found]

        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)
        logger.debug(
            "MuJoCoKinematicsService loaded model_path=%s nq=%d nbody=%d",
            model_path,
            getattr(self._model, "nq", -1),
            getattr(self._model, "nbody", -1),
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        self._require_loaded("set_pose")
        self._pose = pose
        import mujoco  # type: ignore[import-not-found]

        qpos = _canonical_pose_to_qpos(self._model, pose, mujoco)
        self._data.qpos[:] = qpos  # type: ignore[union-attr]
        mujoco.mj_forward(self._model, self._data)

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        self._require_loaded("get_link_transforms")
        import mujoco  # type: ignore[import-not-found]

        transforms: dict[str, npt.NDArray[np.float64]] = {}
        nbody = int(self._model.nbody)  # type: ignore[union-attr]
        for i in range(nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is None or name == "":
                name = f"body_{i}"
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = np.asarray(
                self._data.xmat[i],  # type: ignore[union-attr]
                dtype=np.float64,
            ).reshape(3, 3)
            transform[:3, 3] = np.asarray(
                self._data.xpos[i],  # type: ignore[union-attr]
                dtype=np.float64,
            )
            transforms[name] = transform
        return transforms

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        self._require_loaded("step")
        import mujoco  # type: ignore[import-not-found]

        self._model.opt.timestep = float(dt)  # type: ignore[union-attr]
        mujoco.mj_step(self._model, self._data)

    def reset(self) -> None:
        self._pose = None
        if self._model is None or self._data is None:
            return
        import mujoco  # type: ignore[import-not-found]

        mujoco.mj_resetData(self._model, self._data)

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
