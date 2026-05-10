"""MuJoCo :class:`LiveKinematicsService` implementation.

Lazily imports :mod:`mujoco` and loads an MJCF file via
:func:`mujoco.MjModel.from_xml_path`.  If the wheel is unavailable,
:func:`create_mujoco_service` falls back to a
:class:`MockKinematicsService` configured with ``engine_name="mujoco"``.

The real bridge wires:

- :meth:`load` -> ``MjModel.from_xml_path`` + ``MjData(model)``.
- :meth:`set_pose` -> :class:`MujocoAdapter` to convert the canonical
  pose to a ``qpos`` vector, then ``mj_forward`` to update derived
  quantities (``xpos`` / ``xmat``).
- :meth:`get_link_transforms` -> iterate ``model.nbody`` and read
  ``data.xpos[i]`` / ``data.xmat[i].reshape(3, 3)`` into 4x4 SE(3).
- :meth:`step` -> set ``model.opt.timestep`` and call ``mj_step``.
- :meth:`reset` -> ``mj_resetData`` then re-apply the last pose.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_interchange.adapters.mujoco import MujocoAdapter
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
        self._model: Any = None
        self._data: Any = None
        self._pose: CanonicalPose | None = None
        self._adapter = MujocoAdapter()

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

        q_full = self._adapter.from_canonical(pose)
        nq = int(self._model.nq)  # type: ignore[union-attr]
        # Copy what fits; pad with zeros if adapter q is shorter than nq.
        qpos = np.zeros(nq, dtype=float)
        n = min(nq, q_full.shape[0])
        qpos[:n] = q_full[:n]
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
