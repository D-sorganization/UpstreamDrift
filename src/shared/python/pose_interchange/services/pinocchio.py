"""Pinocchio :class:`LiveKinematicsService` implementation.

Lazily imports :mod:`pinocchio` and loads a URDF via
:func:`pinocchio.buildModelFromUrdf`.

Behaviour with vs. without the real wheel
-----------------------------------------

:func:`create_pinocchio_service` is the **intentional, single point of
fallback** for this engine:

* **Wheel-present** (``pip install pin`` -- the upstream Stack-of-Tasks
  build) -- returns a fully-featured :class:`PinocchioKinematicsService`
  backed by ``pinocchio.aba``, ``pinocchio.forwardKinematics``, etc.
* **Wheel-absent** (no wheel installed, or PyPI placeholder installed)
  -- there is an unrelated package named ``pinocchio`` on PyPI (an abandoned
  0.1 placeholder) that lacks :func:`buildModelFromUrdf`. We detect both
  cases in :func:`_pinocchio_is_importable` and fall back to a
  :class:`MockKinematicsService` configured with ``engine_name="pinocchio"``.
  The mock returns deterministic identity transforms, satisfies the
  :class:`LiveKinematicsService` protocol, and keeps the service registry
  consistent so callers do not have to branch on wheel availability.

The fallback is exercised by
:mod:`tests.unit.pose_interchange.live_kinematics.test_registry_fallback`
and the real-wheel path by
:mod:`tests.integration.pose_interchange.services.test_pinocchio_real`.

Real-bridge wiring
------------------

- :meth:`load` -> ``pin.buildModelFromUrdf`` + ``model.createData()``.
- :meth:`set_pose` -> :class:`PinocchioAdapter` to convert canonical to
  ``q``; ``pin.forwardKinematics`` + ``pin.updateFramePlacements`` so
  that ``data.oMf`` is fresh.
- :meth:`get_link_transforms` -> iterate ``model.frames`` and read
  ``data.oMf[i]`` SE(3) into a 4x4 matrix.
- :meth:`step` -> Euler integration via ``pin.aba`` (forward
  dynamics) + ``pin.integrate``.
- :meth:`reset` -> restore neutral q, zero v.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_interchange.adapters.pinocchio import PinocchioAdapter
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
    """Return ``True`` if :mod:`pinocchio` is the real wheel, not the PyPI placeholder.

    The upstream Stack-of-Tasks ``pinocchio`` wheel exposes
    :func:`buildModelFromUrdf` and :func:`forwardKinematics`. The
    unrelated ``pinocchio`` 0.1 placeholder package on PyPI does not.
    We treat the placeholder as "not importable" so
    :func:`create_pinocchio_service` falls back to the mock service.
    """
    try:
        if importlib.util.find_spec("pinocchio") is None:
            return False
    except (ImportError, ValueError):
        return False
    try:
        import pinocchio as pin  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False
    return hasattr(pin, "buildModelFromUrdf") and hasattr(pin, "forwardKinematics")


class PinocchioKinematicsService:
    """Real-engine :class:`LiveKinematicsService` backed by :mod:`pinocchio`."""

    engine_name: str = ENGINE_NAME

    def __init__(self) -> None:
        self._model: Any = None
        self._data: Any = None
        self._pose: CanonicalPose | None = None
        self._neutral_q: npt.NDArray[np.float64] | None = None
        self._v: npt.NDArray[np.float64] | None = None
        self._q: npt.NDArray[np.float64] | None = None
        self._adapter = PinocchioAdapter()

    def _require_loaded(self, op: str) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError(
                f"PinocchioKinematicsService.{op}: model not loaded; call load() first."
            )

    def load(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            raise TypeError(
                f"model_path must be a pathlib.Path, got {type(model_path).__name__}"
            )
        import pinocchio as pin  # type: ignore[import-not-found]

        self._model = pin.buildModelFromUrdf(str(model_path))
        self._data = self._model.createData()  # type: ignore[union-attr]
        # Cache neutral configuration for reset().
        try:
            self._neutral_q = np.asarray(
                pin.neutral(self._model), dtype=np.float64
            ).copy()
        except Exception:  # noqa: BLE001
            nq = int(getattr(self._model, "nq", 0))
            self._neutral_q = np.zeros(nq, dtype=np.float64)
        nv = int(getattr(self._model, "nv", 0))
        self._v = np.zeros(nv, dtype=np.float64)
        self._q = self._neutral_q.copy()
        logger.debug(
            "PinocchioKinematicsService loaded model_path=%s nq=%d nv=%d",
            model_path,
            int(getattr(self._model, "nq", -1)),
            nv,
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        self._require_loaded("set_pose")
        import pinocchio as pin  # type: ignore[import-not-found]

        self._pose = pose
        q_full = self._adapter.from_canonical(pose)
        nq = int(self._model.nq)  # type: ignore[union-attr]
        q = np.zeros(nq, dtype=np.float64)
        n = min(nq, q_full.shape[0])
        q[:n] = q_full[:n]
        self._q = q
        pin.forwardKinematics(self._model, self._data, q)
        # Refresh frame placements so data.oMf is current.
        if hasattr(pin, "updateFramePlacements"):
            pin.updateFramePlacements(self._model, self._data)

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        self._require_loaded("get_link_transforms")

        transforms: dict[str, npt.NDArray[np.float64]] = {}
        frames = getattr(self._model, "frames", [])
        oMf = getattr(self._data, "oMf", None)
        if oMf is None:
            return transforms
        n_frames = len(frames)
        for i in range(n_frames):
            frame = frames[i]
            name = getattr(frame, "name", f"frame_{i}")
            placement = oMf[i]
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = np.asarray(
                placement.rotation, dtype=np.float64
            ).reshape(3, 3)
            transform[:3, 3] = np.asarray(
                placement.translation, dtype=np.float64
            ).reshape(3)
            transforms[str(name)] = transform
        return transforms

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        self._require_loaded("step")
        import pinocchio as pin  # type: ignore[import-not-found]

        if self._q is None or self._v is None:
            raise RuntimeError(
                "PinocchioKinematicsService.step: state vectors uninitialised; "
                "call load() first."
            )
        nv = int(self._model.nv)  # type: ignore[union-attr]
        tau = np.zeros(nv, dtype=np.float64)
        a = pin.aba(self._model, self._data, self._q, self._v, tau)
        a = np.asarray(a, dtype=np.float64).reshape(nv)
        # Symplectic Euler: v += a*dt; q = integrate(q, v*dt).
        self._v = self._v + a * float(dt)
        self._q = np.asarray(
            pin.integrate(self._model, self._q, self._v * float(dt)),
            dtype=np.float64,
        )
        pin.forwardKinematics(self._model, self._data, self._q)
        if hasattr(pin, "updateFramePlacements"):
            pin.updateFramePlacements(self._model, self._data)

    def reset(self) -> None:
        self._pose = None
        if self._neutral_q is None or self._model is None or self._data is None:
            return
        import pinocchio as pin  # type: ignore[import-not-found]

        self._q = self._neutral_q.copy()
        nv = int(getattr(self._model, "nv", 0))
        self._v = np.zeros(nv, dtype=np.float64)
        pin.forwardKinematics(self._model, self._data, self._q)
        if hasattr(pin, "updateFramePlacements"):
            pin.updateFramePlacements(self._model, self._data)

    def capabilities(self) -> ServiceCapabilities:
        return _PINOCCHIO_CAPABILITIES


def create_pinocchio_service() -> LiveKinematicsService:
    """Return a Pinocchio service if the wheel is installed, else mock.

    Intentional fallback: when the real ``pinocchio`` wheel is missing
    -- or when the unrelated PyPI placeholder (``pinocchio`` 0.1, which
    lacks :func:`buildModelFromUrdf`) is the only thing importable --
    callers still get a working :class:`MockKinematicsService` with
    ``engine_name="pinocchio"`` so the registry stays consistent.
    """
    if _pinocchio_is_importable():
        logger.debug(
            "create_pinocchio_service: pinocchio importable, returning real service."
        )
        return PinocchioKinematicsService()
    logger.info(
        "create_pinocchio_service: pinocchio wheel not importable "
        "(missing, or PyPI 0.1 placeholder); falling back to "
        "MockKinematicsService(engine_name=%r).",
        ENGINE_NAME,
    )
    mock: LiveKinematicsService = MockKinematicsService(engine_name=ENGINE_NAME)
    return mock


__all__ = [
    "ENGINE_NAME",
    "PinocchioKinematicsService",
    "create_pinocchio_service",
]
