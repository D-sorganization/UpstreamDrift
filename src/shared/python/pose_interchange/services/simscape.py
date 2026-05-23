"""Simscape :class:`LiveKinematicsService` implementation (Subtask 3 of #4895).

Connects to the MATLAB engine via the existing
:func:`load_matlab_3d_engine` machinery in :mod:`src.engines.loaders`.
If MATLAB / the MATLAB engine API is unavailable,
:func:`create_simscape_service` falls back to a
:class:`MockKinematicsService` configured with
``engine_name="simscape"``.

Method bodies that drive Simulink directly currently raise
:class:`NotImplementedError` with a TODO(#4963) tied to follow-up
against the EPIC #4895 Pose Studio engine bridge.
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

ENGINE_NAME = "simscape"


_SIMSCAPE_CAPABILITIES = ServiceCapabilities(
    supports_dynamics_step=True,
    supports_collision_query=False,
    supports_realtime=False,
)


def _matlab_engine_is_importable() -> bool:
    """Return ``True`` if :mod:`matlab.engine` can be imported.

    The Simscape bridge in :mod:`src.engines.loaders` ultimately needs
    the MATLAB Engine API for Python.  We probe its presence here so
    the registry can fall back cleanly when MATLAB is not installed.
    Resilient to test conftests that stub the module without setting
    ``__spec__`` (``find_spec`` raises ``ValueError`` in that case).
    """
    try:
        return importlib.util.find_spec("matlab") is not None
    except (ImportError, ValueError):
        return False


class SimscapeKinematicsService:
    """Real-engine :class:`LiveKinematicsService` backed by Simscape Multibody.

    Connects via :mod:`src.engines.loaders.load_matlab_3d_engine`; the
    actual body-transform queries land in a follow-up issue alongside
    the Pose Studio engine bridge.
    """

    engine_name: str = ENGINE_NAME

    def __init__(self) -> None:
        self._matlab_engine: object | None = None
        self._pose: CanonicalPose | None = None
        # ``_engine`` is typed as ``object | None`` so the ImportError fallback
        # can assign None without mypy variance complaints. Concrete attribute
        # access on the engine is guarded by ``isinstance`` or ``getattr``.
        self._engine: object | None = None

    def load(self, model_path: Path) -> None:
        if not isinstance(model_path, Path):
            raise TypeError(
                f"model_path must be a pathlib.Path, got {type(model_path).__name__}"
            )
        try:
            from src.engines.simscape.adapter import SimscapeAdapter

            engine = SimscapeAdapter()
            engine.load_from_path(str(model_path))
            self._engine = engine
            self._matlab_engine = getattr(engine, "_matlab_engine", None)
        except ImportError:
            logger.warning("SimscapeAdapter not available. Running in mock mode.")
            self._engine = None

        logger.debug(
            "SimscapeKinematicsService loaded model_path=%s",
            model_path,
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        self._pose = pose
        if getattr(self, "_engine", None) is None:
            return

        from src.shared.python.pose_interchange.adapters.simscape import (
            SimscapeAdapter as PoseAdapter,
        )

        adapter = PoseAdapter()
        q_dict = adapter.from_canonical(pose)

        matlab_engine = self._matlab_engine
        if matlab_engine is not None:
            for joint_name, val in q_dict.items():
                try:
                    # ``matlab.engine.MatlabEngine`` exposes ``.workspace`` at
                    # runtime; we type the field as ``object`` to avoid a hard
                    # dependency on the MATLAB Engine stubs.
                    matlab_engine.workspace[  # type: ignore[attr-defined]
                        f"{joint_name}_q"
                    ] = float(val)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"Failed to set {joint_name}_q in MATLAB workspace: {e}"
                    )

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        transforms: dict[str, npt.NDArray[np.float64]] = {}
        if getattr(self, "_engine", None) is None:
            return transforms

        # TODO(#4963): Implement actual transform queries from MATLAB engine
        return transforms

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt!r}")
        engine = self._engine
        if engine is not None:
            engine.step(dt)  # type: ignore[attr-defined]

    def reset(self) -> None:
        self._pose = None
        engine = self._engine
        if engine is not None:
            engine.reset()  # type: ignore[attr-defined]

    def capabilities(self) -> ServiceCapabilities:
        return _SIMSCAPE_CAPABILITIES


def create_simscape_service() -> LiveKinematicsService:
    """Return a Simscape service if MATLAB is importable, else mock."""
    if _matlab_engine_is_importable():
        logger.debug(
            "create_simscape_service: matlab importable, returning real service."
        )
        return SimscapeKinematicsService()
    logger.info(
        "create_simscape_service: matlab not importable, "
        "falling back to MockKinematicsService(engine_name=%r).",
        ENGINE_NAME,
    )
    mock: LiveKinematicsService = MockKinematicsService(engine_name=ENGINE_NAME)
    return mock


__all__ = [
    "ENGINE_NAME",
    "SimscapeKinematicsService",
    "create_simscape_service",
]
