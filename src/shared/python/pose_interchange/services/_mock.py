"""Headless :class:`LiveKinematicsService` fallback.

The :class:`MockKinematicsService` is a pure-Python pass-through used
when an engine wheel is unavailable.  It stores the most recent
:class:`CanonicalPose` set via :meth:`set_pose` and synthesises link
transforms on demand from the canonical forward-kinematics evaluator
in :mod:`motion_matching.diagnostics.forward_kinematics`.

The mock's transform set keys are the canonical landmark names
returned by :func:`forward_kinematics` (``pelvis``, ``spine_top``,
``torso_top``, ``l_shoulder``, ``r_shoulder``, ``l_elbow``,
``r_elbow``, ``l_wrist``, ``r_wrist``, ``l_hand``, ``r_hand``,
``butt``, ``clubhead``).  Because the mock is kinematic-only, every
landmark is reported with rotation = identity and translation = the
landmark's Cartesian position; this is enough for Pose Studio to draw
a skeleton without spinning up a real engine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.diagnostics.forward_kinematics import (
    forward_kinematics,
)
from src.shared.python.pose_interchange.canonical import CanonicalPose
from src.shared.python.pose_interchange.live_kinematics import (
    CapabilityError,
    ServiceCapabilities,
)

logger = get_logger(__name__)


_MOCK_CAPABILITIES = ServiceCapabilities(
    supports_dynamics_step=False,
    supports_collision_query=False,
    supports_realtime=False,
)


class MockKinematicsService:
    """Headless :class:`LiveKinematicsService` implementation.

    Parameters
    ----------
    engine_name
        The engine name this mock is impersonating.  Used by the
        registry so callers asking for ``"drake"`` get a service whose
        :attr:`engine_name` is ``"drake"`` even when the real wheel is
        absent.
    """

    def __init__(self, engine_name: str) -> None:
        if not isinstance(engine_name, str):
            raise TypeError(
                f"engine_name must be a str, got {type(engine_name).__name__}"
            )
        if not engine_name:
            raise ValueError("engine_name must be a non-empty str")
        self.engine_name: str = engine_name
        self._pose: CanonicalPose | None = None
        self._model_path: Path | None = None

    # ---- LiveKinematicsService surface ---------------------------------

    def load(self, model_path: Path) -> None:
        """Pretend to load *model_path*.

        The mock service stores the path for diagnostic purposes only;
        the file does not need to exist.
        """
        if not isinstance(model_path, Path):
            raise TypeError(
                f"model_path must be a pathlib.Path, got {type(model_path).__name__}"
            )
        self._model_path = model_path
        logger.debug(
            "MockKinematicsService(%s) loaded model_path=%s",
            self.engine_name,
            model_path,
        )

    def set_pose(self, pose: CanonicalPose) -> None:
        """Store *pose* for subsequent :meth:`get_link_transforms` calls."""
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        self._pose = pose

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return canonical landmark positions as 4x4 SE(3) matrices.

        Each landmark is reported with identity rotation and the
        landmark's Cartesian position as the translation column.
        """
        if self._pose is None:
            # No pose has been set: derive from the all-zero canonical
            # pose so callers always get a well-formed dict.
            angles: dict[str, float] = {}
        else:
            angles = self._pose.angles_full_dict_deg()
        skeleton = forward_kinematics(angles)
        transforms: dict[str, npt.NDArray[np.float64]] = {}
        for name, point in skeleton.points.items():
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = np.asarray(point, dtype=np.float64)
            transforms[name] = transform
        return transforms

    def step(self, dt: float) -> None:
        """Always raises :class:`CapabilityError` for the mock.

        The mock is kinematic-only and reports
        ``supports_dynamics_step=False``.
        """
        raise CapabilityError(
            f"MockKinematicsService(engine_name={self.engine_name!r}) "
            "does not support dynamics step; "
            f"got step(dt={dt!r}). Install the real engine wheel "
            "or branch on capabilities().supports_dynamics_step."
        )

    def reset(self) -> None:
        """Forget the last :meth:`set_pose` value."""
        self._pose = None

    def capabilities(self) -> ServiceCapabilities:
        """Return the mock's capabilities (all ``False``)."""
        return _MOCK_CAPABILITIES


__all__ = ["MockKinematicsService"]
