"""Live kinematics service Protocol for cross-engine pose evaluation.

The :class:`LiveKinematicsService` Protocol defines the runtime surface
that the Pose Studio tool uses to push a :class:`CanonicalPose` into a
running engine and read back link transforms.  Each engine ships one
implementation under :mod:`pose_interchange.services` (Subtask 3 of
EPIC #4895).  When an engine wheel is not installed, the registry falls
back to a headless :class:`MockKinematicsService` that derives link
transforms from the canonical forward-kinematics evaluator
(:func:`forward_kinematics`).

Design by contract:

- ``set_pose`` accepts only :class:`CanonicalPose` instances and stores
  the pose for subsequent ``get_link_transforms`` calls.
- ``get_link_transforms`` returns a mapping of link name to a 4x4
  ``np.float64`` SE(3) matrix.
- ``step`` MUST raise :class:`CapabilityError` when the service's
  :class:`ServiceCapabilities` reports ``supports_dynamics_step is
  False``.  Mock services never advance dynamics.
- ``capabilities`` returns a frozen, deterministic descriptor; callers
  can branch on it without ever invoking a method that would raise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from src.shared.python.pose_interchange.canonical import CanonicalPose


@dataclass(frozen=True, slots=True)
class ServiceCapabilities:
    """Static capability descriptor for a :class:`LiveKinematicsService`.

    Parameters
    ----------
    supports_dynamics_step
        ``True`` if the service can advance time via :meth:`step`.
        Pure-kinematic mocks return ``False`` and must raise
        :class:`CapabilityError` from :meth:`step`.
    supports_collision_query
        ``True`` if the service can answer collision/contact queries.
        Reserved for future use; currently informational only.
    supports_realtime
        ``True`` if the service can be advanced in soft-real-time
        (i.e. ``step(dt)`` returns within roughly ``dt`` wall-clock
        seconds for typical ``dt``).  Informational only.
    """

    supports_dynamics_step: bool
    supports_collision_query: bool
    supports_realtime: bool


class CapabilityError(RuntimeError):
    """Raised when a :class:`LiveKinematicsService` method is unsupported.

    Used by services whose :class:`ServiceCapabilities` declare a feature
    as unsupported (e.g. :meth:`LiveKinematicsService.step` on a
    kinematic-only mock).  Inherits :class:`RuntimeError` so generic
    error handlers still catch it.
    """


@runtime_checkable
class LiveKinematicsService(Protocol):
    """Engine-agnostic live kinematics surface.

    Implementations live under :mod:`pose_interchange.services`, one
    file per engine, with a headless ``MockKinematicsService`` fallback
    for environments where the engine wheel is not installed.

    All implementations declare :attr:`engine_name` as a class
    attribute so :data:`KINEMATICS_SERVICE_REGISTRY` can dispatch on it.
    """

    engine_name: str

    def load(self, model_path: Path) -> None:
        """Load the engine's model file.

        For Drake / Pinocchio this is a URDF; MuJoCo expects MJCF;
        OpenSim expects ``.osim``.  Mock services accept any path
        (including ``Path(".")``) and ignore the contents.
        """

    def set_pose(self, pose: CanonicalPose) -> None:
        """Push a :class:`CanonicalPose` into the engine state."""

    def get_link_transforms(self) -> dict[str, npt.NDArray[np.float64]]:
        """Return the world-frame transform of every named link.

        The mapping key is the engine's link/body name (or, for the mock
        service, the canonical landmark name); the value is a
        ``(4, 4)`` ``np.float64`` SE(3) matrix.
        """

    def step(self, dt: float) -> None:
        """Advance the engine by ``dt`` seconds.

        Raises
        ------
        CapabilityError
            If :meth:`capabilities` reports
            ``supports_dynamics_step is False``.
        """

    def reset(self) -> None:
        """Reset the engine to its post-load state."""

    def capabilities(self) -> ServiceCapabilities:
        """Return the static :class:`ServiceCapabilities` descriptor."""


__all__ = [
    "CapabilityError",
    "LiveKinematicsService",
    "ServiceCapabilities",
]
