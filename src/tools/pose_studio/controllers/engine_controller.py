"""Owns the active live-kinematics service and pose adapter.

The :class:`EngineController` is the single seam through which the GUI
swaps between physics engines without restarting.  It encapsulates:

* the active :class:`PoseConventionAdapter` (per-engine convention
  translator from Subtask 2 of EPIC #4895);
* the active :class:`LiveKinematicsService` (per-engine kinematics
  surface from Subtask 3);
* the most-recently-applied :class:`CanonicalPose`;
* the :class:`EngineStatus` (mock / live / error) for the UI status
  pill.

It is **engine-agnostic and Qt-free**, so the unit tests can construct
it directly and assert behaviour without an X server.
"""

from __future__ import annotations

import os
import time

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_interchange.adapters import ADAPTER_REGISTRY
from src.shared.python.pose_interchange.canonical import (
    CanonicalPose,
    canonical_zero_pose,
)
from src.shared.python.pose_interchange.protocol import PoseConventionAdapter
from src.shared.python.pose_interchange.services import (
    KINEMATICS_SERVICE_REGISTRY,
    MockKinematicsService,
)
from src.shared.python.pose_interchange.live_kinematics import (
    LiveKinematicsService,
)
from src.shared.python.realtime import publish as realtime_publish
from src.tools.pose_studio.core import SUPPORTED_ENGINES, EngineStatus

logger = get_logger(__name__)

# Env-var gate: realtime publishing is opt-in so unit tests and CI never
# touch the IPC layer unless they ask for it. The cross-tool demo
# (Subtask 6 of EPIC #4993) and the dedicated integration test set this
# variable to "1".
_PUBLISH_ENV_VAR = "POSE_STUDIO_PUBLISH_REALTIME"
_PUBLISH_DEBOUNCE_S = 1.0 / 30.0  # cap publish rate at ~30 Hz


class EngineController:
    """Encapsulate the active engine adapter + kinematics service.

    Parameters
    ----------
    engine_name
        Initial engine name; must be a member of :data:`SUPPORTED_ENGINES`.
    """

    def __init__(self, engine_name: str = "drake") -> None:
        if not isinstance(engine_name, str):
            raise TypeError(
                f"engine_name must be str, got {type(engine_name).__name__}"
            )
        if engine_name not in SUPPORTED_ENGINES:
            raise ValueError(
                f"engine_name {engine_name!r} not in SUPPORTED_ENGINES "
                f"{SUPPORTED_ENGINES!r}"
            )
        self._engine_name: str = engine_name
        self._pose: CanonicalPose = canonical_zero_pose()
        self._adapter: PoseConventionAdapter | None = None
        self._service: LiveKinematicsService | None = None
        self._status: EngineStatus = EngineStatus.MOCK
        self._last_error: str | None = None
        # Per-instance debounce timestamp for realtime publish; gated by
        # ``POSE_STUDIO_PUBLISH_REALTIME`` (see :meth:`_maybe_publish_pose`).
        self._last_publish_ts: float = 0.0
        self._activate(engine_name)

    # ---- public surface ------------------------------------------------

    @property
    def engine_name(self) -> str:
        """The currently active engine name."""
        return self._engine_name

    @property
    def status(self) -> EngineStatus:
        """The current :class:`EngineStatus` (mock / live / error)."""
        return self._status

    @property
    def last_error(self) -> str | None:
        """The most recent error message, or ``None`` if the controller
        is in a healthy state."""
        return self._last_error

    @property
    def adapter(self) -> PoseConventionAdapter | None:
        """The currently active :class:`PoseConventionAdapter` or ``None``
        when the controller is in :attr:`EngineStatus.ERROR`."""
        return self._adapter

    @property
    def service(self) -> LiveKinematicsService | None:
        """The currently active :class:`LiveKinematicsService` or ``None``
        when the controller is in :attr:`EngineStatus.ERROR`."""
        return self._service

    @property
    def pose(self) -> CanonicalPose:
        """The most-recently-applied :class:`CanonicalPose`."""
        return self._pose

    def switch_engine(self, engine_name: str) -> EngineStatus:
        """Switch to a different engine.

        Parameters
        ----------
        engine_name
            New engine name; must be a member of :data:`SUPPORTED_ENGINES`.

        Returns
        -------
        EngineStatus
            The status of the new engine after the swap.
        """
        if not isinstance(engine_name, str):
            raise TypeError(
                f"engine_name must be str, got {type(engine_name).__name__}"
            )
        if engine_name not in SUPPORTED_ENGINES:
            raise ValueError(
                f"engine_name {engine_name!r} not in SUPPORTED_ENGINES "
                f"{SUPPORTED_ENGINES!r}"
            )
        self._engine_name = engine_name
        self._activate(engine_name)
        # Replay the current pose through the new service so the 3D
        # view stays consistent across the swap.
        self.set_pose(self._pose)
        return self._status

    def set_pose(self, pose: CanonicalPose) -> None:
        """Apply *pose* to the active service and remember it.

        If the live engine's :meth:`set_pose` raises
        :class:`NotImplementedError` (some engine bridges are still
        partial in EPIC #4895), the controller transparently downgrades
        to a :class:`MockKinematicsService` so the rest of the GUI keeps
        working.  Other exceptions flip the controller into
        :attr:`EngineStatus.ERROR`.
        """
        if not isinstance(pose, CanonicalPose):
            raise TypeError(f"pose must be a CanonicalPose, got {type(pose).__name__}")
        self._pose = pose
        # Mirror the pose to the realtime IPC layer (gated by env var, debounced
        # to 30 Hz) so subscribing tools — e.g. the Pose Subscriber demo — see
        # live updates without each tool poking the others directly.
        self._maybe_publish_pose()
        if self._service is None:
            return
        try:
            self._service.set_pose(pose)
        except NotImplementedError as exc:
            logger.info(
                "EngineController(%s) live set_pose unimplemented; "
                "downgrading to mock: %s",
                self._engine_name,
                exc,
            )
            self._service = MockKinematicsService(self._engine_name)
            self._status = EngineStatus.MOCK
            self._last_error = None
            self._service.set_pose(pose)
        except (RuntimeError, ValueError, TypeError) as exc:
            self._status = EngineStatus.ERROR
            self._last_error = f"set_pose failed: {exc}"
            logger.warning(
                "EngineController(%s) set_pose failed: %s",
                self._engine_name,
                exc,
            )

    # ---- internals -----------------------------------------------------

    def _maybe_publish_pose(self) -> None:
        """Mirror the current pose to ``pose/canonical`` if enabled.

        Three-stage gate:
        1. The ``POSE_STUDIO_PUBLISH_REALTIME`` env var must be truthy
           (so unit tests and CI never accidentally touch the IPC).
        2. A 30 Hz debounce keeps the file transport from melting under
           a slider drag.
        3. Any exception from the realtime layer is logged and swallowed
           — pose editing must never fail because IPC is unhappy.
        """
        if not os.environ.get(_PUBLISH_ENV_VAR):
            return
        now = time.monotonic()
        if now - self._last_publish_ts < _PUBLISH_DEBOUNCE_S:
            return
        self._last_publish_ts = now
        try:
            realtime_publish("pose/canonical", self._pose.to_dict())
        except Exception:
            logger.exception("realtime publish failed")

    def _activate(self, engine_name: str) -> None:
        """Construct adapter + service for *engine_name* and cache them."""
        self._last_error = None
        try:
            adapter_cls = ADAPTER_REGISTRY[engine_name]
            factory = KINEMATICS_SERVICE_REGISTRY[engine_name]
        except KeyError as exc:  # pragma: no cover - guarded by SUPPORTED_ENGINES
            self._adapter = None
            self._service = None
            self._status = EngineStatus.ERROR
            self._last_error = f"registry miss for {engine_name!r}: {exc}"
            return

        try:
            adapter = adapter_cls()
            service = factory()
        except (RuntimeError, ValueError, TypeError, ImportError) as exc:
            self._adapter = None
            self._service = None
            self._status = EngineStatus.ERROR
            self._last_error = f"engine activation failed: {exc}"
            logger.warning(
                "EngineController failed to activate %s: %s",
                engine_name,
                exc,
            )
            return

        self._adapter = adapter
        self._service = service
        self._status = (
            EngineStatus.MOCK
            if isinstance(service, MockKinematicsService)
            else EngineStatus.LIVE
        )
        logger.debug(
            "EngineController activated engine=%s status=%s",
            engine_name,
            self._status.value,
        )


__all__ = ["EngineController"]
