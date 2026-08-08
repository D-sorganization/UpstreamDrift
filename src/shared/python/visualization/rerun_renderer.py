"""Rerun adapter for canonical viewport overlay payloads.

Executes ADR-0027's recorded follow-up ("add Rerun export if recorded review
artifacts become a first-class workflow") — epic #8390, D1/#8405. Consumes
the backend-neutral :class:`ViewportOverlayPayload` (world Z-up, SI) and logs
it to a Rerun recording: a scrubbable timeline with the trajectory line
strip, per-frame trajectory point, marker point clouds, and contact-anchored
force arrows. Recordings can be persisted as ``.rrd`` review artifacts.

``rerun-sdk`` is an opt-in dependency (the ``visualization`` extra). When it
is absent, :func:`require_rerun` raises :class:`RerunNotAvailableError` with
an install hint — importing this module never fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from .viewport import ViewportOverlayPayload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.shared.python.simulation_backends.protocol import Trace

_INSTALL_HINT = (
    "rerun-sdk is not installed. Install the visualization extra: "
    "pip install 'upstream-drift[visualization]'"
)

# Entity paths for the logged scene, kept stable so saved .rrd artifacts
# remain comparable across versions.
_ENTITY_ROOT = "overlay"
_ENTITY_TRAJECTORY = f"{_ENTITY_ROOT}/trajectory"
_ENTITY_TRAJECTORY_POINT = f"{_ENTITY_ROOT}/trajectory_point"
_ENTITY_MARKERS = f"{_ENTITY_ROOT}/markers"
_ENTITY_WRENCH = f"{_ENTITY_ROOT}/wrench"
_TIMELINE = "time"


class RerunNotAvailableError(RuntimeError):
    """Raised when the rerun SDK is required but not importable."""


def rerun_available() -> bool:
    """Whether the ``rerun`` module is importable (mock-tolerant probe)."""
    try:
        return find_spec("rerun") is not None
    except (ValueError, ModuleNotFoundError):
        return False


def require_rerun() -> ModuleType:
    """Import and return ``rerun``, raising with an install hint if absent."""
    if not rerun_available():
        raise RerunNotAvailableError(_INSTALL_HINT)
    return import_module("rerun")


@dataclass(frozen=True)
class RerunRenderConfig:
    """Configuration for a Rerun overlay recording.

    Attributes:
        application_id: Rerun application id for the recording.
        spawn_viewer: Launch the local Rerun viewer and stream to it.
        rrd_path: When set, persist the recording to this ``.rrd`` file.
    """

    application_id: str = "upstream_drift.overlay"
    spawn_viewer: bool = False
    rrd_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.application_id:
            raise ValueError("application_id must be non-empty")
        if not self.spawn_viewer and self.rrd_path is None:
            raise ValueError(
                "recording needs a sink: set spawn_viewer=True and/or rrd_path"
            )


def render_overlay_payload(
    payload: ViewportOverlayPayload,
    config: RerunRenderConfig,
    *,
    rr: ModuleType | None = None,
) -> dict[str, object]:
    """Log ``payload`` to a Rerun recording per ``config``.

    Args:
        payload: Validated canonical overlay payload (world Z-up, SI).
        config: Recording sinks and identity.
        rr: Injected rerun-compatible module (tests); defaults to the real
            SDK via :func:`require_rerun`.

    Returns:
        Summary dict: entities logged, frame count, and the rrd path (if
        persisted).

    Raises:
        RerunNotAvailableError: When rerun-sdk is absent and ``rr`` is not
            injected.
    """
    module = rr if rr is not None else require_rerun()

    recording = module.RecordingStream(application_id=config.application_id)
    if config.spawn_viewer:
        recording.spawn()
    if config.rrd_path is not None:
        recording.save(str(config.rrd_path))

    # The payload contract is world Z-up (right-handed, SI).
    recording.log(
        _ENTITY_ROOT,
        module.ViewCoordinates.RIGHT_HAND_Z_UP,
        static=True,
    )
    # Full trajectory as a static line strip for spatial context.
    recording.log(
        _ENTITY_TRAJECTORY,
        module.LineStrips3D([payload.trajectory_xyz]),
        static=True,
    )

    n = payload.time_s.shape[0]
    marker_labels = list(payload.marker_names) or None
    for i in range(n):
        _set_time(recording, float(payload.time_s[i]))
        recording.log(
            _ENTITY_TRAJECTORY_POINT,
            module.Points3D([payload.trajectory_xyz[i]]),
        )
        if payload.markers_xyz is not None:
            recording.log(
                _ENTITY_MARKERS,
                module.Points3D(payload.markers_xyz[i], labels=marker_labels),
            )
        if payload.wrench is not None:
            origins = (
                payload.contact_points_xyz[i]
                if payload.contact_points_xyz is not None
                else np.zeros((1, 3))
            )
            force = payload.wrench[i, :3]
            vectors = np.broadcast_to(force, origins.shape)
            recording.log(
                _ENTITY_WRENCH,
                module.Arrows3D(origins=origins, vectors=vectors),
            )

    return {
        "frames": n,
        "entities": [
            _ENTITY_TRAJECTORY,
            _ENTITY_TRAJECTORY_POINT,
            *([_ENTITY_MARKERS] if payload.markers_xyz is not None else []),
            *([_ENTITY_WRENCH] if payload.wrench is not None else []),
        ],
        "rrd_path": str(config.rrd_path) if config.rrd_path else None,
    }


def export_trace_rrd(
    trace: Trace,
    rrd_path: Path,
    *,
    marker_names: tuple[str, ...] = (),
    rr: ModuleType | None = None,
) -> dict[str, object]:
    """Export a simulation ``Trace`` as a scrubbable ``.rrd`` artifact.

    First-class consumer for the Rerun provider: builds the canonical
    overlay payload via :meth:`ViewportOverlayPayload.from_trace` and logs
    it to ``rrd_path``.
    """
    payload = ViewportOverlayPayload.from_trace(trace, marker_names=marker_names)
    config = RerunRenderConfig(rrd_path=Path(rrd_path))
    return render_overlay_payload(payload, config, rr=rr)


def _set_time(recording: object, seconds: float) -> None:
    """Set the timeline cursor across rerun SDK generations."""
    if hasattr(recording, "set_time"):
        recording.set_time(_TIMELINE, duration=seconds)
    else:
        # Pre-0.23 streams expose set_time_seconds instead.
        recording.set_time_seconds(_TIMELINE, seconds)
