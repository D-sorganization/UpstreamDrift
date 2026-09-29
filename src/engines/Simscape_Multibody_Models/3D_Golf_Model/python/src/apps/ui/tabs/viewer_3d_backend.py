"""Renderer backend policy for the C3D 3D viewer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

FPS_TARGET = 60
CMU_ACCEPTANCE_MARKERS = 45
CMU_ACCEPTANCE_FRAMES = 1000

PARITY_FEATURES: tuple[str, ...] = (
    "scrubbing",
    "speed_control",
    "loop",
    "marker_groups",
    "view_presets",
    "skeleton_overlay",
)


class RendererBackend(StrEnum):
    """Supported C3D 3D-viewer renderer backends."""

    PYQTGL = "pyqtgl"
    MATPLOTLIB = "matplotlib"


@dataclass(frozen=True, slots=True)
class Viewer3DBackendDecision:
    """Backend decision and acceptance contract for one C3D dataset."""

    backend: RendererBackend
    reason: str
    target_fps: int
    marker_count: int
    frame_count: int
    parity_features: tuple[str, ...] = PARITY_FEATURES

    def __post_init__(self) -> None:
        if not isinstance(self.backend, RendererBackend):
            raise TypeError("backend must be a RendererBackend")
        if not self.reason:
            raise ValueError("reason must be non-empty")
        _validate_positive_int("target_fps", self.target_fps)
        _validate_positive_int("marker_count", self.marker_count)
        _validate_positive_int("frame_count", self.frame_count)
        if not self.parity_features:
            raise ValueError("parity_features must be non-empty")


def select_viewer_3d_backend(
    *,
    prefer_gpu: bool,
    pyqtgl_available: bool,
    marker_count: int,
    frame_count: int,
) -> Viewer3DBackendDecision:
    """Choose the renderer backend for a C3D 3D-viewer dataset.

    Postcondition: the returned decision always carries the 60 fps target
    and the feature-parity checklist required before replacing the
    matplotlib path.
    """
    _validate_positive_int("marker_count", marker_count)
    _validate_positive_int("frame_count", frame_count)
    if not isinstance(prefer_gpu, bool):
        raise TypeError("prefer_gpu must be bool")
    if not isinstance(pyqtgl_available, bool):
        raise TypeError("pyqtgl_available must be bool")

    if not prefer_gpu:
        backend = RendererBackend.MATPLOTLIB
        reason = "matplotlib fallback explicitly requested"
    elif pyqtgl_available:
        backend = RendererBackend.PYQTGL
        reason = "pyqtgraph.opengl available"
    else:
        backend = RendererBackend.MATPLOTLIB
        reason = "pyqtgraph.opengl unavailable; using matplotlib fallback"

    return Viewer3DBackendDecision(
        backend=backend,
        reason=reason,
        target_fps=FPS_TARGET,
        marker_count=marker_count,
        frame_count=frame_count,
    )


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
