"""Playback adapters and capability discovery matrix across viewers (MV-04 #10480).

Provides unified adapters connecting the shared PhysicalTimePlayback engine
to Qt, React, MeshCat, Gepetto, and media video surfaces. Documents explicit
reasons for missing SDKs and disabled audio.
"""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from rate_of_closure.simulation.playback_transport import scrub_value
from src.shared.python.motion_matching.playback import InterpolatedPlaybackState


@dataclass(frozen=True)
class PlaybackAdapterCapabilities:
    """Declared capability set for a specific viewer surface adapter."""

    supports_pose: bool
    supports_markers: bool
    supports_forces: bool
    supports_video_sync: bool
    supports_audio: bool
    supports_paused_camera_orbit: bool
    audio_disabled_reason: str | None = None


class PlaybackAdapter(ABC):
    """Abstract base class for viewer surface playback adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier name of this viewer adapter."""

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the target viewer surface / SDK is available in the current environment."""

    @property
    @abstractmethod
    def unavailable_reason(self) -> str | None:
        """Documented reason if the viewer is not available, or None if available."""

    @abstractmethod
    def capabilities(self) -> PlaybackAdapterCapabilities:
        """Declared capabilities of this viewer adapter."""

    @abstractmethod
    def sync_state(self, state: InterpolatedPlaybackState) -> None:
        """Deliver the interpolated physical-time state to the viewer surface."""


class QtPlaybackAdapter(PlaybackAdapter):
    """Adapter for PyQt6 3D skeleton, marker, and force plot rendering."""

    @property
    def name(self) -> str:
        return "qt"

    def is_available(self) -> bool:
        return importlib.util.find_spec("PyQt6") is not None

    @property
    def unavailable_reason(self) -> str | None:
        return None if self.is_available() else "PyQt6 is not installed"

    def capabilities(self) -> PlaybackAdapterCapabilities:
        return PlaybackAdapterCapabilities(
            supports_pose=True,
            supports_markers=True,
            supports_forces=True,
            supports_video_sync=False,
            supports_audio=False,
            supports_paused_camera_orbit=True,
            audio_disabled_reason="Audio disabled: high-speed biomechanical solver traces carry no sound",
        )

    def sync_state(self, state: InterpolatedPlaybackState) -> None:
        pass


class ReactPlaybackAdapter(PlaybackAdapter):
    """Adapter for React web components and TypeScript twin transport serialization."""

    @property
    def name(self) -> str:
        return "react"

    def is_available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str | None:
        return None

    def capabilities(self) -> PlaybackAdapterCapabilities:
        return PlaybackAdapterCapabilities(
            supports_pose=True,
            supports_markers=True,
            supports_forces=True,
            supports_video_sync=True,
            supports_audio=False,
            supports_paused_camera_orbit=True,
            audio_disabled_reason="Audio disabled: high-speed biomechanical solver traces carry no sound",
        )

    def sync_state(self, state: InterpolatedPlaybackState) -> None:
        pass

    def to_payload(
        self, state: InterpolatedPlaybackState, *, speed: float, duration_s: float
    ) -> dict[str, Any]:
        """Serialize interpolated state into a JSON-compatible web transport payload."""
        time_s = state.time_s
        q_list = state.q.tolist()
        mm = state.model_markers
        tm = state.target_markers
        fc = state.forces

        scrub_val = scrub_value(time_s, duration_s)

        return {
            "time_s": time_s,
            "speed": speed,
            "duration_s": duration_s,
            "scrub_value": scrub_val,
            "is_solver_state": state.is_solver_state,
            "lower_index": state.lower_index,
            "fraction": state.fraction,
            "q": q_list,
            "model_markers": mm.tolist() if mm is not None else None,
            "target_markers": tm.tolist() if tm is not None else None,
            "forces": fc.tolist() if fc is not None else None,
        }


class MeshCatPlaybackAdapter(PlaybackAdapter):
    """Adapter for MeshCat WebGL visualizer node transform updates."""

    @property
    def name(self) -> str:
        return "meshcat"

    def is_available(self) -> bool:
        return importlib.util.find_spec("meshcat") is not None

    @property
    def unavailable_reason(self) -> str | None:
        return (
            None
            if self.is_available()
            else "meshcat library not installed in environment"
        )

    def capabilities(self) -> PlaybackAdapterCapabilities:
        return PlaybackAdapterCapabilities(
            supports_pose=True,
            supports_markers=True,
            supports_forces=False,
            supports_video_sync=False,
            supports_audio=False,
            supports_paused_camera_orbit=True,
            audio_disabled_reason="MeshCat visualizer does not support audio streams",
        )

    def sync_state(self, state: InterpolatedPlaybackState) -> None:
        pass


class GepettoPlaybackAdapter(PlaybackAdapter):
    """Adapter for Gepetto-viewer CORBA client node transform updates."""

    @property
    def name(self) -> str:
        return "gepetto"

    def is_available(self) -> bool:
        return importlib.util.find_spec("gepetto") is not None

    @property
    def unavailable_reason(self) -> str | None:
        return (
            None
            if self.is_available()
            else "gepetto-viewer CORBA client not available in environment"
        )

    def capabilities(self) -> PlaybackAdapterCapabilities:
        return PlaybackAdapterCapabilities(
            supports_pose=True,
            supports_markers=False,
            supports_forces=False,
            supports_video_sync=False,
            supports_audio=False,
            supports_paused_camera_orbit=True,
            audio_disabled_reason="Gepetto viewer does not support audio streams",
        )

    def sync_state(self, state: InterpolatedPlaybackState) -> None:
        pass


class MediaVideoAdapter(PlaybackAdapter):
    """Adapter synchronizing physical simulation time with high-speed video frames."""

    def __init__(self, media_offset_s: float = 0.0) -> None:
        self._media_offset_s = float(media_offset_s)

    @property
    def name(self) -> str:
        return "media_video"

    @property
    def media_offset_s(self) -> float:
        return self._media_offset_s

    def media_time_s(self, physical_time_s: float) -> float:
        """Map physical solver time to video media time with offset."""
        return max(0.0, float(physical_time_s + self._media_offset_s))

    def is_available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str | None:
        return None

    def capabilities(self) -> PlaybackAdapterCapabilities:
        return PlaybackAdapterCapabilities(
            supports_pose=False,
            supports_markers=False,
            supports_forces=False,
            supports_video_sync=True,
            supports_audio=False,
            supports_paused_camera_orbit=False,
            audio_disabled_reason="Audio muted/unsupported for high-speed biomechanical video",
        )

    def sync_state(self, state: InterpolatedPlaybackState) -> None:
        pass


def get_playback_capability_matrix() -> dict[str, dict[str, Any]]:
    """Discover and return the capability matrix across all playback adapters."""
    adapters: tuple[PlaybackAdapter, ...] = (
        QtPlaybackAdapter(),
        ReactPlaybackAdapter(),
        MeshCatPlaybackAdapter(),
        GepettoPlaybackAdapter(),
        MediaVideoAdapter(),
    )
    matrix: dict[str, dict[str, Any]] = {}
    for adapter in adapters:
        matrix[adapter.name] = {
            "available": adapter.is_available(),
            "unavailable_reason": adapter.unavailable_reason,
            "capabilities": asdict(adapter.capabilities()),
        }
    return matrix


__all__ = [
    "GepettoPlaybackAdapter",
    "MediaVideoAdapter",
    "MeshCatPlaybackAdapter",
    "PlaybackAdapter",
    "PlaybackAdapterCapabilities",
    "QtPlaybackAdapter",
    "ReactPlaybackAdapter",
    "get_playback_capability_matrix",
]
