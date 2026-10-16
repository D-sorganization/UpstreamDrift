"""Named camera presets for the MuJoCo viewport (UD #12885).

Single source of truth for the preset names shown by the Controls tab quick
buttons and the Visualization tab combo, and for the free-camera pose each one
applies in :meth:`MuJoCoSimWidget.set_camera`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = ["CAMERA_PRESETS", "CameraPreset", "camera_preset"]


@dataclass(frozen=True)
class CameraPreset:
    """Free-camera orientation: azimuth/elevation in degrees, distance in m."""

    azimuth: float
    elevation: float
    distance: float


CAMERA_PRESETS: Mapping[str, CameraPreset] = MappingProxyType(
    {
        "side": CameraPreset(90.0, -20.0, 3.0),
        "front": CameraPreset(0.0, -20.0, 3.0),
        "top": CameraPreset(90.0, -90.0, 4.0),
        "follow": CameraPreset(135.0, -15.0, 2.5),
        "down-the-line": CameraPreset(180.0, -10.0, 3.5),
    }
)


def camera_preset(name: str) -> CameraPreset:
    """Return the preset called ``name``.

    Raises:
        ValueError: if ``name`` is not a known preset.
    """
    try:
        return CAMERA_PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown camera preset {name!r}; expected one of {sorted(CAMERA_PRESETS)}"
        ) from None
