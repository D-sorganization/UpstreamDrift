"""Camera presets for the putting tile's 3D view (#14473, epic #14452).

The studio's putting cameras, for the PyQt6 tile: the auto-framed orbit,
down the line (behind the ball, looking along the ball-cup line), face on
(square to the line, across from a right-handed player) and overhead. A
:class:`CameraPose` is in pyqtgraph's orbit terms, so the view applies it
as it is; :func:`camera_position` is where that puts the eye.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.tools.putting_green_gui._scene_builder import PuttScene

__all__ = ["CAMERA_PRESETS", "CameraPose", "camera_pose", "camera_position"]

CAMERA_PRESETS = ("orbit", "down_the_line", "face_on", "overhead")


@dataclass(frozen=True)
class CameraPose:
    """An orbit camera: the point it looks at [m], its range [m] and angles [deg]."""

    center: tuple[float, float, float]
    distance: float
    elevation_deg: float
    azimuth_deg: float


def _xyz(point: np.ndarray) -> tuple[float, float, float]:
    return (float(point[0]), float(point[1]), float(point[2]))


def camera_position(pose: CameraPose) -> np.ndarray:
    """The eye of ``pose``, as pyqtgraph's orbit camera places it."""
    elev, azim = math.radians(pose.elevation_deg), math.radians(pose.azimuth_deg)
    offset = np.array(
        [
            math.cos(elev) * math.cos(azim),
            math.cos(elev) * math.sin(azim),
            math.sin(elev),
        ]
    )
    return np.asarray(pose.center, dtype=float) + pose.distance * offset


def camera_pose(scene: PuttScene, preset: str) -> CameraPose:
    """The pose of camera ``preset`` (one of :data:`CAMERA_PRESETS`) on ``scene``.

    Raises:
        ValueError: On an unknown preset.
    """
    if preset not in CAMERA_PRESETS:
        raise ValueError(
            f"camera preset must be one of {CAMERA_PRESETS}; got {preset!r}"
        )
    width, height = scene.green_size
    middle = (width / 2.0, height / 2.0, 0.0)
    start = np.asarray(scene.start_xyz, dtype=float)
    cup = np.asarray(scene.cup_xyz, dtype=float)
    ux, uy = (cup[:2] - start[:2]) / float(np.linalg.norm(cup[:2] - start[:2]))
    if preset == "down_the_line":
        length = float(np.linalg.norm(cup[:2] - start[:2]))
        center = (start + cup) / 2.0
        return CameraPose(
            _xyz(center),
            length / 2.0 + 2.5,
            12.0,
            math.degrees(math.atan2(-uy, -ux)),
        )
    if preset == "face_on":
        return CameraPose(_xyz(start), 3.0, 8.0, math.degrees(math.atan2(-ux, uy)))
    if preset == "overhead":
        return CameraPose(middle, max(width, height) * 1.4 + 2.0, 89.0, -90.0)
    return CameraPose(middle, max(width, height) * 1.5 + 4.0, 32.0, -78.0)
