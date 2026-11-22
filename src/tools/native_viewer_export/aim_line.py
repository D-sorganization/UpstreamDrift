"""Aim line of the shot-shape views: ball to finish, or the address target line.

With ``aim_line = "finish"`` the plan view, side figure and wide-view ground
line use the line from the ball to where the ball finishes (the rest point
when a roll is modelled), so a draw or fade ending on target reads as on
target. ``"address"`` keeps the address line; ``"auto"`` is ``finish`` for a
requested shot shape, else unchanged. Flight numbers never change (#15443).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

Array = np.ndarray
SETTINGS_PATH = Path(__file__).with_name("aim_line_settings.json")
MODES = ("auto", "finish", "address")


@dataclass(frozen=True)
class AimLineSettings:
    """Validated options of the aim line (meanings: the settings JSON)."""

    aim_line: str = "auto"
    finish_text: str = "Aim: finish line"
    address_text: str = "Aim: address line"
    line_color_rgba: tuple[int, ...] = (255, 255, 255, 200)
    line_width_px: int = 3
    line_width_fraction: float = 0.0055
    outline_color_rgba: tuple[int, ...] = (10, 20, 40, 230)
    outline_extra_px: int = 2
    overshoot_fraction: float = 0.1
    min_finish_distance_m: float = 1.0

    def __post_init__(self) -> None:
        if self.aim_line not in MODES:
            raise ValueError(f"aim_line must be one of {MODES}, got {self.aim_line!r}")
        rgba = tuple(self.line_color_rgba)
        if len(rgba) != 4 or not all(
            isinstance(c, int) and 0 <= c <= 255 for c in rgba
        ):
            raise ValueError("line_color_rgba must be four integers in 0..255")
        object.__setattr__(self, "line_color_rgba", rgba)
        if not isinstance(self.line_width_px, int) or self.line_width_px < 1:
            raise ValueError("line_width_px must be a positive integer")
        for name in ("overshoot_fraction", "min_finish_distance_m"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"{name} must be finite and not negative")

    def resolved(self, shot_shape: str | None, override: str | None) -> str:
        """``finish``, ``address`` or ``auto`` for a request.

        ``override`` (``--aim-line``) wins; ``auto`` becomes ``finish`` when a
        shot shape was requested and stays ``auto`` (unchanged picture) else.
        """
        mode = override or self.aim_line
        if mode not in MODES:
            raise ValueError(f"aim_line must be one of {MODES}, got {mode!r}")
        return "finish" if mode == "auto" and shot_shape else mode

    def width_for(self, tile_height_px: int) -> int:
        """Line width (pixels) on a tile ``tile_height_px`` high, never below the minimum."""
        if tile_height_px < 1:
            raise ValueError("tile_height_px must be positive")
        return max(self.line_width_px, round(tile_height_px * self.line_width_fraction))

    def text_for(self, mode: str) -> str:
        """The label stating which line is drawn."""
        return self.finish_text if mode == "finish" else self.address_text


def load_aim_settings(path: Path | None = None) -> AimLineSettings:
    """Settings from ``path`` (default: the file beside this module)."""
    document = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    document.pop("sources", None)
    return AimLineSettings(**document)


def finish_yaw(
    full: Array, roll_offset: tuple[float, float] | None, minimum_m: float
) -> float:
    """Angle (rad, towards left) from the address line to the finish line.

    ``full`` is the flight in (down-range, left, up) metres from the ball and
    ``roll_offset`` the (down-range, left) roll vector. Zero when the finish
    lies closer than ``minimum_m`` (no usable direction).
    """
    down, left = float(full[-1, 0]), float(full[-1, 1])
    if roll_offset is not None:
        down, left = down + roll_offset[0], left + roll_offset[1]
    return math.atan2(left, down) if math.hypot(down, left) >= minimum_m else 0.0


def rotate_plan(points: Array, yaw: float) -> Array:
    """``points`` (down, left, up) with (down, left) turned onto the aim line."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = np.array(points, dtype=float)
    down, left = out[..., 0].copy(), out[..., 1].copy()
    out[..., 0], out[..., 1] = down * c + left * s, -down * s + left * c
    return out


def rotate_offset(offset: tuple[float, float], yaw: float) -> tuple[float, float]:
    """A (down-range, left) vector expressed along the aim line."""
    moved = rotate_plan(np.array([offset[0], offset[1], 0.0]), yaw)
    return (float(moved[0]), float(moved[1]))


def ground_line(
    start_m: Array, axes_m: Array, yaw: float, length_m: float, ground_z_m: float
) -> Array:
    """Ground end points of the aim line from ``start_m`` (``axes_m``: target frame)."""
    axes = np.asarray(axes_m, dtype=float)
    direction = math.cos(yaw) * axes[0] + math.sin(yaw) * axes[1]
    direction = direction / float(np.linalg.norm(direction[:2]))
    start = np.array([start_m[0], start_m[1], ground_z_m], dtype=float)
    return np.vstack([start, start + direction * length_m])


def side_text_args(lateral_m: float) -> tuple[float, str]:
    """``(metres, "left"|"right"|"")`` of a right-positive offset; noise is ``""``."""
    value = 0.0 if abs(lateral_m) < 1e-9 else lateral_m
    return abs(value), "" if value == 0.0 else ("left" if value < 0.0 else "right")


def aimed_fields(flight: Any, mode: str, settings: AimLineSettings) -> dict[str, Any]:
    """Fields of a ``FlightPlayback`` that carry the aim line (for ``replace``).

    ``flight`` is a playback with a path, axes and (maybe) a roll. For
    ``address`` the yaw is zero; ``lateral_aim_m`` is the right-positive side of
    the landing point against the aim line. Raises ``ValueError`` without a
    path or axes.
    """
    if flight.path_m is None or flight.axes_m is None:
        raise ValueError("an aim line needs a flight path and its target frame")
    if mode not in ("finish", "address"):
        raise ValueError(f"mode must be finish or address, got {mode!r}")
    path, axes = np.asarray(flight.path_m), np.asarray(flight.axes_m)
    full = (path - path[0]) @ axes.T
    roll = None if flight.rollout is None else flight.rollout.end_offset_m
    yaw = 0.0
    if mode == "finish":
        yaw = finish_yaw(full, roll, settings.min_finish_distance_m)
    landing = rotate_plan(full[-1], yaw)
    return {
        "aim_yaw_rad": yaw,
        "aim_text": settings.text_for(mode),
        "aim_lateral_m": -float(landing[1]),
    }


def apply_roll_yaw(roll: Any, yaw: float) -> Any:
    """``roll`` with its end offset along the aim line."""
    if roll is None or yaw == 0.0:
        return roll
    return replace(roll, end_offset_m=rotate_offset(roll.end_offset_m, yaw))
