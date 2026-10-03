"""Helper utilities for Simscape 3D viewer tab (ADR-0052, #11305)."""

from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...core.models import C3DDataModel
from .overview_tab import _scalar_value

_CLUB_MARKER_RE = re.compile(r"^Marker_\d+:\d+:", re.IGNORECASE)


def is_club_marker(name: str) -> bool:
    """Return True when ``name`` matches the generic club/cluster marker pattern."""
    return bool(_CLUB_MARKER_RE.match(name))


def validate_speed(speed: float) -> float:
    """Validate a playback speed multiplier."""
    if isinstance(speed, bool) or not isinstance(speed, (int, float)):
        raise TypeError(f"speed must be a real number, got {type(speed).__name__}")
    if not np.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"speed must be positive and finite, got {speed!r}")
    return float(speed)


def validate_frame(frame: int, n_frames: int) -> int:
    """Validate a frame index against ``n_frames``."""
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise TypeError(f"frame must be int, got {type(frame).__name__}")
    if n_frames <= 0:
        raise ValueError("no frames loaded")
    if not 0 <= frame < n_frames:
        raise ValueError(f"frame {frame} out of range [0, {n_frames})")
    return frame


def transform_positions_by_screen_axes(
    pos: np.ndarray, raw_parameters: Any | None
) -> np.ndarray:
    """Apply axis convention transformation based on X_SCREEN and Y_SCREEN."""
    if pos.size == 0 or raw_parameters is None:
        return pos
    raw = raw_parameters if isinstance(raw_parameters, dict) else {}
    point = raw.get("POINT", {}) if isinstance(raw, dict) else {}

    x_screen = point.get("X_SCREEN") if isinstance(point, dict) else None
    y_screen = point.get("Y_SCREEN") if isinstance(point, dict) else None

    x_val = _scalar_value(x_screen) if x_screen is not None else None
    y_val = _scalar_value(y_screen) if y_screen is not None else None

    # If no convention parameters, defaults are +X and +Z
    x_str = str(x_val).upper().strip() if x_val else "+X"
    y_str = str(y_val).upper().strip() if y_val else "+Z"

    # Validate format (must be like +X, -Y, etc.)
    def parse_axis(s: str, default_axis: int, default_sign: float) -> tuple[int, float]:
        if len(s) >= 2 and s[0] in ("+", "-") and s[1] in ("X", "Y", "Z"):
            axis = {"X": 0, "Y": 1, "Z": 2}[s[1]]
            sign = 1.0 if s[0] == "+" else -1.0
            return axis, sign
        if len(s) == 1 and s in ("X", "Y", "Z"):
            return {"X": 0, "Y": 1, "Z": 2}[s], 1.0
        return default_axis, default_sign

    x_axis, x_sign = parse_axis(x_str, 0, 1.0)
    z_axis, z_sign = parse_axis(y_str, 2, 1.0)

    # If axes are same (invalid convention), fallback to identity
    if x_axis == z_axis:
        return pos

    y_axis = 3 - x_axis - z_axis

    def levi_civita(i: int, j: int, k: int) -> float:
        if {i, j, k} != {0, 1, 2}:
            return 0.0
        if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            return 1.0
        return -1.0

    y_sign = x_sign * z_sign * levi_civita(x_axis, y_axis, z_axis)

    transformed = np.empty_like(pos)
    transformed[..., 0] = pos[..., x_axis] * x_sign
    transformed[..., 1] = pos[..., y_axis] * y_sign
    transformed[..., 2] = pos[..., z_axis] * z_sign
    return transformed


def finalize_scene_view(
    ax: Any, positions: np.ndarray, selected: Sequence[str]
) -> None:
    """Set 3D axis labels, equal aspect box, and legend."""
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Marker Trajectories")

    finite = positions[np.isfinite(positions).all(axis=2)]
    if finite.size > 0:
        mn = finite.min(axis=0)
        mx = finite.max(axis=0)
        max_range = float(np.max(mx - mn))
        if max_range > 0.0:
            mid = 0.5 * (mx + mn)
            half = max_range / 2.0
            ax.set_xlim(mid[0] - half, mid[0] + half)
            ax.set_ylim(mid[1] - half, mid[1] + half)
            ax.set_zlim(mid[2] - half, mid[2] + half)

    if len(selected) <= 12:
        ax.legend(loc="upper right", fontsize=8)


def export_markers_to_csv(
    path: str | Path,
    model: C3DDataModel | None,
    selected_names: Sequence[str],
    n_frames: int,
) -> None:
    """Write selected marker trajectories to CSV."""
    if not isinstance(path, (str, Path)) or not str(path):
        raise ValueError("path must be a non-empty string or Path")
    if model is None:
        raise ValueError("no model loaded")
    names = list(selected_names) or model.marker_names()
    if not names:
        raise ValueError("no markers selected to export")

    header = ["frame", "time_s"]
    for name in names:
        header += [f"{name}_x", f"{name}_y", f"{name}_z"]

    time_arr = (
        model.point_time if model.point_time is not None else np.full(n_frames, np.nan)
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fr in range(n_frames):
            row: list[Any] = [fr, float(time_arr[fr]) if fr < len(time_arr) else ""]
            for name in names:
                m = model.markers.get(name)
                if m is None or m.position.size == 0 or fr >= m.position.shape[0]:
                    row += ["", "", ""]
                else:
                    x, y, z = m.position[fr]
                    row += [float(x), float(y), float(z)]
            writer.writerow(row)
