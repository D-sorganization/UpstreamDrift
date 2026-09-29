"""Colour helpers for golf simulation rendering.

Pure-numpy gradient sampling and golf-domain palettes (turf elevation,
ball speed, putt roll mode).  No Qt/OpenGL dependency, so these run in any
headless environment and are exhaustively unit-tested.

Design by Contract:
    * Public functions validate argument shapes and raise ``ValueError`` /
      ``TypeError`` with descriptive messages.
    * All returned arrays are ``(N, 4)`` RGBA in the closed range
      ``[0, 1]`` and contain only finite values.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

RGBA = tuple[float, float, float, float]

# Roll-mode palette using golf-broadcast semantics: amber while the ball
# skids (energy bleeding off), green once it is purely rolling, grey at rest.
ROLL_MODE_RGBA: dict[str, RGBA] = {
    "sliding": (0.95, 0.55, 0.15, 1.0),
    "rolling": (0.30, 0.85, 0.35, 1.0),
    "stopped": (0.62, 0.64, 0.70, 1.0),
}

# Lush fairway green at low elevation tanning toward dry mound at the top.
_TERRAIN_STOPS: tuple[RGBA, ...] = (
    (0.16, 0.52, 0.22, 1.0),
    (0.34, 0.60, 0.26, 1.0),
    (0.62, 0.56, 0.34, 1.0),
)

# Cool (slow) to hot (fast) clubhead/ball-speed ramp with monotonic red.
_SPEED_STOPS: tuple[RGBA, ...] = (
    (0.20, 0.45, 0.85, 1.0),
    (0.30, 0.80, 0.45, 1.0),
    (0.85, 0.75, 0.25, 1.0),
    (0.95, 0.25, 0.20, 1.0),
)

__all__ = [
    "RGBA",
    "ROLL_MODE_RGBA",
    "roll_mode_colors",
    "sample_gradient",
    "speed_colors",
    "terrain_colors",
]


def sample_gradient(
    values: Iterable[float],
    stops: Sequence[RGBA],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Map scalar ``values`` onto an evenly-spaced RGBA ``stops`` ramp.

    Args:
        values: Scalars to colour. Non-finite entries are treated as ``vmin``.
        stops: Two or more RGBA 4-tuples, evenly spaced across the value range.
        vmin: Lower bound of the value range (defaults to ``min(values)``).
        vmax: Upper bound of the value range (defaults to ``max(values)``).

    Returns:
        ``(N, 4)`` float array of RGBA colours in ``[0, 1]``.

    Raises:
        ValueError: If fewer than two stops are given, or a stop is not RGBA.
    """
    try:
        stops_arr = np.asarray(stops, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("colour stops must be a sequence of RGBA tuples") from exc
    if stops_arr.ndim != 2 or stops_arr.shape[0] < 2:
        raise ValueError("gradient needs at least two colour stops")
    if stops_arr.shape[1] != 4:
        raise ValueError("each colour stop must be an RGBA 4-tuple")

    vals = np.asarray(list(values), dtype=float).reshape(-1)
    if vals.size == 0:
        return np.empty((0, 4), dtype=float)

    finite = np.isfinite(vals)
    if vmin is None:
        vmin = float(np.min(vals[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.max(vals[finite])) if finite.any() else 0.0

    span = float(vmax) - float(vmin)
    if span <= 0.0:
        t = np.zeros(vals.size, dtype=float)
    else:
        clean = np.where(finite, vals, vmin)
        t = (clean - vmin) / span
    t = np.clip(t, 0.0, 1.0)

    positions = np.linspace(0.0, 1.0, stops_arr.shape[0])
    out = np.empty((vals.size, 4), dtype=float)
    for channel in range(4):
        out[:, channel] = np.interp(t, positions, stops_arr[:, channel])
    return np.clip(out, 0.0, 1.0)


def terrain_colors(
    elevations: Iterable[float],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Colour turf elevations from lush green (low) to dry tan (high)."""
    return sample_gradient(elevations, _TERRAIN_STOPS, vmin=vmin, vmax=vmax)


def speed_colors(
    values: Iterable[float],
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """Colour speed magnitudes from cool (slow) to hot (fast)."""
    return sample_gradient(values, _SPEED_STOPS, vmin=vmin, vmax=vmax)


def roll_mode_colors(modes: Iterable[object]) -> np.ndarray:
    """Map putt roll-mode labels to the :data:`ROLL_MODE_RGBA` palette.

    Accepts plain strings (``"sliding"``), ``RollMode`` enum members (whose
    ``str()`` is ``"RollMode.SLIDING"``), or any object whose string form ends
    in a known mode name. Matching is case-insensitive; unknown labels fall
    back to the ``"stopped"`` colour.
    """
    mode_list = list(modes)
    out = np.empty((len(mode_list), 4), dtype=float)
    fallback = ROLL_MODE_RGBA["stopped"]
    for i, mode in enumerate(mode_list):
        name = str(mode).rsplit(".", maxsplit=1)[-1].strip().lower()
        out[i] = ROLL_MODE_RGBA.get(name, fallback)
    return out
