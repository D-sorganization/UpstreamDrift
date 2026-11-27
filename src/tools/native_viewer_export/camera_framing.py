"""Camera look-at, look-at height and distance as export settings (#15150).

The automatic look-at is the golfer's centre of mass at a height read from
``camera_framing_settings.json``. ``--camera-lookat`` replaces the whole point,
``--camera-height`` replaces only its height and ``--camera-distance`` replaces
the view preset's distance.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path(__file__).with_name("camera_framing_settings.json")


def default_lookat_height_m() -> float:
    """Automatic look-at height (m) from the settings file; finite or ValueError."""
    raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    return check_finite("default_lookat_height_m", raw["default_lookat_height_m"])


def check_finite(name: str, value: Any) -> float:
    """``value`` as a float; raises ``ValueError`` if it is not finite."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"camera {name} must be a number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"camera {name} must be finite, got {value!r}")
    return number


def parse_lookat(text: str) -> tuple[float, float, float]:
    """Parse ``X,Y,Z`` metres; raises ``ValueError`` unless three finite numbers."""
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(f"camera lookat must be X,Y,Z in metres, got {text!r}")
    x, y, z = (check_finite("lookat", p) for p in parts)
    return (x, y, z)


def check_distance(value: float) -> float:
    """Camera distance (m); raises ``ValueError`` unless positive and finite."""
    distance = check_finite("distance", value)
    if distance <= 0.0:
        raise ValueError(f"camera distance must be positive, got {value!r}")
    return distance


def resolve_lookat(
    auto: tuple[float, float, float], settings: Any
) -> tuple[float, float, float]:
    """The look-at the views use.

    Precondition: ``settings`` has ``lookat_explicit``, ``lookat_m`` and
    ``lookat_height_m``. Postcondition: an explicit look-at wins over the
    automatic one; else a set height replaces only ``auto``'s z; else ``auto``.
    """
    if settings.lookat_explicit:
        return tuple(settings.lookat_m)  # type: ignore[return-value]
    if settings.lookat_height_m is not None:
        return (auto[0], auto[1], settings.lookat_height_m)
    return auto
