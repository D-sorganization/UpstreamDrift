"""Place model ball flights on a golf hole (epic #15150, tracks A2 and A3).

Pure data, no Qt: flights in target-frame axes (down-range, left, up), as
:class:`~src.shared.python.impact_parameters.flight_report.FlightSummary`
reports them, are mapped to hole coordinates. The hole is any object with
``tee_position`` and ``pin_position`` (a ``CourseHole``): the ball starts on
the tee and the target line runs from the tee to the pin, so down-range is the
horizontal tee-to-pin direction, left is that direction turned 90 degrees
counter-clockwise seen from above, and up is +z. ``side_m`` is right-positive.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SETTINGS_PATH = Path(__file__).with_name("shots_settings.json")
Colour = tuple[float, float, float, float]


@dataclass(frozen=True)
class TargetFlight:
    """A flight path (n, 3) in target-frame axes plus its carry and side (right +)."""

    trajectory_m: Any
    carry_m: float
    lateral_m: float


@dataclass(frozen=True)
class HoleShot:
    """One shot in hole coordinates: ``path_m`` is (n, 3) starting on the tee."""

    label: str
    colour: Colour
    path_m: Any
    carry_m: float
    side_m: float

    def __post_init__(self) -> None:
        path = np.asarray(self.path_m, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
            raise ValueError(f"path must be (n>=2, 3), got {path.shape}")
        if not np.isfinite(path).all():
            raise ValueError("path must be finite")
        if not (math.isfinite(self.carry_m) and math.isfinite(self.side_m)):
            raise ValueError("carry and side must be finite")
        if len(self.colour) != 4:
            raise ValueError("colour must be an RGBA tuple")


def load_shots_settings(path: Path | None = None) -> dict[str, Any]:
    """Read ``shots_settings.json`` (``KeyError`` if a section is missing)."""
    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    missing = {"sources", "hole", "shot_colours_rgba", "decimals", "figure"} - set(raw)
    if missing:
        raise KeyError(f"shots settings lack {sorted(missing)}")
    return dict(raw)


def hole_from_settings(settings: dict[str, Any]) -> Any:
    """A ``CourseHole`` whose pin lies ``length_m`` down the +x axis from the tee.

    Imports the Qt module lazily so the rest of this module stays Qt-free.
    """
    from src.tools.golf_environment.gui import CourseHole

    spec = settings["hole"]
    length = float(spec["length_m"])
    return CourseHole(
        str(spec["name"]),
        int(spec["par"]),
        round(length / float(spec["metres_per_yard"])),
        pin_position=(length, 0.0, 0.0),
        fairway_width=float(spec["fairway_width_m"]),
        green_radius=float(spec["green_radius_m"]),
    )


def hole_axes(hole: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(tee, forward, left)``: tee point and unit horizontal tee-to-pin axes.

    Raises ``ValueError`` when the pin is straight above or on the tee.
    """
    tee = np.asarray(hole.tee_position, dtype=float)
    delta = np.asarray(hole.pin_position, dtype=float) - tee
    length = float(np.hypot(delta[0], delta[1]))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("pin must be a finite horizontal distance from the tee")
    forward = np.array([delta[0] / length, delta[1] / length, 0.0])
    return tee, forward, np.array([-forward[1], forward[0], 0.0])


def shots_on_hole(
    flights: Sequence[Any],
    hole: Any,
    labels: Sequence[str],
    colours: Sequence[Colour],
) -> list[HoleShot]:
    """Map target-frame ``flights`` to hole coordinates, one ``HoleShot`` each.

    Preconditions: one label and colour per flight; each ``trajectory_m`` is
    finite (n >= 2, 3). Postcondition: output length equals input length, each
    path starts at the tee and a +lateral flight ends right of the tee-to-pin
    line. Raises ``ValueError`` otherwise.
    """
    if not (len(flights) == len(labels) == len(colours)):
        raise ValueError("flights, labels and colours must have the same length")
    tee, forward, left = hole_axes(hole)
    up = np.array([0.0, 0.0, 1.0])
    axes = np.vstack([forward, left, up])
    out = []
    for flight, label, colour in zip(flights, labels, colours, strict=True):
        local = np.asarray(flight.trajectory_m, dtype=float)
        if local.ndim != 2 or local.shape[1] != 3:
            raise ValueError(f"trajectory must be (n, 3), got {local.shape}")
        path = tee + (local - local[0]) @ axes
        out.append(
            HoleShot(
                str(label),
                tuple(colour),
                path,
                float(flight.carry_m),
                float(flight.lateral_m),
            )
        )
    return out


def add_shots_to_renderer(renderer: Any, shots: Sequence[HoleShot]) -> None:
    """Draw each shot with ``renderer.add_trajectory`` (an ``EnvironmentRenderer``)."""
    for shot in shots:
        renderer.add_trajectory(np.asarray(shot.path_m, dtype=float), shot.colour)
