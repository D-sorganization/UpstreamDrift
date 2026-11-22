"""Opt-in overhead hole view of a native export flight (#15150, track A4).

A broadcast-style plan of the hole: fairway, green, the shot trace, the landing
point and the carry label, as a still next to the clips so shots can be compared
without a GUI. The hole and figure code is ``golf_environment`` (``shots`` and
``hole_figure``); this module only feeds it the export's own flight, whose carry
and side are the flight model's. Off unless ``TracerSettings.hole_view`` is set
(``--hole-view``). Options: ``hole_view_settings.json`` (see ``sources``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tools.native_viewer_export.flight_inset import flight_coordinates

if TYPE_CHECKING:
    from src.tools.golf_environment.shots import HoleShot

SETTINGS_PATH = Path(__file__).with_name("hole_view_settings.json")
_FIGURE_KEYS = ("width_in", "height_in", "dpi")


@dataclass(frozen=True)
class HoleViewSettings:
    """Validated options of the hole view (meanings: the settings JSON)."""

    preset: str
    label: str
    colour_rgba: tuple[float, float, float, float]
    figure: dict[str, float]

    def __post_init__(self) -> None:
        if not self.preset:
            raise ValueError("preset must name a hole preset")
        if not self.label:
            raise ValueError("label must not be empty")
        if len(self.colour_rgba) != 4 or not all(
            0.0 <= c <= 1.0 for c in self.colour_rgba
        ):
            raise ValueError("colour_rgba must be four channels in 0..1")
        for key in _FIGURE_KEYS:
            value = self.figure.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"figure {key} must be a finite number")
            if value <= 0:
                raise ValueError(f"figure {key} must be positive")


def load_hole_view_settings(
    path: Path | None = None, preset: str | None = None
) -> HoleViewSettings:
    """Settings from ``path`` (default: beside this module).

    ``preset`` replaces the file's hole preset and must be a preset of
    ``shots_settings.json``. Raises ``ValueError`` for a bad or unknown entry.
    """
    from src.tools.golf_environment.shots import load_shots_settings

    raw = json.loads((path or SETTINGS_PATH).read_text(encoding="utf-8"))
    raw.pop("sources", None)
    raw["colour_rgba"] = tuple(raw["colour_rgba"])
    if preset is not None:
        raw["preset"] = preset
    known = load_shots_settings()["hole"]["presets"]
    if raw["preset"] not in known:
        raise ValueError(f"preset must be one of {sorted(known)}, not {raw['preset']}")
    return HoleViewSettings(**raw)


def hole_shot(flight: Any, cfg: HoleViewSettings) -> HoleShot:
    """The flight as a ``HoleShot`` on the tee of ``cfg``'s hole.

    Preconditions: ``flight`` has a path, axes, carry and side. Postcondition:
    the shot starts on the tee and keeps the model's carry and side. Raises
    ``ValueError`` for a flight that cannot be drawn.
    """
    from src.tools.golf_environment.shots import (
        TargetFlight,
        hole_from_settings,
        load_shots_settings,
        shots_on_hole,
    )

    if flight.path_m is None or flight.axes_m is None:
        raise ValueError(f"no drawable flight: {flight.status}")
    local = flight_coordinates(flight.path_m, flight.axes_m)
    target = TargetFlight(local, float(flight.carry_m), float(flight.lateral_m))
    hole = hole_from_settings(load_shots_settings(), cfg.preset)
    return shots_on_hole([target], hole, [cfg.label], [cfg.colour_rgba])[0]


def write_still(
    overlays: Any, out_dir: Path, swing: str, engine: str
) -> dict[str, Path]:
    """``{"hole_view": png}`` for the flight of ``overlays``, else ``{}``.

    Nothing is written without a flight, without ``hole_view`` settings or when
    the flight has no drawable path. The file is ``<swing>_<engine>_hole_view.png``.
    """
    flight = None if overlays is None else overlays.flight
    cfg = None if flight is None else flight.settings.hole_view
    if cfg is None or flight is None or flight.path_m is None:
        return {}
    from src.tools.golf_environment.hole_figure import write_hole_figure
    from src.tools.golf_environment.shots import hole_from_settings, load_shots_settings

    settings = load_shots_settings()
    settings["figure"] = {**settings["figure"], **cfg.figure}
    hole = hole_from_settings(settings, cfg.preset)
    png = Path(out_dir) / f"{swing}_{engine}_hole_view.png"
    shot = hole_shot(flight, cfg)
    write_hole_figure([shot], hole, png, settings, flight.settings.display_units)
    return {"hole_view": png}
