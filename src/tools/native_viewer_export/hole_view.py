"""Opt-in overhead hole view of a native export flight (#15150, track A4).

A broadcast-style plan of the hole: fairway, green, the shot trace, the landing
point, the roll to the rest point and a carry, side, roll and total label, as a still next to the clips so shots can be compared
without a GUI. The hole and figure code is ``golf_environment`` (``shots`` and
``hole_figure``); this module only feeds it the export's own flight, whose carry
and side are the flight model's. Off unless ``TracerSettings.hole_view`` is set
(``--hole-view``). Options: ``hole_view_settings.json`` (see ``sources``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.tools.native_viewer_export import aim_line
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
    label_at_landing: bool = True

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


def _aimed_local(path_m: Any, flight: Any) -> tuple[Any, float]:
    """``path_m`` in ``flight``'s target frame, turned onto its aim line; the yaw."""
    local = flight_coordinates(path_m, flight.axes_m)
    yaw = getattr(flight, "aim_yaw_rad", None) or 0.0
    if yaw:
        local = aim_line.rotate_plan(local, yaw)  # finish aim: tee-to-finish on line
    return local, yaw


def sample_paths(flight: Any, hole: Any) -> list[Any]:
    """Hole-frame paths of ``flight``'s dispersion samples (``[]`` without any).

    Each sample takes the single shot's projection: ``flight``'s target axes
    and aim line, then ``shots_on_hole``, so every path starts on the tee of
    ``hole``. Raises ``ValueError`` when ``flight`` has samples but no axes.
    """
    from src.tools.golf_environment.shots import TargetFlight, shots_on_hole

    overlay = getattr(flight, "dispersion", None)
    if overlay is None:
        return []
    if flight.axes_m is None:
        raise ValueError("dispersion samples need the nominal flight's axes")
    flights = []
    for sample in overlay.samples:
        local, _ = _aimed_local(sample.path_m, flight)
        flights.append(TargetFlight(local, float(local[-1, 0]), -float(local[-1, 1])))
    colours = [overlay.settings.hole_sample_style().colour] * len(flights)
    placed = shots_on_hole(flights, hole, [""] * len(flights), colours)
    return [shot.path_m for shot in placed]


def hole_shot(flight: Any, cfg: HoleViewSettings) -> HoleShot:
    """The flight as a ``HoleShot`` on the tee of ``cfg``'s hole.

    Preconditions: ``flight`` has a path, axes, carry and side. Postcondition:
    the shot starts on the tee and keeps the model's carry and side; the roll
    of ``flight.rollout`` (the export's roll model), if any, is added. Raises
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
    local, yaw = _aimed_local(flight.path_m, flight)
    side_m = float(flight.aim_lateral_m if yaw else flight.lateral_m)
    target = TargetFlight(local, float(flight.carry_m), side_m)
    hole = hole_from_settings(load_shots_settings(), cfg.preset)
    shot = shots_on_hole([target], hole, [cfg.label], [cfg.colour_rgba])[0]
    roll = getattr(flight, "rollout", None)
    if roll is None:
        return shot
    offset = (
        aim_line.rotate_offset(roll.end_offset_m, yaw) if yaw else roll.end_offset_m
    )
    return replace(
        shot,
        roll_m=float(roll.roll_m),
        total_m=float(roll.total_m),
        roll_offset_m=(float(offset[0]), float(offset[1])),
    )


def aim_note(flight: Any) -> str | None:
    """The aim label of ``flight`` (``aim_line_settings.json`` text), or ``None``."""
    return getattr(flight, "aim_text", None) or None


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
    from matplotlib import pyplot as plt

    png = Path(out_dir) / f"{swing}_{engine}_hole_view.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig, _, _ = build_still(flight)
    fig.savefig(png)
    plt.close(fig)
    return {"hole_view": png}


def build_still(flight: Any) -> tuple[Any, Any, Any]:
    """The hole figure of ``flight`` and its (plan, profile) axes; caller closes it.

    Preconditions: ``flight.settings.hole_view`` is set and the flight is
    drawable (``ValueError``). Postcondition: the dispersion samples of
    ``flight``, if any, are drawn thin beneath the shot.
    """
    cfg = flight.settings.hole_view
    if cfg is None:
        raise ValueError("the flight has no hole_view settings")
    from src.tools.golf_environment.hole_figure import build_hole_figure
    from src.tools.golf_environment.shots import hole_from_settings, load_shots_settings

    settings = load_shots_settings()
    settings["figure"] = {**settings["figure"], **cfg.figure}
    hole = hole_from_settings(settings, cfg.preset)
    units = flight.settings.display_units
    fig, plan, side = build_hole_figure(
        [hole_shot(flight, cfg)],
        hole,
        settings,
        units,
        cfg.label_at_landing,
        aim_note(flight),
    )
    if flight.dispersion is not None:
        paths = sample_paths(flight, hole)
        flight.dispersion.paint_hole((plan, side), paths, hole, units.distance_scale)
    return fig, plan, side
