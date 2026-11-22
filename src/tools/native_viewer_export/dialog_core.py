"""Pure data layer of the Export Clips dialog (#15150 G11, no PyQt6).

Turns dialog choices into an export request and validates it with the one
request validator (``request_schema.validate``); defaults come from the
request schema and ``dialog_settings.json``, never literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
import json
from pathlib import Path
from typing import Any

from src.tools.native_viewer_export import request_schema
from src.tools.native_viewer_export.display_units import (
    DISTANCE_UNITS,
    SPEED_UNITS,
    load_display_units,
)

_SETTINGS_PATH = Path(__file__).with_name("dialog_settings.json")


@cache
def load_dialog_settings() -> dict[str, Any]:
    """Dialog settings; raises ``ValueError`` unless every number is positive."""
    settings = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    for key in (
        "poll_interval_ms",
        "impact_window_seconds",
        "dialog_width",
        "dialog_height",
    ):
        if not settings[key] > 0:
            raise ValueError(f"{key} must be positive, got {settings[key]!r}")
    return settings


def impact_window_seconds() -> float:
    """Swing seconds of the impact clip, from the settings."""
    return float(load_dialog_settings()["impact_window_seconds"])


@dataclass(frozen=True)
class ClipChoices:
    """What the dialog collects. Engines and speeds are ordered tuples."""

    bundle: str
    swing: str
    out: str
    engines: tuple[str, ...]
    speeds: tuple[float, ...]
    impact_clip: bool
    distance_unit: str = field(
        default_factory=lambda: load_display_units().distance_unit
    )
    speed_unit: str = field(default_factory=lambda: load_display_units().speed_unit)


def _unit_options(units: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    labels = load_dialog_settings()["unit_labels"]
    return tuple((unit, labels[unit]) for unit in units)


def distance_unit_options() -> tuple[tuple[str, str], ...]:
    """``(value, label)`` rows of the distance unit combo box."""
    return _unit_options(DISTANCE_UNITS)


def speed_unit_options() -> tuple[tuple[str, str], ...]:
    """``(value, label)`` rows of the speed unit combo box."""
    return _unit_options(SPEED_UNITS)


def _schema_default(name: str) -> str:
    options = request_schema.describe({})["options"]
    return next(o["default"] for o in options if o["name"] == name)


def available_engines() -> tuple[str, ...]:
    """Engines the exporter offers (the schema default lists all of them)."""
    return tuple(e for e in _schema_default("engines").split(",") if e)


def available_speeds() -> tuple[float, ...]:
    """Speed variants the exporter writes by default."""
    return tuple(float(s) for s in _schema_default("speeds").split(",") if s)


def default_choices() -> ClipChoices:
    """Every engine and speed, no impact clip, output below the output root."""
    _, out_root = request_schema.allowed_roots()
    subfolder = load_dialog_settings()["default_output_subfolder"]
    return ClipChoices(
        bundle="",
        swing="",
        out=str(out_root / subfolder),
        engines=available_engines(),
        speeds=available_speeds(),
        impact_clip=False,
    )


def build_request(choices: ClipChoices) -> dict[str, Any]:
    """The export request for ``choices`` (not yet validated)."""
    options: dict[str, Any] = {
        "bundle": choices.bundle,
        "swing": choices.swing,
        "out": choices.out,
        "engines": ",".join(choices.engines),
        "speeds": ",".join(f"{s:g}" for s in choices.speeds),
        "ball_flight_distance_unit": choices.distance_unit,
        "ball_flight_speed_unit": choices.speed_unit,
    }
    if choices.impact_clip:
        options["impact_window"] = impact_window_seconds()
    return {"options": options}


def validate_choices(choices: ClipChoices) -> dict[str, Any]:
    """Validate ``choices``; returns ``{"argv", "command_line", ...}``.

    Raises:
        ValueError: nothing chosen, a missing bundle or swing, or anything
            the request validator rejects (paths outside the allowed roots,
            unknown engines, speeds outside the exporter's range).
    """
    for message, value in (
        ("choose at least one engine", choices.engines),
        ("choose at least one speed variant", choices.speeds),
        ("choose a bundle file", choices.bundle),
        ("enter a swing name", choices.swing),
    ):
        if not value:
            raise ValueError(message)
    return request_schema.validate(build_request(choices))


def manifest_lines(manifest: dict[str, Any] | None) -> list[str]:
    """One readable line per clip, or per skipped engine, of a job manifest."""
    lines: list[str] = []
    for row in (manifest or {}).get("engines", []):
        if row.get("skipped_reason"):
            lines.append(f"{row['engine']} skipped: {row['skipped_reason']}")
        lines += [f"{row['engine']} {n}: {p}" for n, p in row["clips"].items()]
    return lines
