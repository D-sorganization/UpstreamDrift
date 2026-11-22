"""Pure data layer of the Export Clips dialog (#15150 G11, no PyQt6).

Turns dialog choices into an export request and validates it with the one
request validator (``request_schema.validate``); defaults come from the
request schema and ``dialog_settings.json``, never literals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
        "result_height",
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
    extra_options: dict[str, Any] = field(default_factory=dict)


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


# Options the dialog edits with its own controls; every other remotely
# settable option gets a generated widget (#15150 G6).
HAND_OPTIONS = frozenset(
    {
        "bundle",
        "swing",
        "out",
        "engines",
        "speeds",
        "impact_window",
        "ball_flight_distance_unit",
        "ball_flight_speed_unit",
    }
)
_SETTABLE = ("allowed", "sandboxed_path")
_WIDGET_BY_KIND = {
    "bool": "checkbox",
    "flag": "checkbox",
    "integer": "int_spin",
    "number": "float_spin",
    "path": "path",
    "list": "line",
    "string": "line",
}


@dataclass(frozen=True)
class OptionSpec:
    """One generated control: option name, group, widget type and default."""

    name: str
    group: str
    widget: str
    help: str
    default: Any
    choices: tuple[Any, ...] = ()
    multi: bool = False


def widget_for(option: dict[str, Any]) -> str:
    """Widget type of a schema option: ``checkbox``, ``combo``, ``int_spin``,
    ``float_spin``, ``path`` or ``line``.

    Raises:
        ValueError: the option kind is not one the schema produces.
    """
    kind = option["kind"]
    if kind not in _WIDGET_BY_KIND:
        raise ValueError(f"no widget for option kind {kind!r}")
    if option["choices"] and kind == "string":
        return "combo"
    return _WIDGET_BY_KIND[kind]


def settable_option_names() -> frozenset[str]:
    """Names of every option a request may set (the parity target)."""
    options = request_schema.describe({})["options"]
    return frozenset(o["name"] for o in options if o["remote"] in _SETTABLE)


def grouped_specs() -> list[tuple[str, list[OptionSpec]]]:
    """``(group, specs)`` in schema order for the options without a hand control."""
    described = request_schema.describe({})
    rows: dict[str, list[OptionSpec]] = {g["name"]: [] for g in described["groups"]}
    for option in described["options"]:
        if option["remote"] not in _SETTABLE or option["name"] in HAND_OPTIONS:
            continue
        rows[option["group"]].append(
            OptionSpec(
                option["name"],
                option["group"],
                widget_for(option),
                option["help"],
                option["default"],
                tuple(option["choices"] or ()),
                option["kind"] == "list",
            )
        )
    return [(name, specs) for name, specs in rows.items() if specs]


def spin_settings() -> dict[str, Any]:
    """Shared spin box range, precision and unset label, from the settings."""
    s = load_dialog_settings()
    return {k: s[k] for k in ("spin_minimum", "spin_maximum", "spin_decimals")}


def collect_options(specs: list[OptionSpec], raw: dict[str, Any]) -> dict[str, Any]:
    """Request options for the generated controls; ``raw`` maps name to value.

    Only options the user changed are returned: a value equal to the schema
    default, an empty text or an unset number or combo (``None``) is left out
    so the exporter keeps its own default. List options split on commas.
    """
    options: dict[str, Any] = {}
    for spec in specs:
        value = raw.get(spec.name)
        if spec.widget == "checkbox":
            if bool(value) != bool(spec.default):
                options[spec.name] = bool(value)
            continue
        if isinstance(value, str):
            value = value.strip()
        if value is None or value == "" or value == spec.default:
            continue
        options[spec.name] = (
            [w.strip() for w in value.split(",") if w.strip()] if spec.multi else value
        )
    return options


def prefill_choices(choices: ClipChoices, bundle: str | Path | None) -> ClipChoices:
    """``choices`` with the bundle and swing name taken from a loaded run.

    The swing name is the bundle file stem; ``None`` leaves ``choices`` as is.
    """
    if bundle is None:
        return choices
    path = Path(bundle)
    return replace(choices, bundle=str(path), swing=path.stem)


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
    options.update(choices.extra_options)
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
