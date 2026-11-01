"""Putting-study tools for the simulation MCP adapter (#14177).

Two thin wrappers over :mod:`src.shared.python.putting_analytics.service`:

* ``optimize_putt`` -- best aim and pace for one putt, by strokes gained
  against the caller's expected-putts curve or by make rate (``must_make``).
* ``putting_misread_cost`` -- expected-strokes cost of a set of green reads
  (slope and speed misjudgements) against a perfect read.

The service is imported lazily so listing tools never loads the roll
integrator. Specs are hand-written: ``_tool_spec`` cannot express arrays.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

ToolHandler = Callable[[dict[str, Any]], dict[str, Any]]

_NUMBER = {"type": "number"}
_NUMBERS = {"type": "array", "items": _NUMBER, "minItems": 1}

_STUDY_PROPERTIES: dict[str, Any] = {
    "auth_token": {"type": "string"},
    "distance_ft": {**_NUMBER, "description": "Putt length, feet (0, 100]."},
    "grade_pct": {**_NUMBER, "description": "Green slope, percent [0, 10]."},
    "aspect_deg": {**_NUMBER, "description": "Fall line, deg CCW of the hole line."},
    "stimp": {**_NUMBER, "description": "Green speed, feet [5, 16]."},
    "line_sd_deg": {**_NUMBER, "description": "Start-line error SD, degrees."},
    "speed_sd": {**_NUMBER, "description": "Log speed-ratio error SD."},
    "baseline": {
        "type": "object",
        "description": "The player's expected putts by distance; prices a miss.",
        "properties": {"distances_ft": _NUMBERS, "expected_putts": _NUMBERS},
        "required": ["distances_ft", "expected_putts"],
    },
    "aims_deg": {**_NUMBERS, "description": "Candidate aims, degrees left."},
    "paces_ft": {**_NUMBERS, "description": "Candidate paces, feet past the hole."},
    "objective": {"type": "string", "enum": ["sg", "must_make"]},
    "n": {"type": "integer", "description": "Putts per candidate."},
    "seed": {"type": "integer"},
}

_READ = {
    "type": "object",
    "properties": {"slope_scale": _NUMBER, "stimp_ratio": _NUMBER},
}


def _spec(name: str, description: str, extra: dict[str, Any]) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {**_STUDY_PROPERTIES, **extra},
        "required": ["distance_ft", "baseline", *extra],
    }
    return {"name": name, "description": description, "inputSchema": schema}


def putting_tool_specs() -> list[dict[str, Any]]:
    """MCP ``tools/list`` entries for the putting tools."""
    reads = {"reads": {"type": "array", "items": _READ, "minItems": 1}}
    return [
        _spec(
            "optimize_putt",
            "Best aim and pace for one putt under the player's stroke noise",
            {},
        ),
        _spec(
            "putting_misread_cost",
            "Expected-strokes cost of green misreads versus a perfect read",
            reads,
        ),
    ]


def _service_tool(function_name: str) -> ToolHandler:
    def run(arguments: dict[str, Any]) -> dict[str, Any]:
        from src.shared.python.putting_analytics import service

        payload = getattr(service, function_name)(arguments)
        return {"content": [{"type": "text", "text": json.dumps(payload)}]}

    return run


def putting_tool_handlers(check_auth: Callable[[Any], None]) -> dict[str, ToolHandler]:
    """``tools/call`` handlers; each checks the auth token before any work."""

    def guarded(run: ToolHandler) -> ToolHandler:
        def handler(arguments: dict[str, Any]) -> dict[str, Any]:
            check_auth(arguments.get("auth_token"))
            return run(arguments)

        return handler

    return {
        "optimize_putt": guarded(_service_tool("optimize_putt")),
        "putting_misread_cost": guarded(_service_tool("misread_cost")),
    }
