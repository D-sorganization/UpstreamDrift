"""Swing-event table and CSV/JSON export of anatomical angles (HRG-E4, #13175).

The table samples the wrist and forearm series of :mod:`anatomical_angles` at
caller-supplied swing events (address, top, impact, finish). It does not detect
events: the frames come from an existing detector (for example
``swing_comparison.events.detect_events``) and the caller names it in
``event_source``, which every export carries.

Terms, frames and sign conventions come from the one golf-term description
(:func:`anatomical_angles.terms.wrist_term_specs`); this module adds no term.
Lead and trail sides follow ``core.handedness``. Every value carries its
observability, receipt path and calibration ids; a value that is NaN in the
series stays missing (JSON ``null``, empty CSV cell) with a reason. Pelvis
reversal and trunk terms are not on ``main`` yet and are not accepted here.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.anatomical_angles.terms import (
    WristTermSpec,
    wrist_term_reference,
    wrist_term_specs,
)
from src.shared.python.anatomical_angles.wrist import WristForearm, check_side
from src.shared.python.core.handedness import (
    lead_side,
    trail_side,
    validate_handedness,
)

SCHEMA_ID = "upstreamdrift.anatomical_event_table"
SCHEMA_VERSION = 1
#: Swing events in their required temporal order.
EVENT_ORDER: tuple[str, ...] = ("address", "top", "impact", "finish")
UNITS: dict[str, str] = {"angle": "deg", "time": "s"}
CSV_COLUMNS: tuple[str, ...] = (
    "event",
    "frame",
    "time_s",
    "term_id",
    "role",
    "side",
    "value_deg",
    "invalid_reason",
    "observability",
    "receipt_path",
    "calibration",
)
_NAN_REASON = "value is NaN in the source series"
#: (profile, component) -> (series, angle attribute, reason key).
_SERIES: dict[tuple[str, int], tuple[str, str, str]] = {
    ("isb_wrist", 0): ("wrist", "flexion_rad", "flexion"),
    ("isb_wrist", 1): ("wrist", "ulnar_deviation_rad", "ulnar_deviation"),
    ("isb_wrist", 2): ("wrist", "axial_rotation_rad", "axial_rotation"),
    ("isb_elbow", 2): ("forearm", "pronation_rad", "pronation"),
}


@dataclass(frozen=True)
class EventRow:
    """One anatomical term sampled at one swing event."""

    event: str
    frame: int
    time_s: float
    term_id: str
    role: str
    side: str
    value_deg: float  # NaN when the source sample is invalid
    invalid_reason: str  # empty for a valid value
    observability: str
    receipt_path: str
    calibration: tuple[str, str]


@dataclass(frozen=True)
class AnatomicalEventTable:
    """Rows in event order, then registry term order, with export metadata."""

    rows: tuple[EventRow, ...]
    events: dict[str, tuple[int, float]]
    handedness: str
    event_source: str
    terms: tuple[WristTermSpec, ...]


def _check_events(events: Mapping[str, Any], n: int) -> dict[str, int]:
    if not events:
        raise ValueError("at least one swing event is required")
    unknown = sorted(set(events) - set(EVENT_ORDER))
    if unknown:
        raise ValueError(f"unknown event(s) {unknown}; choose from {list(EVENT_ORDER)}")
    frames: dict[str, int] = {}
    for name in EVENT_ORDER:
        if name not in events:
            continue
        idx = events[name]
        if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
            raise ValueError(f"event {name!r} frame must be an integer, got {idx!r}")
        if not 0 <= int(idx) < n:
            raise ValueError(f"event {name!r} frame {idx} out of range [0, {n})")
        frames[name] = int(idx)
    ordered = list(frames.values())
    if ordered != sorted(ordered):
        raise ValueError(f"event frames must follow the order {EVENT_ORDER}: {frames}")
    return frames


def _check_arms(arms: Mapping[str, WristForearm], n: int) -> None:
    if not arms:
        raise ValueError("at least one arm (side 'L' or 'R') is required")
    for side, arm in arms.items():
        check_side(side)
        if arm.wrist.side != side or arm.forearm.side != side:
            raise ValueError(f"arm keyed {side!r} does not match its series side")
        for attr, name, _ in _SERIES.values():
            if len(getattr(getattr(arm, attr), name)) != n:
                raise ValueError(f"{side} {attr} series length differs from time_s")


def _sample(
    arm: WristForearm, spec: WristTermSpec, frame: int
) -> tuple[float, str, Any]:
    attr, name, reason_key = _SERIES[(spec.profile, spec.component)]
    series = getattr(arm, attr)
    value = float(np.rad2deg(getattr(series, name)[frame]))
    reason = str(series.reasons[reason_key][frame]) if math.isnan(value) else ""
    if math.isnan(value) and not reason:
        reason = _NAN_REASON
    return value, reason, series


def anatomical_event_table(
    arms: Mapping[str, WristForearm],
    events: Mapping[str, Any],
    time_s: ArrayLike,
    *,
    handedness: str,
    event_source: str,
) -> AnatomicalEventTable:
    """Wrist and forearm terms of ``arms`` sampled at ``events``.

    ``arms`` maps ``"L"``/``"R"`` to the series of that side; a missing side
    yields no rows for its role (nothing is filled in). ``events`` maps names
    from :data:`EVENT_ORDER` to frame indices that are non-decreasing in that
    order. ``time_s`` gives one time per frame. ``event_source`` names the
    detector or annotation that produced the frames. Raises ``ValueError`` on
    any violated precondition.
    """
    handedness = validate_handedness(handedness)
    if not isinstance(event_source, str) or not event_source.strip():
        raise ValueError("event_source must name the event detector or annotation")
    times = np.asarray(time_s, dtype=float).reshape(-1)
    if not np.all(np.isfinite(times)):
        raise ValueError("time_s must be finite")
    _check_arms(arms, times.size)
    frames = _check_events(events, times.size)
    sides = {"lead": lead_side(handedness), "trail": trail_side(handedness)}
    specs = wrist_term_specs()
    rows: list[EventRow] = []
    for event, frame in frames.items():
        for spec in specs:
            side = sides[spec.role]
            if side not in arms:
                continue
            value, reason, series = _sample(arms[side], spec, frame)
            rows.append(
                EventRow(
                    event,
                    frame,
                    float(times[frame]),
                    spec.term_id,
                    spec.role,
                    side,
                    value,
                    reason,
                    series.observability,
                    series.receipt_path,
                    tuple(series.calibration),
                )
            )
    return AnatomicalEventTable(
        tuple(rows),
        {e: (f, float(times[f])) for e, f in frames.items()},
        handedness,
        event_source,
        specs,
    )


def _term_meta(spec: WristTermSpec) -> dict[str, str]:
    return {
        "unit": UNITS["angle"],
        "role": spec.role,
        "profile": spec.profile,
        "frame": spec.frame,
        "sign_convention": spec.sign_convention,
        "description": spec.description,
    }


def to_json_dict(table: AnatomicalEventTable) -> dict[str, Any]:
    """Strict-JSON document (no NaN) matching :data:`EVENT_TABLE_JSON_SCHEMA`."""
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "units": dict(UNITS),
        "handedness": table.handedness,
        "event_source": table.event_source,
        "reference": wrist_term_reference(),
        "events": {e: {"frame": f, "time_s": t} for e, (f, t) in table.events.items()},
        "terms": {s.term_id: _term_meta(s) for s in table.terms},
        "rows": [
            {
                "event": r.event,
                "frame": r.frame,
                "time_s": r.time_s,
                "term_id": r.term_id,
                "role": r.role,
                "side": r.side,
                "value_deg": None if math.isnan(r.value_deg) else r.value_deg,
                "invalid_reason": r.invalid_reason or None,
                "observability": r.observability,
                "receipt_path": r.receipt_path,
                "calibration": list(r.calibration),
            }
            for r in table.rows
        ],
    }


def to_json(table: AnatomicalEventTable) -> str:
    """:func:`to_json_dict` as indented strict JSON text."""
    return json.dumps(to_json_dict(table), indent=2, allow_nan=False)


def _csv_header(table: AnatomicalEventTable) -> Sequence[str]:
    lines = [
        f"# schema: {SCHEMA_ID} v{SCHEMA_VERSION}",
        f"# units: angle={UNITS['angle']} time={UNITS['time']}",
        f"# handedness: {table.handedness}",
        f"# event_source: {table.event_source}",
        f"# reference: {wrist_term_reference()}",
        "# value_deg is empty where the sample is invalid; see invalid_reason",
    ]
    for spec in table.terms:
        lines.append(f"# frame {spec.term_id}: {spec.frame}")
        lines.append(f"# sign_convention {spec.term_id}: {spec.sign_convention}")
    return lines


def to_csv(table: AnatomicalEventTable) -> str:
    """CSV text: ``#`` header lines (units, sign conventions), then rows."""
    buf = io.StringIO()
    buf.write("\n".join(_csv_header(table)) + "\n")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for r in table.rows:
        writer.writerow(
            [
                r.event,
                r.frame,
                repr(r.time_s),
                r.term_id,
                r.role,
                r.side,
                "" if math.isnan(r.value_deg) else repr(r.value_deg),
                r.invalid_reason,
                r.observability,
                r.receipt_path,
                ";".join(r.calibration),
            ]
        )
    return buf.getvalue()


_STR = {"type": "string", "minLength": 1}
EVENT_TABLE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Anatomical Event Table",
    "type": "object",
    "required": [
        "schema",
        "schema_version",
        "units",
        "handedness",
        "event_source",
        "reference",
        "events",
        "terms",
        "rows",
    ],
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "schema_version": {"const": SCHEMA_VERSION},
        "units": {
            "type": "object",
            "properties": {"angle": {"const": "deg"}, "time": {"const": "s"}},
            "required": ["angle", "time"],
        },
        "handedness": {"enum": ["right", "left"]},
        "event_source": _STR,
        "reference": _STR,
        "events": {
            "type": "object",
            "propertyNames": {"enum": list(EVENT_ORDER)},
            "minProperties": 1,
            "additionalProperties": {
                "type": "object",
                "required": ["frame", "time_s"],
                "properties": {
                    "frame": {"type": "integer", "minimum": 0},
                    "time_s": {"type": "number"},
                },
            },
        },
        "terms": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["unit", "frame", "sign_convention"],
                "properties": {
                    "unit": {"const": "deg"},
                    "frame": _STR,
                    "sign_convention": _STR,
                },
            },
        },
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": list(CSV_COLUMNS),
                "properties": {
                    "event": {"enum": list(EVENT_ORDER)},
                    "frame": {"type": "integer", "minimum": 0},
                    "time_s": {"type": "number"},
                    "term_id": _STR,
                    "role": {"enum": ["lead", "trail"]},
                    "side": {"enum": ["L", "R"]},
                    "value_deg": {"type": ["number", "null"]},
                    "invalid_reason": {"type": ["string", "null"]},
                    "observability": _STR,
                    "receipt_path": _STR,
                    "calibration": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "if": {"properties": {"value_deg": {"type": "null"}}},
                "then": {"properties": {"invalid_reason": _STR}},
            },
        },
    },
}

__all__ = [
    "CSV_COLUMNS",
    "EVENT_ORDER",
    "EVENT_TABLE_JSON_SCHEMA",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "UNITS",
    "AnatomicalEventTable",
    "EventRow",
    "anatomical_event_table",
    "to_csv",
    "to_json",
    "to_json_dict",
]
