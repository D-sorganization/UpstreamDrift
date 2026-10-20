"""Swing-event table and CSV/JSON export of anatomical angles (HRG-E4, #13175).

The table samples the wrist and forearm series of :mod:`anatomical_angles`, and
optionally the lab-frame pelvis and thorax terms, at swing events. Address,
top, impact and finish come from the caller: an existing detector (for example
``swing_comparison.events.detect_events``) named in ``event_source``, which
every export carries. Pelvis reversal is detected here from the pelvis turn
when trunk series are given (:mod:`anatomical_angles.trunk_events`); its
definition is exported with it, and an undetected reversal is reported with a
reason, never guessed.

Terms, frames and sign conventions come from the one golf-term description
(:func:`anatomical_angles.terms.wrist_term_specs` and
:func:`anatomical_angles.trunk_pelvis.term_specs`); this module adds no term.
Lead and trail sides follow ``core.handedness``; trunk rows are ``midline``.
Every value carries its observability, receipt path and calibration ids; a
value that is NaN in the series stays missing (JSON ``null``, empty CSV cell)
with a reason.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.anatomical_angles.terms import (
    WristTermSpec,
    wrist_term_reference,
    wrist_term_specs,
)
from src.shared.python.anatomical_angles.trunk_events import (
    PELVIS_REVERSAL,
    PELVIS_REVERSAL_DEFINITION,
    SINGULAR_REASON,
    TrunkPelvisSeries,
    detect_pelvis_reversal,
    segment_role,
    trunk_term_degrees,
)
from src.shared.python.anatomical_angles.trunk_pelvis import (
    TrunkPelvisTermSpec,
    term_frame,
    term_reference,
    term_specs,
)
from src.shared.python.anatomical_angles.wrist import WristForearm, check_side
from src.shared.python.core.handedness import (
    lead_side,
    trail_side,
    validate_handedness,
)

SCHEMA_ID = "upstreamdrift.anatomical_event_table"
SCHEMA_VERSION = 2
#: Caller-supplied swing events in their required temporal order.
EVENT_ORDER: tuple[str, ...] = ("address", "top", "impact", "finish")
#: Every table event, in row order (pelvis reversal is detected, not supplied).
TABLE_EVENTS: tuple[str, ...] = ("address", PELVIS_REVERSAL, "top", "impact", "finish")
#: ``side`` of the unsided trunk and pelvis rows.
MIDLINE = "midline"
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
    """Rows in :data:`TABLE_EVENTS` order, then wrist and trunk registry order.

    ``event_detection`` maps each detected event to its definition;
    ``undetected_events`` maps a detectable event that was not found to why.
    """

    rows: tuple[EventRow, ...]
    events: dict[str, tuple[int, float]]
    handedness: str
    event_source: str
    terms: tuple[WristTermSpec, ...]
    trunk_terms: tuple[TrunkPelvisTermSpec, ...] = ()
    event_detection: dict[str, str] = field(default_factory=dict)
    undetected_events: dict[str, str] = field(default_factory=dict)


def _check_events(events: Mapping[str, Any], n: int) -> dict[str, int]:
    if not events:
        raise ValueError("at least one swing event is required")
    if PELVIS_REVERSAL in events:
        raise ValueError(
            "pelvis_reversal is detected from the pelvis series, not supplied"
        )
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


def _wrist_rows(
    arms: Mapping[str, WristForearm],
    sides: Mapping[str, str],
    event: str,
    frame: int,
    time: float,
) -> list[EventRow]:
    rows: list[EventRow] = []
    for spec in wrist_term_specs():
        side = sides[spec.role]
        if side not in arms:
            continue
        value, reason, series = _sample(arms[side], spec, frame)
        rows.append(
            EventRow(
                event,
                frame,
                time,
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
    return rows


def _trunk_rows(
    values: Mapping[str, np.ndarray],
    trunk: TrunkPelvisSeries,
    event: str,
    frame: int,
    time: float,
) -> list[EventRow]:
    rows: list[EventRow] = []
    for spec in term_specs():
        value = float(values[spec.term_id][frame])
        rows.append(
            EventRow(
                event,
                frame,
                time,
                spec.term_id,
                segment_role(spec),
                MIDLINE,
                value,
                SINGULAR_REASON if math.isnan(value) else "",
                trunk.observability,
                trunk.receipt_path,
                tuple(trunk.calibration),
            )
        )
    return rows


def _detect_events(
    values: Mapping[str, np.ndarray], frames: dict[str, int], n: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Add the pelvis reversal to ``frames``; return (detected, undetected)."""
    frame, why = detect_pelvis_reversal(
        values["pelvis_turn_isb"], frames.get("address", 0), frames.get("impact", n - 1)
    )
    if frame is None:
        return {}, {PELVIS_REVERSAL: why}
    frames[PELVIS_REVERSAL] = frame
    return {PELVIS_REVERSAL: PELVIS_REVERSAL_DEFINITION}, {}


def anatomical_event_table(
    arms: Mapping[str, WristForearm],
    events: Mapping[str, Any],
    time_s: ArrayLike,
    *,
    handedness: str,
    event_source: str,
    trunk: TrunkPelvisSeries | None = None,
) -> AnatomicalEventTable:
    """Wrist, forearm and (with ``trunk``) pelvis/thorax terms at ``events``.

    ``arms`` maps ``"L"``/``"R"`` to the series of that side; a missing side
    yields no rows for its role (nothing is filled in). ``events`` maps names
    from :data:`EVENT_ORDER` to frame indices that are non-decreasing in that
    order. ``time_s`` gives one time per frame. ``event_source`` names the
    detector or annotation that produced the frames. With ``trunk`` the table
    adds the registered trunk/pelvis terms and the detected pelvis reversal
    (searched between address and impact, or the whole series without them).
    Raises ``ValueError`` on any violated precondition, including neither
    arms nor trunk.
    """
    handedness = validate_handedness(handedness)
    if not isinstance(event_source, str) or not event_source.strip():
        raise ValueError("event_source must name the event detector or annotation")
    times = np.asarray(time_s, dtype=float).reshape(-1)
    if not np.all(np.isfinite(times)):
        raise ValueError("time_s must be finite")
    if not arms and trunk is None:
        raise ValueError("at least one arm (side 'L' or 'R') or trunk is required")
    _check_arms(arms, times.size)
    values: dict[str, np.ndarray] = {}
    if trunk is not None:
        values = trunk_term_degrees(trunk, handedness, times.size)
    frames = _check_events(events, times.size)
    detected: dict[str, str] = {}
    undetected: dict[str, str] = {}
    if trunk is not None:
        detected, undetected = _detect_events(values, frames, times.size)
    ordered = {e: frames[e] for e in TABLE_EVENTS if e in frames}
    sides = {"lead": lead_side(handedness), "trail": trail_side(handedness)}
    rows: list[EventRow] = []
    for event, frame in ordered.items():
        time = float(times[frame])
        rows.extend(_wrist_rows(arms, sides, event, frame, time))
        if trunk is not None:
            rows.extend(_trunk_rows(values, trunk, event, frame, time))
    return AnatomicalEventTable(
        tuple(rows),
        {e: (f, float(times[f])) for e, f in ordered.items()},
        handedness,
        event_source,
        wrist_term_specs() if arms else (),
        term_specs() if trunk is not None else (),
        detected,
        undetected,
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


def _trunk_meta(spec: TrunkPelvisTermSpec) -> dict[str, str]:
    return {
        "unit": UNITS["angle"],
        "role": segment_role(spec),
        "profile": "isb_thorax",
        "frame": term_frame(),
        "sign_convention": spec.sign_convention,
        "description": spec.description,
        "reference": term_reference(),
    }


def _reference(table: AnatomicalEventTable) -> str:
    refs = [wrist_term_reference()] if table.terms else []
    if table.trunk_terms:
        refs.append(term_reference())
    return "; ".join(refs)


def _event_entry(table: AnatomicalEventTable, event: str) -> dict[str, Any]:
    frame, time = table.events[event]
    entry: dict[str, Any] = {"frame": frame, "time_s": time}
    if event in table.event_detection:
        entry["detection"] = table.event_detection[event]
    return entry


def to_json_dict(table: AnatomicalEventTable) -> dict[str, Any]:
    """Strict-JSON document (no NaN) matching :data:`EVENT_TABLE_JSON_SCHEMA`."""
    return {
        "schema": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "units": dict(UNITS),
        "handedness": table.handedness,
        "event_source": table.event_source,
        "reference": _reference(table),
        "events": {e: _event_entry(table, e) for e in table.events},
        "undetected_events": dict(table.undetected_events),
        "terms": {
            **{s.term_id: _term_meta(s) for s in table.terms},
            **{s.term_id: _trunk_meta(s) for s in table.trunk_terms},
        },
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
        f"# reference: {_reference(table)}",
        "# value_deg is empty where the sample is invalid; see invalid_reason",
    ]
    for event, definition in table.event_detection.items():
        lines.append(f"# detection {event}: {definition}")
    for event, reason in table.undetected_events.items():
        lines.append(f"# undetected {event}: {reason}")
    frames = [(s.term_id, s.frame, s.sign_convention) for s in table.terms]
    frames += [(s.term_id, term_frame(), s.sign_convention) for s in table.trunk_terms]
    for term_id, frame, sign in frames:
        lines.append(f"# frame {term_id}: {frame}")
        lines.append(f"# sign_convention {term_id}: {sign}")
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
        "undetected_events",
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
            "propertyNames": {"enum": list(TABLE_EVENTS)},
            "minProperties": 1,
            "additionalProperties": {
                "type": "object",
                "required": ["frame", "time_s"],
                "properties": {
                    "frame": {"type": "integer", "minimum": 0},
                    "time_s": {"type": "number"},
                    "detection": _STR,
                },
            },
        },
        "undetected_events": {
            "type": "object",
            "propertyNames": {"enum": [PELVIS_REVERSAL]},
            "additionalProperties": _STR,
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
                    "event": {"enum": list(TABLE_EVENTS)},
                    "frame": {"type": "integer", "minimum": 0},
                    "time_s": {"type": "number"},
                    "term_id": _STR,
                    "role": {"enum": ["lead", "trail", "pelvis", "thorax"]},
                    "side": {"enum": ["L", "R", MIDLINE]},
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
    "MIDLINE",
    "SCHEMA_ID",
    "SCHEMA_VERSION",
    "TABLE_EVENTS",
    "UNITS",
    "AnatomicalEventTable",
    "EventRow",
    "anatomical_event_table",
    "to_csv",
    "to_json",
    "to_json_dict",
]
