"""Canonical metrics record for motion-matching diagnostics.

Mirrors `motion_matching/shared/METRICS_SCHEMA.md` and the MATLAB
``+metrics/Metrics`` classdef.  The Python and MATLAB emitters are
guaranteed to produce byte-for-byte identical JSON for the same record.

Public API:

    Metrics            -- frozen dataclass per METRICS_SCHEMA.md
    SCHEMA_VERSION     -- the current canonical schema version (semver)
    legacy_struct_to_metrics -- backwards-compat shim for the old result struct
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

from src.shared.python.contracts import postcondition, precondition

SCHEMA_VERSION = "1.0.0"

_ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_VALID_OPTIONS = frozenset({1, 2, 3, 4})

# Field order is canonical — used for CSV round-tripping and for JSON
# emission with sort_keys=False where ordering matters for diff readability.
_FIELD_ORDER: tuple[str, ...] = (
    "swing_id",
    "option",
    "solver",
    "n_iterations",
    "rmse_clubhead_mm",
    "rmse_butt_mm",
    "rmse_orientation_deg",
    "clubhead_speed_at_impact_mph",
    "clubhead_speed_meas_mph",
    "total_work_J",
    "peak_power_W",
    "wall_clock_s",
    "git_commit",
    "matlab_version",
    "python_version",
    "timestamp_iso8601",
    "schema_version",
)


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


_NUMERIC_FIELDS: tuple[str, ...] = (
    "rmse_clubhead_mm",
    "rmse_butt_mm",
    "rmse_orientation_deg",
    "clubhead_speed_at_impact_mph",
    "clubhead_speed_meas_mph",
    "total_work_J",
    "peak_power_W",
    "wall_clock_s",
)
_NONNEG_FIELDS: tuple[str, ...] = (
    "rmse_clubhead_mm",
    "rmse_butt_mm",
    "rmse_orientation_deg",
    "clubhead_speed_at_impact_mph",
    "clubhead_speed_meas_mph",
    "wall_clock_s",
)


def _validate_timestamp(ts: str) -> None:
    if not _ISO8601_RE.match(ts):
        raise ValueError(
            f"timestamp_iso8601 must be ISO-8601 UTC ending in 'Z', got {ts!r}"
        )
    try:
        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        return
    except ValueError:
        pass
    try:
        datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError(f"timestamp_iso8601 unparseable: {ts!r}") from exc


def _validate_numeric(m: Metrics) -> None:
    for name in _NUMERIC_FIELDS:
        value = getattr(m, name)
        if not _is_finite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
    for name in _NONNEG_FIELDS:
        value = getattr(m, name)
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value!r}")


def _validate(m: Metrics) -> None:
    """Raise ``ValueError`` if any schema rule is violated."""
    if m.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version {m.schema_version!r} != current {SCHEMA_VERSION!r}"
        )
    if m.option not in _VALID_OPTIONS:
        raise ValueError(f"option must be in {{1,2,3,4}}, got {m.option!r}")
    if m.n_iterations < 0:
        raise ValueError(f"n_iterations must be >= 0, got {m.n_iterations!r}")
    _validate_numeric(m)
    if not _GIT_SHA_RE.match(m.git_commit):
        raise ValueError(
            f"git_commit must be 40 lowercase hex chars, got {m.git_commit!r}"
        )
    _validate_timestamp(m.timestamp_iso8601)


@dataclass(frozen=True)
class Metrics:
    """Canonical metrics record per METRICS_SCHEMA.md.

    All fields are required.  Validation runs in ``__post_init__``; any
    violation raises :class:`ValueError`.
    """

    swing_id: str
    option: int
    solver: str
    n_iterations: int
    rmse_clubhead_mm: float
    rmse_butt_mm: float
    rmse_orientation_deg: float
    clubhead_speed_at_impact_mph: float
    clubhead_speed_meas_mph: float
    total_work_J: float
    peak_power_W: float
    wall_clock_s: float
    git_commit: str
    matlab_version: str
    python_version: str
    timestamp_iso8601: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate(self)

    # --- JSON ----------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise to canonical JSON (sorted keys, no whitespace)."""
        payload = {name: getattr(self, name) for name in _FIELD_ORDER}
        return json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    @classmethod
    @precondition(
        lambda cls, s: isinstance(s, str) and len(s) > 0,
        "JSON payload must be a non-empty string",
    )
    @postcondition(lambda r: isinstance(r, Metrics), "from_json must return a Metrics")
    def from_json(cls, s: str) -> Metrics:
        """Deserialise canonical JSON back to a :class:`Metrics`."""
        data = json.loads(s)
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object, got {type(data).__name__}")
        return cls(**{name: data[name] for name in _FIELD_ORDER})

    # --- CSV -----------------------------------------------------------------

    def to_csv_row(self) -> dict[str, str]:
        """Serialise to a stringly-typed dict suitable for ``csv.DictWriter``."""
        return {name: str(getattr(self, name)) for name in _FIELD_ORDER}

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> Metrics:
        """Inverse of :meth:`to_csv_row`."""
        kwargs: dict[str, Any] = {}
        for f in fields(cls):
            raw = row[f.name]
            if f.type == "int" or f.type is int:
                kwargs[f.name] = int(raw)
            elif f.type == "float" or f.type is float:
                kwargs[f.name] = float(raw)
            else:
                kwargs[f.name] = raw
        return cls(**kwargs)


# --- Legacy compat -----------------------------------------------------------


def legacy_struct_to_metrics(struct: dict[str, Any]) -> Metrics:
    """Convert a legacy MATLAB result struct to a :class:`Metrics`.

    The legacy struct used the field set
    ``{swing_id, option, solver, rmse_clubhead, rmse_butt, rmse_orient,
    chs_impact, chs_meas, total_work, peak_power, wall_clock, git_commit,
    timestamp}`` (millimetres / degrees / mph / J / W / s / SHA / ISO).
    Missing optional fields are filled with safe defaults so the
    backwards-compat path remains lossless for the fields it does carry.
    """
    legacy_to_new = {
        "rmse_clubhead": "rmse_clubhead_mm",
        "rmse_butt": "rmse_butt_mm",
        "rmse_orient": "rmse_orientation_deg",
        "chs_impact": "clubhead_speed_at_impact_mph",
        "chs_meas": "clubhead_speed_meas_mph",
        "total_work": "total_work_J",
        "peak_power": "peak_power_W",
        "wall_clock": "wall_clock_s",
        "timestamp": "timestamp_iso8601",
    }
    converted = dict(asdict_safe(struct))
    for old, new in legacy_to_new.items():
        if old in converted and new not in converted:
            converted[new] = converted.pop(old)

    converted.setdefault("n_iterations", 0)
    converted.setdefault("matlab_version", "")
    converted.setdefault("python_version", "")
    converted.setdefault("schema_version", SCHEMA_VERSION)
    return Metrics(**{name: converted[name] for name in _FIELD_ORDER})


def asdict_safe(struct: Any) -> dict[str, Any]:
    """Coerce mapping-like input to a plain dict without mutating it."""
    if isinstance(struct, dict):
        return dict(struct)
    if hasattr(struct, "_asdict"):
        return dict(struct._asdict())
    raise TypeError(f"cannot convert {type(struct).__name__!r} to dict")


__all__ = [
    "Metrics",
    "SCHEMA_VERSION",
    "legacy_struct_to_metrics",
]
