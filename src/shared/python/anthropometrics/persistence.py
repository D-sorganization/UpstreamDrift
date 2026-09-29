"""JSON persistence for :class:`SubjectAnthropometrics`.

This module implements the schema-versioned JSON serialiser /
deserialiser referenced by the :class:`Reader` and :class:`Writer`
Protocols defined in :mod:`anthropometrics.contracts`. The wire
format is intentionally simple: a single JSON object containing
subject metadata (``subject_id``, ``height_m``, ``mass_kg``,
``sex``, ``age_years``, ``source_method``) and a list of every
:class:`SegmentProperties` for the subject. ``com_xyz_m`` and
``inertia_tensor`` are stored as nested lists.

A ``schema_version`` integer is written at the top level so future
breaking changes can be detected and rejected with a clear error.

Validation invariants (triangle inequality, positive-definite
inertia tensors, ...) are re-applied automatically during
:func:`load_subject` because :class:`SubjectAnthropometrics` and
:class:`SegmentProperties` validate on construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ._subject_anthropometrics import SubjectAnthropometrics
from .segment_properties import SegmentProperties

SCHEMA_VERSION: int = 1
"""Current on-disk schema version. Bumped on any breaking change."""

_DEFAULT_SUBJECTS_DIRNAME = ".golf_modeling_suite"
_DEFAULT_SUBJECTS_SUBDIR = "subjects"


# --------------------------------------------------------------------------- #
# Default location helper.                                                    #
# --------------------------------------------------------------------------- #
def default_subjects_dir() -> Path:
    """Return the default per-user directory for persisted subjects.

    Resolves to ``~/.golf_modeling_suite/subjects/``. The directory
    is **not** created here — :func:`save_subject` creates parent
    directories on demand instead.
    """
    return Path.home() / _DEFAULT_SUBJECTS_DIRNAME / _DEFAULT_SUBJECTS_SUBDIR


# --------------------------------------------------------------------------- #
# Save.                                                                       #
# --------------------------------------------------------------------------- #
def save_subject(record: SubjectAnthropometrics, path: Path) -> None:
    """Serialise *record* to *path* as JSON v1.

    Args:
        record: A fully-validated :class:`SubjectAnthropometrics`.
        path: Destination file. Parent directories are created
            automatically.

    Raises:
        TypeError: If *record* is not a :class:`SubjectAnthropometrics`.
    """
    if not isinstance(record, SubjectAnthropometrics):
        raise TypeError(
            f"record must be a SubjectAnthropometrics, got {type(record).__name__}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = _subject_to_dict(record)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Load.                                                                       #
# --------------------------------------------------------------------------- #
def load_subject(path: Path) -> SubjectAnthropometrics:
    """Read a JSON v1 file and return the validated record.

    Re-runs every :class:`SubjectAnthropometrics` /
    :class:`SegmentProperties` invariant on the loaded data, so
    corruption (e.g. an inertia tensor that no longer satisfies the
    triangle inequality) raises :class:`ValueError`.

    Args:
        path: Source JSON file written by :func:`save_subject`.

    Returns:
        The deserialised :class:`SubjectAnthropometrics`.

    Raises:
        FileNotFoundError: When *path* does not exist.
        ValueError: When the file is not valid JSON, is missing the
            ``schema_version`` key, has a schema version this code
            does not understand, or violates a contract invariant.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"subject record not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error.msg}") from error

    if not isinstance(payload, dict):
        raise ValueError(
            f"{path}: top-level JSON value must be an object, "
            f"got {type(payload).__name__}"
        )

    if "schema_version" not in payload:
        raise ValueError(f"{path}: missing required 'schema_version' field")

    schema_version = payload["schema_version"]
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version {schema_version!r} "
            f"(this build supports {SCHEMA_VERSION})"
        )

    return _subject_from_dict(payload)


# --------------------------------------------------------------------------- #
# Internals.                                                                  #
# --------------------------------------------------------------------------- #
def _segment_to_dict(seg: SegmentProperties) -> dict[str, Any]:
    """Return the JSON-serialisable dict form of *seg*."""
    return {
        "name": seg.name,
        "body_part_id": seg.body_part_id,
        "length_m": float(seg.length_m),
        "proximal_marker": seg.proximal_marker,
        "distal_marker": seg.distal_marker,
        "mass_kg": float(seg.mass_kg),
        "com_xyz_m": np.asarray(seg.com_xyz_m, dtype=float).tolist(),
        "inertia_tensor": np.asarray(seg.inertia_tensor, dtype=float).tolist(),
        "source_method": seg.source_method,
        "source_subject_height_m": float(seg.source_subject_height_m),
        "source_subject_mass_kg": float(seg.source_subject_mass_kg),
    }


def _subject_to_dict(record: SubjectAnthropometrics) -> dict[str, Any]:
    """Return the JSON-serialisable dict form of *record*."""
    return {
        "schema_version": SCHEMA_VERSION,
        "subject_id": record.subject_id,
        "height_m": float(record.height_m),
        "mass_kg": float(record.mass_kg),
        "age_years": (
            float(record.age_years) if record.age_years is not None else None
        ),
        "sex": record.sex,
        "source_method": record.source_method,
        "segments": [
            {"name": name, "properties": _segment_to_dict(props)}
            for name, props in record.segments
        ],
    }


def _segment_from_dict(data: Any) -> SegmentProperties:
    """Construct a :class:`SegmentProperties` from a JSON dict."""
    if not isinstance(data, dict):
        raise ValueError(
            f"segment properties payload must be an object, got {type(data).__name__}"
        )
    required = {
        "name",
        "body_part_id",
        "length_m",
        "proximal_marker",
        "distal_marker",
        "mass_kg",
        "com_xyz_m",
        "inertia_tensor",
        "source_method",
        "source_subject_height_m",
        "source_subject_mass_kg",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(
            f"segment properties payload missing required keys: {sorted(missing)}"
        )
    return SegmentProperties(
        name=data["name"],
        body_part_id=data["body_part_id"],
        length_m=float(data["length_m"]),
        proximal_marker=data["proximal_marker"],
        distal_marker=data["distal_marker"],
        mass_kg=float(data["mass_kg"]),
        com_xyz_m=np.asarray(data["com_xyz_m"], dtype=float),
        inertia_tensor=np.asarray(data["inertia_tensor"], dtype=float),
        source_method=data["source_method"],
        source_subject_height_m=float(data["source_subject_height_m"]),
        source_subject_mass_kg=float(data["source_subject_mass_kg"]),
    )


def _subject_from_dict(payload: dict[str, Any]) -> SubjectAnthropometrics:
    """Construct a :class:`SubjectAnthropometrics` from the parsed JSON."""
    required = {
        "subject_id",
        "height_m",
        "mass_kg",
        "source_method",
        "segments",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"subject payload missing required keys: {sorted(missing)}")

    raw_segments = payload["segments"]
    if not isinstance(raw_segments, list):
        raise ValueError(
            f"segments must be a JSON array, got {type(raw_segments).__name__}"
        )

    segments: list[tuple[str, SegmentProperties]] = []
    for entry in raw_segments:
        if not (isinstance(entry, dict) and "name" in entry and "properties" in entry):
            raise ValueError(
                "each segments entry must be an object with "
                "'name' and 'properties' keys"
            )
        seg = _segment_from_dict(entry["properties"])
        segments.append((str(entry["name"]), seg))

    age_raw = payload.get("age_years")
    age_years = float(age_raw) if age_raw is not None else None

    return SubjectAnthropometrics(
        subject_id=str(payload["subject_id"]),
        height_m=float(payload["height_m"]),
        mass_kg=float(payload["mass_kg"]),
        segments=tuple(segments),
        source_method=str(payload["source_method"]),
        age_years=age_years,
        sex=str(payload.get("sex", "unspecified")),
    )


__all__ = [
    "SCHEMA_VERSION",
    "default_subjects_dir",
    "load_subject",
    "save_subject",
]
