"""Load and save user-defined segment sets as JSON.

Schema v1::

    {
      "schema_version": 1,
      "segments": [
        {"a": "<marker_a>", "b": "<marker_b>",
         "geometry": "line"|"cylinder",
         "group": "<free-form>",
         "visible": true,
         "radius": 0.015}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_RADIUS_M = 0.015
_VALID_GEOMETRIES = ("line", "cylinder")


@dataclass(frozen=True)
class SegmentSpec:
    """A user-editable segment description."""

    a: str
    b: str
    geometry: str = "line"
    group: str = "auto"
    visible: bool = True
    radius: float = DEFAULT_RADIUS_M

    def __post_init__(self) -> None:
        if not isinstance(self.a, str) or not self.a:
            raise ValueError(
                f"SegmentSpec.a must be a non-empty string, got {self.a!r}"
            )
        if not isinstance(self.b, str) or not self.b:
            raise ValueError(
                f"SegmentSpec.b must be a non-empty string, got {self.b!r}"
            )
        if self.a == self.b:
            raise ValueError(
                f"SegmentSpec endpoints must differ, got a == b == {self.a!r}"
            )
        if self.geometry not in _VALID_GEOMETRIES:
            raise ValueError(
                f"SegmentSpec.geometry must be one of {_VALID_GEOMETRIES}, "
                f"got {self.geometry!r}"
            )
        if not isinstance(self.group, str) or not self.group:
            raise ValueError(
                f"SegmentSpec.group must be a non-empty string, got {self.group!r}"
            )
        if not isinstance(self.visible, bool):
            raise TypeError(
                f"SegmentSpec.visible must be bool, got {type(self.visible).__name__}"
            )
        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, (int, float))
            or self.radius <= 0.0
        ):
            raise ValueError(
                f"SegmentSpec.radius must be a positive number, got {self.radius!r}"
            )


@dataclass
class SegmentSet:
    """A persisted set of user segments."""

    segments: tuple[SegmentSpec, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION


def default_segment_set_path() -> Path:
    """Return the default JSON path for persisting user segments."""
    return Path.home() / ".golf_modeling_suite" / "c3d_viewer_segments.json"


def to_dict(segment_set: SegmentSet) -> dict[str, Any]:
    """Serialise a ``SegmentSet`` to a JSON-ready dict."""
    if not isinstance(segment_set, SegmentSet):
        raise TypeError(
            f"segment_set must be SegmentSet, got {type(segment_set).__name__}"
        )
    return {
        "schema_version": segment_set.schema_version,
        "segments": [asdict(s) for s in segment_set.segments],
    }


def from_dict(payload: dict[str, Any]) -> SegmentSet:
    """Parse a dict (from ``json.load``) into a ``SegmentSet``."""
    if not isinstance(payload, dict):
        raise TypeError(f"payload must be dict, got {type(payload).__name__}")
    version = int(payload.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported segment-set schema_version {version!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    raw_segments = payload.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("'segments' must be a list")
    parsed: list[SegmentSpec] = []
    for entry in raw_segments:
        if not isinstance(entry, dict):
            raise ValueError(
                f"segment entry must be a dict, got {type(entry).__name__}"
            )
        parsed.append(
            SegmentSpec(
                a=str(entry["a"]),
                b=str(entry["b"]),
                geometry=str(entry.get("geometry", "line")),
                group=str(entry.get("group", "auto")),
                visible=bool(entry.get("visible", True)),
                radius=float(entry.get("radius", DEFAULT_RADIUS_M)),
            )
        )
    return SegmentSet(segments=tuple(parsed), schema_version=version)


def save_segment_set(path: Path | str, segment_set: SegmentSet) -> Path:
    """Write ``segment_set`` to ``path`` as JSON, creating parents as needed."""
    if path is None:
        raise ValueError("path must be provided")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(to_dict(segment_set), f, indent=2, sort_keys=True)
    return out


def load_segment_set(path: Path | str) -> SegmentSet:
    """Load a segment set from ``path``."""
    if path is None:
        raise ValueError("path must be provided")
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"segment-set file not found: {src}")
    with src.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return from_dict(payload)


__all__ = [
    "DEFAULT_RADIUS_M",
    "SCHEMA_VERSION",
    "SegmentSet",
    "SegmentSpec",
    "default_segment_set_path",
    "from_dict",
    "load_segment_set",
    "save_segment_set",
    "to_dict",
]
