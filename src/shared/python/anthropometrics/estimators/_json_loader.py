"""JSON ratio-table loader for the file-backed estimators.

The JSON schema (shared by :file:`ratios/dempster_1955.json` and
:file:`ratios/zatsiorsky_seluyanov_1985.json`) is::

    {
      "method": "<method_name>",
      "citation": "<bibliographic citation>",
      "segments": {
        "<class_id>": {
          "mass_ratio": <float>,
          "length_ratio": <float>,
          "com_proximal_ratio": <float>,
          "gyration_radii": {
            "sagittal": <float>,
            "transverse": <float>,
            "longitudinal": <float>
          }
        },
        ...
      },
      "segment_name_map": {
        "<anatomical_name>": "<class_id>",
        ...
      }
    }

This module parses that schema into the same in-memory
representation (:class:`SegmentRatios` + a flat name map) used by
the de Leva wrapper, so all three estimators run through one
shared driver in :mod:`_base`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._base import SegmentRatios


@dataclass(frozen=True)
class LoadedRatioTable:
    """Parsed ratio-table JSON file."""

    method_name: str
    citation: str
    segment_classes: dict[str, SegmentRatios]
    segment_name_map: dict[str, str]


def _require_keys(obj: dict[str, Any], keys: tuple[str, ...], context: str) -> None:
    """Raise ``ValueError`` if any *keys* are missing from *obj*."""
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ValueError(f"{context} is missing required keys: {sorted(missing)}")


def _parse_segment_ratios(class_id: str, raw: dict[str, Any]) -> SegmentRatios:
    """Convert a single segments-table entry into a :class:`SegmentRatios`."""
    _require_keys(
        raw,
        ("mass_ratio", "length_ratio", "com_proximal_ratio", "gyration_radii"),
        f"segment {class_id!r}",
    )
    gyr = raw["gyration_radii"]
    if not isinstance(gyr, dict):
        raise ValueError(
            f"segment {class_id!r} gyration_radii must be a mapping, "
            f"got {type(gyr).__name__}"
        )
    _require_keys(
        gyr,
        ("sagittal", "transverse", "longitudinal"),
        f"segment {class_id!r} gyration_radii",
    )
    return SegmentRatios(
        mass_ratio=float(raw["mass_ratio"]),
        length_ratio=float(raw["length_ratio"]),
        com_proximal_ratio=float(raw["com_proximal_ratio"]),
        gyration_sagittal=float(gyr["sagittal"]),
        gyration_transverse=float(gyr["transverse"]),
        gyration_longitudinal=float(gyr["longitudinal"]),
    )


def load_ratio_table(path: Path) -> LoadedRatioTable:
    """Parse a ratio-table JSON file into a :class:`LoadedRatioTable`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the JSON is structurally invalid (missing keys, wrong
        types, empty tables, name-map references to unknown
        classes).
    """
    if not path.exists():
        raise FileNotFoundError(f"ratio table file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"ratio table {path!s} root must be an object, got {type(data).__name__}"
        )
    _require_keys(
        data,
        ("method", "citation", "segments", "segment_name_map"),
        f"ratio table {path!s}",
    )
    raw_segments = data["segments"]
    raw_name_map = data["segment_name_map"]
    if not isinstance(raw_segments, dict) or not raw_segments:
        raise ValueError(f"ratio table {path!s} 'segments' must be a non-empty mapping")
    if not isinstance(raw_name_map, dict) or not raw_name_map:
        raise ValueError(
            f"ratio table {path!s} 'segment_name_map' must be a non-empty mapping"
        )

    classes = {
        class_id: _parse_segment_ratios(class_id, raw)
        for class_id, raw in raw_segments.items()
    }
    name_map = {str(k): str(v) for k, v in raw_name_map.items()}

    unknown = sorted({c for c in name_map.values() if c not in classes})
    if unknown:
        raise ValueError(
            f"ratio table {path!s} segment_name_map references unknown "
            f"classes: {unknown}"
        )

    return LoadedRatioTable(
        method_name=str(data["method"]),
        citation=str(data["citation"]),
        segment_classes=classes,
        segment_name_map=name_map,
    )
