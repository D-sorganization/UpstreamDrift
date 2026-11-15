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
      },
      "table_kind": "coarse_eight_class"  # optional; Dempster / Zatsiorsky
    }

Coarse eight-class tables (:attr:`COARSE_TABLE_KIND`) ship only the published
Dempster (1955) or Zatsiorsky-Seluyanov (1985) segment classes. The legacy
``segment_name_map`` that duplicated trunk mass across partitions is **not**
used; :mod:`_humanoid_expansion` replaces it at estimate time (#15014).

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

COARSE_TABLE_KIND = "coarse_eight_class"


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
    table_kind = str(data.get("table_kind", ""))
    coarse_table = table_kind == COARSE_TABLE_KIND
    required_root = ("method", "citation", "segments")
    if not coarse_table:
        required_root = (*required_root, "segment_name_map")
    _require_keys(data, required_root, f"ratio table {path!s}")
    raw_segments = data["segments"]
    raw_name_map = data.get("segment_name_map", {})
    if not isinstance(raw_segments, dict) or not raw_segments:
        raise ValueError(f"ratio table {path!s} 'segments' must be a non-empty mapping")
    if not coarse_table:
        if not isinstance(raw_name_map, dict) or not raw_name_map:
            raise ValueError(
                f"ratio table {path!s} 'segment_name_map' must be a non-empty mapping"
            )
    elif raw_name_map:
        raise ValueError(
            f"ratio table {path!s} sets table_kind={COARSE_TABLE_KIND!r}; "
            "omit segment_name_map (humanoid expansion supplies the map)"
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
