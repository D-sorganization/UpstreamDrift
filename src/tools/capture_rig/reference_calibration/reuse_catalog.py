"""Named saved calibration choices from the existing capture catalog."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..capture_library import CaptureLibrary, LibraryEntry
from .reuse_evidence import read_bounded, source_result_path

MAX_CHOICES = 200


def _choice(entry: LibraryEntry, path: Path) -> dict[str, str]:
    source_result_path(entry.root, path.relative_to(entry.root).as_posix())
    payload = json.loads(read_bounded(path))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "capture-reference-solve/1"
        or payload.get("operator_reviewed") is not True
        or payload.get("capture_id") != entry.capture_id
    ):
        raise ValueError("Result is not an original reviewed estimate for this capture")
    reviewed = datetime.fromisoformat(payload["reviewed_utc"])
    if reviewed.utcoffset() is None:
        raise ValueError("Reviewed date needs a time zone")
    scene = payload.get("scene_id")
    if not isinstance(scene, str) or not 1 <= len(scene) <= 200:
        raise ValueError("Reviewed result needs a valid scene name")
    stamp = reviewed.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
    archived = " · Archived" if entry.archived else ""
    return {
        "label": f"{entry.title} · {stamp} · {scene}{archived}",
        "path": str(path.resolve()),
    }


def list_reviewed_layouts(request: dict[str, Any]) -> dict[str, Any]:
    """List bounded metadata; selecting a choice triggers full evidence review."""
    try:
        library = CaptureLibrary(Path(request["parameters"]["library_root"]))
        entries = library.catalog_entries(archived=None)
    except sqlite3.Error as exc:
        raise ValueError(f"Cannot read the capture library: {exc}") from exc
    choices: list[dict[str, str]] = []
    problems: list[str] = []
    scanned = 0
    for entry in entries:
        if entry.capture_id == request["capture_id"]:
            continue
        if entry.problem:
            problems.append(f"{entry.title}: {entry.problem}")
            continue
        for path in (entry.root / "reference_calibration/results").glob(
            "reviewed-*.json"
        ):
            scanned += 1
            if scanned > MAX_CHOICES:
                problems.append(
                    "Showing the first 200 saved results. Open a layout file to choose another."
                )
                return {"choices": choices, "problems": problems[:8]}
            try:
                choices.append(_choice(entry, path))
            except (ValueError, TypeError, KeyError, OSError) as exc:
                problems.append(f"{entry.title}: {exc}")
    return {
        "choices": sorted(choices, key=lambda item: item["label"]),
        "problems": problems[:8],
    }
