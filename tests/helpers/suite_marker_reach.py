"""Suite-marker reach and directory-based inference helpers (#14947)."""

from __future__ import annotations

import os
from typing import Any

import pytest

from tests.support.suite_markers import item_has_suite_marker

DIRECTORY_SUITE_MARKERS: tuple[tuple[str, str], ...] = (
    ("tests/unit/", "unit"),
    ("tests/integration/", "integration"),
)

OPT_OUT_MARKER: str = "no_auto_suite_mark"
AUTO_MARK_ENV_VAR: str = "UD_AUTO_MARK_BY_DIRECTORY"


def infer_directory_suite_marker(nodeid_or_path: str) -> str | None:
    """Infer suite marker ('unit' or 'integration') from normalized path."""
    norm = str(nodeid_or_path).replace("\\", "/")
    for prefix, marker in DIRECTORY_SUITE_MARKERS:
        if norm.startswith(prefix) or f"/{prefix}" in norm or prefix in norm:
            return marker
    return None


def _is_auto_mark_enabled(enabled: bool | None = None) -> bool:
    if enabled is not None:
        return enabled
    return os.environ.get(AUTO_MARK_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def apply_directory_suite_auto_markers(
    items: list[Any], *, enabled: bool | None = None
) -> int:
    """Apply inferred suite markers to unmarked items in mapped directories."""
    if not _is_auto_mark_enabled(enabled):
        return 0

    count = 0
    for item in items:
        if item_has_suite_marker(item):
            continue
        markers = getattr(item, "iter_markers", None)
        if markers is not None and any(
            m.name == OPT_OUT_MARKER for m in item.iter_markers()
        ):
            continue
        nodeid = getattr(item, "nodeid", str(item))
        inferred = infer_directory_suite_marker(nodeid)
        if inferred is not None:
            marker = getattr(pytest.mark, inferred)
            if hasattr(item, "add_marker"):
                item.add_marker(marker)
            count += 1
    return count


def summarize_gate_selectability(
    total_collected: int, selected_count: int
) -> dict[str, int | float | str]:
    """Summarize gate selectability metrics and generate summary line."""
    deselected = total_collected - selected_count
    pct = (selected_count / total_collected * 100.0) if total_collected > 0 else 0.0
    return {
        "selected": selected_count,
        "total": total_collected,
        "deselected": deselected,
        "selected_pct": round(pct, 2),
        "summary_line": f"selected/total: {selected_count}/{total_collected} ({pct:.1f}%)",
    }
