"""Shared Excel swing-event label normalization (CO-00 #10604).

Workbook sheets alternate between bare labels (``A``) and equals-suffixed
labels (``A=``). All loaders must share one normalization so event sample
IDs are not silently dropped.
"""

from __future__ import annotations

from typing import Any

# Single-letter swing events plus club-head speed field.
_EVENT_LABELS: frozenset[str] = frozenset({"A", "T", "I", "F", "CHS"})


def normalize_event_label(cell: Any) -> str | None:
    """Return canonical event label or ``None`` when the cell is not an event.

    Accepts ``A`` / ``A=`` / ``A =`` and ``CHS`` / ``CHS=``. Does not invent
    labels for unknown cells.
    """
    if cell is None:
        return None
    if isinstance(cell, float) and cell != cell:  # NaN
        return None
    text = str(cell).strip()
    if not text:
        return None
    if text.endswith("="):
        text = text[:-1].strip()
    if not text:
        return None
    key = text.upper() if text.upper() == "CHS" else text
    if key not in _EVENT_LABELS:
        return None
    return key


def parse_event_marker_cells(row: list[Any] | tuple[Any, ...]) -> dict[str, float]:
    """Parse a row-1 style event header into canonical label -> value map.

    Values that cannot be coerced to float are omitted (fail soft for the
    field, not the whole row). Empty / missing companion cells are omitted.
    """
    if row is None:
        raise ValueError("event marker row must be provided")
    out: dict[str, float] = {}
    for index, cell in enumerate(row):
        label = normalize_event_label(cell)
        if label is None:
            continue
        if index + 1 >= len(row):
            continue
        raw = row[index + 1]
        if raw is None:
            continue
        if isinstance(raw, float) and raw != raw:
            continue
        try:
            out[label] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


def axis_component(value: Any, *, default: float) -> float:
    """Convert an axis cell to float without treating ``0.0`` as missing.

    ``value or default`` is incorrect for direction cosines: a real zero
    component must be preserved. Only ``None`` and NaN fall back to
    ``default``.
    """
    if value is None:
        return float(default)
    if isinstance(value, float) and value != value:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"axis component is not numeric: {value!r}") from exc
