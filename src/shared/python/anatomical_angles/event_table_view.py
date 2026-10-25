"""Display view of the anatomical event table (HRG-E4, #13175).

Both surfaces render the strict-JSON export document
(:func:`anatomical_angles.event_table.to_json_dict`, which is also what
``POST /biomechanics/anatomical-event-table`` returns): the desktop widget
through :func:`event_table_view`, the web table through its TypeScript mirror
(``ui/src/components/analysis/anatomicalEventTable/eventTableView.ts``). The
one formatting rule lives here and in that mirror only, and a shared golden
fixture pins identical cell text on both sides.

Numbers are rounded half away from zero on their exact binary value, which is
what JavaScript ``Number.prototype.toFixed`` does, and negative zero prints
without a sign. Terms are read from the document, so a term added to the
export (for example the trunk and pelvis terms) needs no change here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from src.shared.python.anatomical_angles.event_table import SCHEMA_ID

COLUMNS: tuple[str, ...] = (
    "Event",
    "Frame",
    "Time (s)",
    "Term",
    "Side",
    "Value (deg)",
    "Observability",
    "Note",
)
MISSING_TEXT = "missing"
VALUE_DIGITS = 2
TIME_DIGITS = 3


@dataclass(frozen=True)
class EventTableView:
    """Text-only table: a caption, the column titles and one row per value."""

    caption: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        """JSON-shaped form, the shape of the golden fixture's ``view``."""
        return {
            "caption": self.caption,
            "columns": list(self.columns),
            "rows": [list(r) for r in self.rows],
        }


def format_fixed(value: float, digits: int) -> str:
    """``value`` with ``digits`` decimals, rounded as JavaScript ``toFixed``.

    Raises ``ValueError`` for a non-finite value or a negative ``digits``.
    """
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    if digits < 0:
        raise ValueError(f"digits must be >= 0, got {digits!r}")
    quantum = Decimal(1).scaleb(-digits)
    text = f"{Decimal(value + 0.0).quantize(quantum, rounding=ROUND_HALF_UP):f}"
    if text.startswith("-") and not text.strip("-0."):
        return text[1:]
    return text


def _caption(document: Mapping[str, Any]) -> str:
    units = document["units"]
    return (
        f"Events from {document['event_source']}; "
        f"{document['handedness']}-handed; "
        f"angles in {units['angle']}, time in {units['time']}; "
        f"reference: {document['reference']}"
    )


def _row(row: Mapping[str, Any], terms: Mapping[str, Any]) -> tuple[str, ...]:
    if row["term_id"] not in terms:
        raise ValueError(f"row term {row['term_id']!r} is not in the document terms")
    value = row["value_deg"]
    return (
        row["event"],
        str(row["frame"]),
        format_fixed(row["time_s"], TIME_DIGITS),
        row["term_id"],
        f"{row['role']} ({row['side']})",
        MISSING_TEXT if value is None else format_fixed(value, VALUE_DIGITS),
        row["observability"],
        row["invalid_reason"] or "",
    )


def event_table_view(document: Mapping[str, Any]) -> EventTableView:
    """Text rows of an exported event-table document, in export row order.

    Raises ``ValueError`` when ``document`` is not an anatomical event table
    export or a row names a term the document does not describe.
    """
    if not isinstance(document, Mapping) or document.get("schema") != SCHEMA_ID:
        raise ValueError(f"document schema must be {SCHEMA_ID!r}")
    terms = document["terms"]
    rows = tuple(_row(r, terms) for r in document["rows"])
    return EventTableView(_caption(document), COLUMNS, rows)


__all__ = [
    "COLUMNS",
    "MISSING_TEXT",
    "TIME_DIGITS",
    "VALUE_DIGITS",
    "EventTableView",
    "event_table_view",
    "format_fixed",
]
