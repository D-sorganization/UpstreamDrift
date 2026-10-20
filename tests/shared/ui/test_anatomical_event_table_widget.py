"""Anatomical event table widget (HRG-E4, #13175): headless, view driven.

The widget must show exactly the golden cells the web table shows for the
same export document (shared fixture under ``ui/``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6.QtWidgets")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.anatomical_angles.event_table_view import (  # noqa: E402
    MISSING_TEXT,
)
from src.shared.python.ui.anatomical_event_table_widget import (  # noqa: E402
    AnatomicalEventTableWidget,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = (
    ROOT
    / "ui/src/components/analysis/anatomicalEventTable/fixtures/eventTableView.json"
)


@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _cells(widget: AnatomicalEventTableWidget) -> list[list[str]]:
    table = widget.table
    return [
        [table.item(r, c).text() for c in range(table.columnCount())]
        for r in range(table.rowCount())
    ]


def _headers(widget: AnatomicalEventTableWidget) -> list[str]:
    table = widget.table
    return [table.horizontalHeaderItem(c).text() for c in range(table.columnCount())]


def test_widget_shows_the_golden_cells_shared_with_the_web(app, fixture) -> None:
    widget = AnatomicalEventTableWidget(fixture["document"])
    assert _headers(widget) == fixture["view"]["columns"]
    assert _cells(widget) == fixture["view"]["rows"]
    assert widget.caption_label.text() == fixture["view"]["caption"]


def test_widget_starts_empty_and_can_be_cleared(app, fixture) -> None:
    widget = AnatomicalEventTableWidget()
    assert widget.table.rowCount() == 0
    widget.set_document(fixture["document"])
    assert widget.table.rowCount() == len(fixture["view"]["rows"])
    widget.clear()
    assert widget.table.rowCount() == 0
    assert widget.caption_label.text() == ""


def test_missing_value_cell_carries_its_reason_as_tooltip(app, fixture) -> None:
    widget = AnatomicalEventTableWidget(fixture["document"])
    value_col = fixture["view"]["columns"].index("Value (deg)")
    note_col = fixture["view"]["columns"].index("Note")
    missing = [
        r
        for r in range(widget.table.rowCount())
        if widget.table.item(r, value_col).text() == MISSING_TEXT
    ]
    assert missing
    for r in missing:
        item = widget.table.item(r, value_col)
        assert item.toolTip() == widget.table.item(r, note_col).text() != ""


def test_cells_are_read_only(app, fixture) -> None:
    widget = AnatomicalEventTableWidget(fixture["document"])
    assert widget.table.editTriggers() == widget.table.EditTrigger.NoEditTriggers


def test_invalid_document_is_rejected_without_touching_the_table(app, fixture) -> None:
    widget = AnatomicalEventTableWidget(fixture["document"])
    before = _cells(widget)
    with pytest.raises(ValueError):
        widget.set_document({"schema": "nope"})
    assert _cells(widget) == before
