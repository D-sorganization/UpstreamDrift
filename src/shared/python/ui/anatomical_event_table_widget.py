"""PyQt widget for the anatomical event table (HRG-E4, #13175).

Renders the event-table export document through
:func:`anatomical_angles.event_table_view.event_table_view`, the same text the
web table shows; it formats nothing itself. A missing value shows its reason
as the cell tooltip as well as in the Note column.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PyQt6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.anatomical_angles.event_table_view import (
    COLUMNS,
    MISSING_TEXT,
    EventTableView,
    event_table_view,
)

_VALUE_COLUMN = COLUMNS.index("Value (deg)")
_NOTE_COLUMN = COLUMNS.index("Note")


class AnatomicalEventTableWidget(QWidget):
    """Read-only table of anatomical terms sampled at swing events."""

    def __init__(
        self,
        document: Mapping[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.caption_label = QLabel("")
        self.caption_label.setWordWrap(True)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.caption_label)
        layout.addWidget(self.table)
        if document is not None:
            self.set_document(document)

    def set_document(self, document: Mapping[str, Any]) -> None:
        """Show ``document``; raises ``ValueError`` and keeps the table if invalid."""
        self._show(event_table_view(document))

    def clear(self) -> None:
        """Remove every row and the caption."""
        self.table.setRowCount(0)
        self.caption_label.setText("")

    def _show(self, view: EventTableView) -> None:
        self.caption_label.setText(view.caption)
        self.table.setRowCount(len(view.rows))
        for r, row in enumerate(view.rows):
            for c, text in enumerate(row):
                self.table.setItem(r, c, QTableWidgetItem(text))
            if row[_VALUE_COLUMN] == MISSING_TEXT:
                self.table.item(r, _VALUE_COLUMN).setToolTip(row[_NOTE_COLUMN])
        self.table.resizeColumnsToContents()


__all__ = ["AnatomicalEventTableWidget"]
