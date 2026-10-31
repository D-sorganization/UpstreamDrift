"""Desktop panel for the anatomical event table (HRG-E4, #13175).

Opens an event-table export (``event_table.to_json``, the API route's body) in
the shared widget; an invalid file is reported and never replaced by samples.
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.ui.anatomical_event_table_widget import (
    AnatomicalEventTableWidget,
)


class AnatomicalEventTablePanel(QWidget):
    """Open-file control, source and error lines over the shared table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        open_button = QPushButton("Open Event Table JSON…")
        open_button.clicked.connect(self._choose_file)
        self.source_label = QLabel("")
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.table_widget = AnatomicalEventTableWidget()
        row = QHBoxLayout()
        row.addWidget(open_button)
        row.addWidget(self.source_label, 1)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.error_label)
        layout.addWidget(self.table_widget)

    def open_document(self, path: Path) -> None:
        """Show the export at ``path``; on failure keep the table, set the error."""
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("event table export must be a JSON object")
            self.table_widget.set_document(document)
        except (OSError, ValueError, TypeError) as exc:
            self.error_label.setText(f"Could not open {path}: {exc}")
            return
        self.source_label.setText(str(path))
        self.error_label.setText("")

    def _choose_file(self) -> None:
        # noqa: gui-thread/ok - opens a modal file dialog; the export JSON is small.
        name, _ = QFileDialog.getOpenFileName(
            self, "Open Event Table JSON", "", "JSON (*.json)"
        )
        if name:
            self.open_document(Path(name))


__all__ = ["AnatomicalEventTablePanel"]
