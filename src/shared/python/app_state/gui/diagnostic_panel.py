"""DiagnosticPanel — Qt widget showing diagnostic results with a Run button.

PyQt6 imports are guarded; callers should handle ``ImportError`` when
running in headless environments.
"""

from __future__ import annotations

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

from src.shared.python.app_state._diagnostic import DiagnosticEngine, DiagnosticResult
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_STATUS_COLORS = {
    "PASS": "#2ecc71",
    "FAIL": "#e74c3c",
    "SKIP": "#f39c12",
}


if _QT_AVAILABLE:

    class DiagnosticPanel(QWidget):  # type: ignore[misc]
        """Widget that runs and displays diagnostic check results.

        Provides a Run button that triggers
        :meth:`DiagnosticEngine.run_checks` and populates a table with
        the ``name``, ``status``, and ``message`` columns.

        Args:
            engine: The :class:`DiagnosticEngine` to query.
            parent: Optional parent widget.
        """

        _COL_NAME = 0
        _COL_STATUS = 1
        _COL_MESSAGE = 2
        _COLUMN_COUNT = 3

        def __init__(
            self,
            engine: DiagnosticEngine,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._engine = engine
            self._setup_ui()

        def _setup_ui(self) -> None:
            """Build the widget layout."""
            layout = QVBoxLayout(self)

            header_row = QHBoxLayout()
            header_row.addWidget(QLabel("Diagnostic Checks"))
            header_row.addStretch()
            self._run_btn = QPushButton("Run Diagnostics")
            self._run_btn.clicked.connect(self._on_run)
            header_row.addWidget(self._run_btn)
            layout.addLayout(header_row)

            self._table = QTableWidget(0, self._COLUMN_COUNT)
            self._table.setHorizontalHeaderLabels(["Check", "Status", "Message"])
            header = self._table.horizontalHeader()
            if header is not None:
                header.setSectionResizeMode(
                    self._COL_NAME, QHeaderView.ResizeMode.ResizeToContents
                )
                header.setSectionResizeMode(
                    self._COL_STATUS, QHeaderView.ResizeMode.ResizeToContents
                )
                header.setSectionResizeMode(
                    self._COL_MESSAGE, QHeaderView.ResizeMode.Stretch
                )
            self._table.setEditTriggers(
                QTableWidget.EditTrigger.NoEditTriggers  # type: ignore[attr-defined]
            )
            layout.addWidget(self._table)

        def _on_run(self) -> None:
            """Execute checks and populate the table."""
            results = self._engine.run_checks()
            self._populate_table(results)

        def _populate_table(self, results: list[DiagnosticResult]) -> None:
            """Fill the results table from a list of DiagnosticResult objects."""
            self._table.setRowCount(len(results))
            for row, result in enumerate(results):
                self._table.setItem(row, self._COL_NAME, QTableWidgetItem(result.name))
                status_item = QTableWidgetItem(result.status)
                color = _STATUS_COLORS.get(result.status, "#ffffff")
                status_item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(color)
                )
                self._table.setItem(row, self._COL_STATUS, status_item)
                self._table.setItem(
                    row, self._COL_MESSAGE, QTableWidgetItem(result.message)
                )

else:
    DiagnosticPanel = None  # type: ignore[assignment,misc]
