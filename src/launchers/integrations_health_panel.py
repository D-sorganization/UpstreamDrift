"""Qt widget for the Integrations Health dashboard.

Provides :class:`IntegrationsHealthPanel`, a ``QWidget`` that displays a
live table of all configured integrations (MCP servers, CLI agents, API
adapters) and exposes Refresh / Copy Diagnostics actions.

The panel delegates all data collection to the Qt-free
:mod:`src.launchers.integrations_health_data` module so the logic stays
unit-testable in headless environments.

Issue #5643: feat(launcher): integrations health dashboard — one pane of glass
for clients, MCP, CLI, API.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QClipboard, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.launchers.integrations_health_data import (
    IntegrationRecord,
    collect_all,
    copy_diagnostics,
)
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Status → display colour mapping (Catppuccin-inspired, works on dark/light)
# ---------------------------------------------------------------------------
_STATUS_COLOURS: dict[str, str] = {
    "healthy": "#a6e3a1",  # green
    "configured": "#89dceb",  # sky
    "warning": "#f9e2af",  # yellow
    "error": "#f38ba8",  # red
    "unconfigured": "#6c7086",  # overlay0
    "unknown": "#cdd6f4",  # text
}

_COLUMNS = ("Kind", "Name", "Status", "Last Checked", "Notes")


class IntegrationsHealthPanel(QWidget):
    """One-pane-of-glass dashboard for all configured integrations.

    Displays a table with a row per integration, a **Refresh** button to
    re-run all probes, and a **Copy Diagnostics** button that copies a
    Markdown summary (with secrets redacted) to the clipboard.

    Usage::

        panel = IntegrationsHealthPanel(parent=some_widget)
        panel.show()
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._records: list[IntegrationRecord] = []
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct all child widgets and lay them out."""
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Title row
        title = QLabel("Integrations Health")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        root.addWidget(title)

        # Table
        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(list(_COLUMNS))
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setAlternatingRowColors(True)
        root.addWidget(self._table)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setToolTip("Re-probe all integrations")
        self._refresh_btn.clicked.connect(self.refresh)
        btn_row.addWidget(self._refresh_btn)

        self._copy_btn = QPushButton("Copy Diagnostics")
        self._copy_btn.setToolTip(
            "Copy a Markdown health report to clipboard (secrets redacted)"
        )
        self._copy_btn.clicked.connect(self._copy_diagnostics)
        btn_row.addWidget(self._copy_btn)

        btn_row.addStretch()

        self._status_label = QLabel("")
        btn_row.addWidget(self._status_label)

        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-run all integration probes and update the table."""
        self._refresh_btn.setEnabled(False)
        self._status_label.setText("Checking…")
        QApplication.processEvents()

        try:
            self._records = collect_all()
        except Exception as exc:  # noqa: BLE001
            logger.warning("collect_all raised unexpectedly: %s", exc)
            self._records = []

        self._populate_table(self._records)

        healthy = sum(1 for r in self._records if r.status in ("healthy", "configured"))
        total = len(self._records)
        self._status_label.setText(f"{healthy}/{total} OK")
        self._refresh_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _populate_table(self, records: list[IntegrationRecord]) -> None:
        """Replace table contents with *records*."""
        self._table.setRowCount(0)
        for row_idx, rec in enumerate(records):
            self._table.insertRow(row_idx)
            cells = [
                rec.kind,
                rec.name,
                rec.status,
                (
                    rec.last_checked.strftime("%Y-%m-%d %H:%M:%S")
                    if rec.last_checked
                    else "—"
                ),
                rec.detail or rec.last_error or "",
            ]
            colour = QColor(_STATUS_COLOURS.get(rec.status, "#cdd6f4"))
            for col_idx, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
                )
                if col_idx == 2:  # Status column — colour the cell
                    item.setBackground(colour)
                self._table.setItem(row_idx, col_idx, item)

    def _copy_diagnostics(self) -> None:
        """Copy a Markdown diagnostics report to the system clipboard."""
        try:
            md = copy_diagnostics(self._records)
        except Exception as exc:  # noqa: BLE001
            logger.warning("copy_diagnostics failed: %s", exc)
            return

        clipboard: QClipboard | None = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(md)
            self._status_label.setText("Copied!")
        else:
            logger.warning("No clipboard available")


__all__ = ["IntegrationsHealthPanel"]
