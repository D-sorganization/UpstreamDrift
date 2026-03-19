"""Advanced diagnostics tracker for the Pendulum Simulator.

Captures, stores, and displays all errors, warnings, and diagnostic events
that occur during application lifetime.  Events are persisted to a JSON-Lines
file so they survive application restarts.

Usage
-----
    from .diagnostics import DiagnosticsTracker, get_tracker

    tracker = get_tracker()
    tracker.record("simulation_error", "ODE solver diverged at t=0.42", severity="error")

    # Open the diagnostics viewer from the menu bar:
    tracker.show_viewer(parent_widget)

Design by Contract
------------------
- All recorded events include a UTC timestamp, severity, category, and message.
- Events are immediately flushed to the log file.
- The viewer shows the most recent 500 events.
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .theme import SEVERITY_COLORS, STYLE_DIAGNOSTICS_DIALOG

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

_LOG_DIR = Path.home() / ".pendulum_simulator"
_DIAG_FILE = _LOG_DIR / "diagnostics.jsonl"
_MAX_VIEWER_EVENTS = 500


@dataclass
class DiagnosticEvent:
    """A single diagnostic event."""

    timestamp: str
    severity: str  # "info", "warning", "error", "critical"
    category: str  # e.g. "simulation", "ui", "import", "uncaught"
    message: str
    details: str = ""  # traceback or extra context
    source: str = ""  # file:line where the event was recorded
    extra: dict[str, Any] = field(default_factory=dict)


# Severity → color mapping for the viewer (imported from theme)
_SEVERITY_COLORS = SEVERITY_COLORS


# ---------------------------------------------------------------------------
# Singleton tracker
# ---------------------------------------------------------------------------

_instance: DiagnosticsTracker | None = None


def get_tracker() -> DiagnosticsTracker:
    """Return the global diagnostics tracker (lazily created)."""
    global _instance
    if _instance is None:
        _instance = DiagnosticsTracker()
    return _instance


class DiagnosticsTracker:
    """Collects and persists diagnostic events to a JSONL file.

    Also installs a global exception hook to capture uncaught exceptions.
    """

    def __init__(self) -> None:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._events: list[DiagnosticEvent] = []
        self._file = _DIAG_FILE
        self._load_history()
        self._install_exception_hook()
        logger.info(
            "DiagnosticsTracker initialized — log: %s (%d historical events)",
            self._file,
            len(self._events),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        category: str,
        message: str,
        *,
        severity: str = "error",
        details: str = "",
        source: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record a diagnostic event and flush to disk."""
        assert category is not None, "category must be provided"
        event = DiagnosticEvent(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            severity=severity,
            category=category,
            message=message,
            details=details,
            source=source,
            extra=extra or {},
        )
        self._events.append(event)
        self._flush(event)
        # Also log via standard logging
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL,
        }.get(severity, logging.ERROR)
        logger.log(
            log_level,
            "[DIAG] %s | %s: %s",
            severity.upper(),
            category,
            message,
        )

    def record_exception(
        self,
        category: str,
        exc: BaseException,
        *,
        context: str = "",
    ) -> None:
        """Convenience: record an exception with its full traceback."""
        assert category is not None, "category must be provided"
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        msg = f"{type(exc).__name__}: {exc}"
        if context:
            msg = f"{context} — {msg}"
        self.record(
            category,
            msg,
            severity="error",
            details=tb,
            source=self._caller_source(depth=2),
        )

    @property
    def events(self) -> list[DiagnosticEvent]:
        """Return all recorded events (most recent last)."""
        return list(self._events)

    @property
    def error_count(self) -> int:
        """Count of error and critical events."""
        return sum(1 for e in self._events if e.severity in ("error", "critical"))

    def clear(self) -> None:
        """Clear all events and truncate the log file."""
        self._events.clear()
        self._file.write_text("", encoding="utf-8")
        logger.info("Diagnostics log cleared")

    def show_viewer(self, parent: QWidget | None = None) -> None:
        """Open the diagnostics viewer dialog."""
        dlg = DiagnosticsViewer(self, parent)
        dlg.exec()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush(self, event: DiagnosticEvent) -> None:
        """Append a single event to the JSONL file."""
        try:
            with self._file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), default=str) + "\n")
        except OSError:
            pass  # Best-effort persistence

    def _load_history(self) -> None:
        """Load events from previous sessions."""
        if not self._file.exists():
            return
        try:
            for line in self._file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._events.append(DiagnosticEvent(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        except OSError:
            pass

    def _install_exception_hook(self) -> None:
        """Install a sys.excepthook that captures uncaught exceptions."""
        original_hook = sys.excepthook

        def _hook(
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_tb: Any,
        ) -> None:
            assert exc_type is not None, "exc_type must be provided"
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self.record(
                "uncaught_exception",
                f"{exc_type.__name__}: {exc_value}",
                severity="critical",
                details=tb_str,
            )
            # Call original hook so the error still prints to stderr
            original_hook(exc_type, exc_value, exc_tb)

        sys.excepthook = _hook

    @staticmethod
    def _caller_source(depth: int = 1) -> str:
        """Return file:line of the caller at the given stack depth."""
        import inspect

        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is not None:
                frame = frame.f_back
        if frame is not None:
            return f"{frame.f_code.co_filename}:{frame.f_lineno}"
        return ""


# ---------------------------------------------------------------------------
# Viewer dialog
# ---------------------------------------------------------------------------


class DiagnosticsViewer(QDialog):
    """Modal dialog that displays diagnostic events in a searchable table."""

    def __init__(
        self,
        tracker: DiagnosticsTracker,
        parent: QWidget | None = None,
    ) -> None:
        assert tracker is not None, "tracker must be provided"
        super().__init__(parent)
        self._tracker = tracker
        self.setWindowTitle("Diagnostics Tracker — Pendulum Simulator")
        self.resize(950, 600)
        self.setStyleSheet(STYLE_DIAGNOSTICS_DIALOG)
        self._build_ui()
        self._populate()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Header row
        header = QHBoxLayout()
        title = QLabel("🔍 Diagnostics Log")
        title.setFont(QFont("Sans", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #c0c0e0;")
        header.addWidget(title)

        self._count_label = QLabel()
        self._count_label.setStyleSheet("color: #808090; font-size: 11px;")
        header.addWidget(self._count_label)
        header.addStretch()

        # Severity filter
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(["All", "Critical", "Error", "Warning", "Info"])
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        header.addWidget(QLabel("Filter:"))
        header.addWidget(self._filter_combo)

        btn_clear = QPushButton("🗑 Clear Log")
        btn_clear.clicked.connect(self._on_clear)
        header.addWidget(btn_clear)

        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self._populate)
        header.addWidget(btn_refresh)

        layout.addLayout(header)

        # Events table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Time", "Severity", "Category", "Message", "Source"]
        )
        header_view = self._table.horizontalHeader()
        assert header_view is not None
        header_view.setStretchLastSection(True)
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.currentCellChanged.connect(self._on_row_selected)
        layout.addWidget(self._table, stretch=3)

        # Details panel
        details_label = QLabel("Details / Traceback:")
        details_label.setStyleSheet("color: #808090; font-size: 11px;")
        layout.addWidget(details_label)

        self._details = QTextEdit()
        self._details.setReadOnly(True)
        self._details.setMinimumHeight(120)
        layout.addWidget(self._details, stretch=1)

        # Footer
        footer = QHBoxLayout()
        self._log_path_label = QLabel(f"Log file: {_DIAG_FILE}")
        self._log_path_label.setStyleSheet("color: #505070; font-size: 10px;")
        footer.addWidget(self._log_path_label)
        footer.addStretch()

        btn_copy = QPushButton("📋 Copy Details")
        btn_copy.clicked.connect(self._copy_details)
        footer.addWidget(btn_copy)

        layout.addLayout(footer)

    def _populate(self) -> None:
        """Reload events from the tracker into the table."""
        severity_filter = self._filter_combo.currentText().lower()
        events = self._tracker.events[-_MAX_VIEWER_EVENTS:]

        if severity_filter != "all":
            events = [e for e in events if e.severity == severity_filter]

        self._table.setRowCount(len(events))
        self._displayed_events = events

        for row, event in enumerate(reversed(events)):  # newest first
            # Time — show local time
            try:
                dt = datetime.fromisoformat(event.timestamp)
                time_str = dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, OSError):
                time_str = event.timestamp[:19]

            items = [
                time_str,
                event.severity.upper(),
                event.category,
                event.message[:200],
                event.source.split("/")[-1].split("\\")[-1] if event.source else "",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                if col == 1:  # severity column — color coded
                    color = _SEVERITY_COLORS.get(event.severity, "#808080")
                    item.setForeground(QColor(color))
                    item.setFont(QFont("Sans", 10, QFont.Weight.Bold))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, col, item)

        error_count = self._tracker.error_count
        total = len(self._tracker.events)
        self._count_label.setText(
            f"{total} events total • {error_count} errors/critical"
        )

    def _on_row_selected(
        self, row: int, _col: int, _prev_row: int, _prev_col: int
    ) -> None:
        """Show details for the selected event."""
        # Events are displayed newest-first (reversed)
        if 0 <= row < len(self._displayed_events):
            event = list(reversed(self._displayed_events))[row]
            text = event.details if event.details else "(no additional details)"
            if event.extra:
                text += f"\n\nExtra data:\n{json.dumps(event.extra, indent=2)}"
            self._details.setPlainText(text)

    def _on_filter_changed(self, _text: str) -> None:
        self._populate()

    def _on_clear(self) -> None:
        self._tracker.clear()
        self._populate()
        self._details.clear()

    def _copy_details(self) -> None:
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setText(self._details.toPlainText())
