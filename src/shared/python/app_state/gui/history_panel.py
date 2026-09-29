"""HistoryPanel — Qt widget showing the event log with a Clear button.

PyQt6 imports are guarded; callers should handle ``ImportError`` when
running in headless environments.
"""

from __future__ import annotations

try:
    from PyQt6.QtCore import Qt  # noqa: F401
    from PyQt6.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QListWidget,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    _QT_AVAILABLE = True
except ImportError:
    _QT_AVAILABLE = False

from src.shared.python.app_state._history_store import HistoryStore
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def _require_qt() -> None:
    """Raise ImportError if PyQt6 is not available."""
    if not _QT_AVAILABLE:
        raise ImportError(
            "PyQt6 is required for HistoryPanel. Install with: pip install PyQt6"
        )


if _QT_AVAILABLE:

    class HistoryPanel(QWidget):  # type: ignore[misc]
        """Widget that displays the application event log.

        Shows a scrollable list of events and provides a Clear button
        that removes all events from the attached :class:`HistoryStore`.

        Args:
            store: The :class:`HistoryStore` to display and manage.
            parent: Optional parent widget.
        """

        def __init__(
            self,
            store: HistoryStore,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._store = store
            self._setup_ui()
            self.refresh()

        def _setup_ui(self) -> None:
            """Build the widget layout."""
            layout = QVBoxLayout(self)

            header_row = QHBoxLayout()
            header_row.addWidget(QLabel("Application Event History"))
            header_row.addStretch()
            self._clear_btn = QPushButton("Clear")
            self._clear_btn.setToolTip("Remove all recorded events")
            self._clear_btn.clicked.connect(self._on_clear)
            header_row.addWidget(self._clear_btn)
            layout.addLayout(header_row)

            self._list = QListWidget()
            self._list.setAlternatingRowColors(True)
            layout.addWidget(self._list)

        def refresh(self) -> None:
            """Repopulate the list from the current store contents."""
            self._list.clear()
            for event in self._store.snapshot():
                ts = event.timestamp.strftime("%H:%M:%S")
                text = f"[{ts}] {event.type}"
                if event.payload:
                    payload_str = ", ".join(
                        f"{k}={v}" for k, v in event.payload.items()
                    )
                    text += f"  ({payload_str})"
                self._list.addItem(text)
            # Scroll to the most recent event
            self._list.scrollToBottom()

        def _on_clear(self) -> None:
            """Clear the store and refresh the view."""
            self._store.clear()
            self._list.clear()
            logger.debug("HistoryPanel: store cleared by user")

else:
    # Provide a stub so ``from ... import HistoryPanel`` doesn't fail in
    # headless environments — callers receive None instead of a widget.
    HistoryPanel = None  # type: ignore[assignment,misc]
