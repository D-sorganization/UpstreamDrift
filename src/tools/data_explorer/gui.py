"""GUI for the Data Explorer.

Provides a Qt :class:`MainWidget` that surfaces the existing
:mod:`data_explorer_app` discovery / loading helpers in an embeddable
form, plus a thin :class:`DataExplorerWindow` ``QMainWindow`` shell so
the tool can still launch as a standalone window.

The widget intentionally keeps Qt construction lightweight: it does not
trigger any disk scan on construction. Tests can therefore instantiate
``MainWidget(None)`` cheaply under ``QT_QPA_PLATFORM=offscreen``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logger_utils import get_logger
from src.tools.data_explorer.data_explorer_app import (
    SUPPORTED_EXTENSIONS,
    discover_datasets,
    load_dataset,
)

logger = get_logger(__name__)

__all__ = ["DataExplorerWindow", "MainWidget"]


class MainWidget(QtWidgets.QWidget):
    """Embeddable Data Explorer widget.

    Layout:
        - A toolbar with a directory chooser that triggers
          :func:`discover_datasets` against the chosen directory.
        - A table listing discovered datasets (name / format / size).
        - A status label summarising the current workspace.

    The widget never modifies the data files it inspects (matches the
    read-only invariant of :mod:`data_explorer_app`).
    """

    # Emitted when the user picks a directory and discovery completes.
    # Test hook: lets unit tests assert discovery without scraping the
    # status label text.
    datasetsDiscovered = QtCore.pyqtSignal(list)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self._datasets: list[Path] = []

        # --- Toolbar row ------------------------------------------------
        self._dir_label = QtWidgets.QLabel("Workspace: (none selected)")
        self._dir_label.setObjectName("DataExplorerDirLabel")
        self._choose_button = QtWidgets.QPushButton("Choose directory…")
        self._choose_button.setObjectName("DataExplorerChooseButton")
        self._choose_button.clicked.connect(self._on_choose_directory)

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.addWidget(self._dir_label, stretch=1)
        toolbar.addWidget(self._choose_button)

        # --- Dataset table ---------------------------------------------
        self._table = QtWidgets.QTableWidget(0, 3, self)
        self._table.setObjectName("DataExplorerTable")
        self._table.setHorizontalHeaderLabels(["Name", "Format", "Size (bytes)"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )

        # --- Status label ----------------------------------------------
        formats = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        self._status_label = QtWidgets.QLabel(f"Supported formats: {formats}")
        self._status_label.setObjectName("DataExplorerStatus")
        self._status_label.setStyleSheet("color: #888;")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(toolbar)
        layout.addWidget(self._table, stretch=1)
        layout.addWidget(self._status_label)

    # ------------------------------------------------------------------
    # Public API used by tests / embed adapter
    # ------------------------------------------------------------------

    def load_directory(self, directory: Path) -> list[Path]:
        """Discover datasets under ``directory`` and refresh the table.

        Args:
            directory: Directory to scan. Must exist.

        Returns:
            The list of discovered dataset paths (also stored on the
            widget and emitted via :pyattr:`datasetsDiscovered`).

        Raises:
            FileNotFoundError: Propagated from
                :func:`discover_datasets` when the directory is absent.
        """
        datasets = discover_datasets(directory)
        self._datasets = list(datasets)
        self._dir_label.setText(f"Workspace: {directory}")
        self._populate_table(datasets)
        self.datasetsDiscovered.emit(list(datasets))
        return datasets

    def cleanup(self) -> None:
        """Release any resources held by the widget.

        Currently a no-op — the widget does not own subscriptions or
        background threads — but kept on the public surface so the
        embed adapter can call it idempotently during host shutdown.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_choose_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose dataset directory"
        )
        if not directory:
            return
        try:
            self.load_directory(Path(directory))
        except FileNotFoundError:
            logger.warning("Chosen directory disappeared: %s", directory)

    def _populate_table(self, datasets: list[Path]) -> None:
        self._table.setRowCount(len(datasets))
        for row, path in enumerate(datasets):
            try:
                info = load_dataset(path)
            except (FileNotFoundError, ValueError, OSError) as exc:
                logger.warning("Skipping unreadable dataset %s: %s", path, exc)
                self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(path.name))
                self._table.setItem(row, 1, QtWidgets.QTableWidgetItem("error"))
                self._table.setItem(row, 2, QtWidgets.QTableWidgetItem("-"))
                continue
            self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(path.name))
            self._table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(info.get("format", "")))
            )
            self._table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(str(info.get("size_bytes", "")))
            )


class DataExplorerWindow(QtWidgets.QMainWindow):
    """Standalone shell that hosts :class:`MainWidget` in a window.

    Used when the launcher chooses :class:`LaunchMode.NEW_WINDOW`. The
    embed path goes through ``_DataExplorerEmbedAdapter`` instead.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Data Explorer")
        self._main_widget = MainWidget(self)
        self.setCentralWidget(self._main_widget)
        self.resize(900, 600)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m src.tools.data_explorer``."""
    if argv is None:
        argv = sys.argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    window = DataExplorerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
