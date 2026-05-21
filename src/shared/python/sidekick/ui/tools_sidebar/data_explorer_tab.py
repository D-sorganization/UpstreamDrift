"""Qt Data Explorer tab for the shared Sidekick sidebar."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .data_explorer_service import (
    DEFAULT_DATA_EXPLORER_MAX_FILE_SIZE_BYTES,
    DEFAULT_DATA_EXPLORER_PREVIEW_ROWS,
    DataExplorerError,
    DataExplorerPreview,
    DataExplorerService,
    _count_delimited_rows,
)
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .qt_compat import QtCore, QtWidgets, Signal
from .settings import SidebarTabSettingsDescriptor, SidebarTabSettingsSchema

DATA_EXPLORER_TAB_ID = "data_explorer"
DATA_EXPLORER_TAB_SETTINGS = SidebarTabSettingsDescriptor(
    schema=SidebarTabSettingsSchema(
        version=1,
        defaults={
            "preview_rows": DEFAULT_DATA_EXPLORER_PREVIEW_ROWS,
            "max_file_size_mb": DEFAULT_DATA_EXPLORER_MAX_FILE_SIZE_BYTES
            // (1024 * 1024),
        },
        allowed_keys=frozenset({"preview_rows", "max_file_size_mb"}),
    )
)


def build_data_explorer_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the lightweight Data Explorer tab."""
    widget = SidekickDataExplorerWidget(sidebar)
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP[DATA_EXPLORER_TAB_ID]["summary"])
    return widget


class _RowCountWorker(QtCore.QThread):
    """Background worker that counts rows in a delimited file off the Qt main thread.

    Emits ``progress`` with the current running count every 10 000 rows.
    Emits ``finished`` with the final row count when done (or cancelled).
    """

    progress = Signal(int)
    finished = Signal(int)

    def __init__(
        self,
        path: Path,
        cancel_event: threading.Event,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._path = path
        self._cancel_event = cancel_event

    def run(self) -> None:
        count = _count_delimited_rows(
            self._path,
            cancel_event=self._cancel_event,
            progress_cb=self.progress.emit,
        )
        self.finished.emit(count)


class SidekickDataExplorerWidget(QtWidgets.QWidget):
    """Small bounded data-file preview surface for Sidekick."""

    def __init__(self, sidebar: Any) -> None:
        super().__init__(sidebar)
        self.setObjectName("SidekickDataExplorerTab")
        self._sidebar = sidebar
        self._preview: DataExplorerPreview | None = None
        self._row_count_worker: _RowCountWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        path_row = QtWidgets.QHBoxLayout()
        self._path_input = QtWidgets.QLineEdit(self)
        self._path_input.setObjectName("SidekickDataExplorerPath")
        self._path_input.setPlaceholderText(
            "project-relative or absolute data file path"
        )
        self._path_input.setToolTip("Enter a CSV, TSV, JSON, Parquet, or Excel file.")
        path_row.addWidget(self._path_input, stretch=1)

        browse = QtWidgets.QPushButton("Browse", self)
        browse.setObjectName("SidekickDataExplorerBrowse")
        browse.setToolTip("Choose a project-scoped data file.")
        browse.clicked.connect(self._choose_file)
        path_row.addWidget(browse)

        self._show_files = QtWidgets.QPushButton("Show in Files Tab", self)
        self._show_files.setObjectName("SidekickDataExplorerShowFiles")
        self._show_files.setToolTip("Show this file in the sidebar's Files tab.")
        self._show_files.clicked.connect(self._show_in_files_tab)
        path_row.addWidget(self._show_files)

        self._show_os = QtWidgets.QPushButton("Show in OS Explorer", self)
        self._show_os.setObjectName("SidekickDataExplorerShowOS")
        self._show_os.setToolTip("Open this file's folder in the OS explorer.")
        self._show_os.clicked.connect(self._show_in_os_explorer)
        path_row.addWidget(self._show_os)

        self._load_button = QtWidgets.QPushButton("Load Preview", self)
        self._load_button.setObjectName("SidekickDataExplorerLoad")
        self._load_button.setToolTip("Preview the file using bounded Sidekick limits.")
        self._load_button.clicked.connect(self.load_preview)
        path_row.addWidget(self._load_button)
        layout.addLayout(path_row)

        self._status = QtWidgets.QLabel(
            DEFAULT_SIDEBAR_TAB_HELP[DATA_EXPLORER_TAB_ID]["summary"],
            self,
        )
        self._status.setObjectName("SidekickDataExplorerStatus")
        self._status.setWordWrap(True)
        self._status.setToolTip(
            "Shows preview mode, schema size, and validation errors."
        )
        layout.addWidget(self._status)

        # Progress bar and cancel button for large-file row counting
        count_row = QtWidgets.QHBoxLayout()
        self._progress_bar = QtWidgets.QProgressBar(self)
        self._progress_bar.setObjectName("SidekickDataExplorerProgress")
        self._progress_bar.setRange(0, 0)  # indeterminate by default
        self._progress_bar.setToolTip("Row count progress for large files.")
        self._progress_bar.setVisible(False)
        count_row.addWidget(self._progress_bar, stretch=1)

        self._cancel_button = QtWidgets.QPushButton("Cancel", self)
        self._cancel_button.setObjectName("SidekickDataExplorerCancel")
        self._cancel_button.setToolTip("Cancel the ongoing row count operation.")
        self._cancel_button.clicked.connect(self._cancel_row_count)
        self._cancel_button.setVisible(False)
        count_row.addWidget(self._cancel_button)
        layout.addLayout(count_row)

        self._preview_table = QtWidgets.QTableWidget(self)
        self._preview_table.setObjectName("SidekickDataExplorerPreviewTable")
        self._preview_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._preview_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        layout.addWidget(self._preview_table, stretch=1)

        options_row = QtWidgets.QHBoxLayout()
        self._columns_input = QtWidgets.QLineEdit(self)
        self._columns_input.setObjectName("SidekickDataExplorerColumns")
        self._columns_input.setPlaceholderText("column_a, column_b (blank = all)")
        self._columns_input.setToolTip(
            "Comma-separated columns to export or hand off to Data Processor."
        )
        options_row.addWidget(self._columns_input, stretch=1)

        self._variable_input = QtWidgets.QLineEdit(self)
        self._variable_input.setObjectName("SidekickDataExplorerVariable")
        self._variable_input.setPlaceholderText("workspace_variable_name")
        self._variable_input.setToolTip(
            "Workspace variable name for exported preview data."
        )
        options_row.addWidget(self._variable_input, stretch=1)
        layout.addLayout(options_row)

        action_row = QtWidgets.QHBoxLayout()
        self._export_button = QtWidgets.QPushButton("Export Preview", self)
        self._export_button.setObjectName("SidekickDataExplorerExport")
        self._export_button.setToolTip(
            "Export the selected preview columns into the shared Sidekick workspace."
        )
        self._export_button.clicked.connect(self.export_preview)
        action_row.addWidget(self._export_button)

        self._handoff_button = QtWidgets.QPushButton("Send to Data Processor", self)
        self._handoff_button.setObjectName("SidekickDataExplorerSendToDataProcessor")
        self._handoff_button.setToolTip(
            "Emit a structured Data Processor launch request for the active preview."
        )
        self._handoff_button.clicked.connect(self.send_to_data_processor)
        action_row.addWidget(self._handoff_button)
        layout.addLayout(action_row)

    def load_preview(self) -> None:
        """Load the current file path into the preview table."""
        try:
            preview = self._service().preview_file(self._path_input.text().strip())
        except DataExplorerError as exc:
            self._preview = None
            self._status.setText(str(exc))
            self._preview_table.setRowCount(0)
            self._preview_table.setColumnCount(0)
            return

        self._preview = preview
        self._variable_input.setText(Path(preview.source_path).stem + "_preview")
        self._populate_preview_table(preview)

        if preview.load_mode == "sampled":
            # Row count was done synchronously inside preview_file for small CSV;
            # for large files the count is already available via preview.total_rows.
            # Show it if available, otherwise start the async count.
            if preview.total_rows is not None:
                self._status.setText(
                    f"{preview.format.upper()} preview loaded: "
                    f"{preview.total_rows} rows, "
                    f"{preview.total_columns} columns ({preview.load_mode})."
                )
            else:
                self._start_row_count(Path(preview.source_path))
        else:
            row_count = (
                preview.total_rows if preview.total_rows is not None else "unknown"
            )
            self._status.setText(
                f"{preview.format.upper()} preview loaded: {row_count} rows, "
                f"{preview.total_columns} columns ({preview.load_mode})."
            )

    def export_preview(self) -> None:
        """Export the current preview selection to the shared workspace."""
        if self._preview is None:
            self._status.setText("Load a preview before exporting.")
            return
        try:
            variable = self._service().export_selection(
                self._preview,
                self._sidebar.registry,
                self._variable_input.text(),
                selected_columns=self._selected_columns(),
            )
        except DataExplorerError as exc:
            self._status.setText(str(exc))
            return

        self._sidebar.set_context_variable(
            variable.name,
            self._sidebar.registry.get(variable.name),
        )
        self._status.setText(
            f"Exported preview to workspace variable '{variable.name}'."
        )

    def send_to_data_processor(self) -> None:
        """Emit a structured request for a future Data Processor host handoff."""
        if self._preview is None:
            self._status.setText("Load a preview before sending to Data Processor.")
            return
        try:
            payload = self._service().build_data_processor_request(
                self._preview,
                selected_columns=self._selected_columns(),
            )
        except DataExplorerError as exc:
            self._status.setText(str(exc))
            return
        self._sidebar.request_tool_launch("data_processor", payload)
        self._status.setText("Prepared Data Processor handoff request.")

    def _start_row_count(self, path: Path) -> None:
        """Start an async row-count worker and show the progress bar."""
        self._cancel_row_count()  # stop any prior worker
        self._cancel_event = threading.Event()
        self._row_count_worker = _RowCountWorker(path, self._cancel_event, parent=self)
        self._row_count_worker.progress.connect(self._on_row_count_progress)
        self._row_count_worker.finished.connect(self._on_row_count_finished)
        self._progress_bar.setVisible(True)
        self._cancel_button.setVisible(True)
        self._load_button.setEnabled(False)
        self._status.setText("Counting rows…")
        self._row_count_worker.start()

    def _cancel_row_count(self) -> None:
        """Signal the current worker to stop and hide the progress UI."""
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._row_count_worker is not None:
            self._row_count_worker.wait()
            self._row_count_worker = None
        self._cancel_event = None
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._load_button.setEnabled(True)

    def _on_row_count_progress(self, count: int) -> None:
        self._status.setText(f"Counting rows… {count:,} so far")

    def _on_row_count_finished(self, count: int) -> None:
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._load_button.setEnabled(True)
        self._row_count_worker = None
        self._cancel_event = None
        if self._preview is not None:
            self._status.setText(
                f"{self._preview.format.upper()} preview loaded: "
                f"{count} rows, "
                f"{self._preview.total_columns} columns ({self._preview.load_mode})."
            )

    def _choose_file(self) -> None:
        filename, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose a data file",
            str(self._sidebar.project_root),
            "Data Files (*.csv *.tsv *.json *.parquet *.xls *.xlsx)",
        )
        if filename:
            self._path_input.setText(filename)

    def _show_in_files_tab(self) -> None:
        path = self._path_input.text().strip()
        if not path:
            return
        if hasattr(self._sidebar, "set_tab_visible"):
            self._sidebar.set_tab_visible("files", True)
            self._sidebar.setCurrentTab("files")

    def _show_in_os_explorer(self) -> None:
        from .qt_compat import QtCore, QtGui

        path = self._path_input.text().strip()
        if not path:
            return
        full_path = Path(path)
        if not full_path.is_absolute():
            full_path = self._sidebar.project_root / path
        if full_path.exists():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(full_path.parent))
            )

    def _populate_preview_table(self, preview: DataExplorerPreview) -> None:
        self._preview_table.setColumnCount(len(preview.columns))
        self._preview_table.setHorizontalHeaderLabels(
            [column.name for column in preview.columns]
        )
        self._preview_table.setRowCount(len(preview.preview_rows))
        for row_index, row in enumerate(preview.preview_rows):
            for column_index, column in enumerate(preview.columns):
                value = row.get(column.name)
                text = "" if value is None else str(value)
                self._preview_table.setItem(
                    row_index,
                    column_index,
                    QtWidgets.QTableWidgetItem(text),
                )

    def _selected_columns(self) -> list[str] | None:
        raw = self._columns_input.text().strip()
        if not raw:
            return None
        return [column.strip() for column in raw.split(",") if column.strip()]

    def _service(self) -> DataExplorerService:
        settings = self._sidebar.tab_settings(DATA_EXPLORER_TAB_ID)["values"]
        preview_rows = int(settings["preview_rows"])
        max_file_size_bytes = int(settings["max_file_size_mb"]) * 1024 * 1024
        return DataExplorerService(
            project_root=self._sidebar.project_root,
            preview_rows=preview_rows,
            max_file_size_bytes=max_file_size_bytes,
        )
