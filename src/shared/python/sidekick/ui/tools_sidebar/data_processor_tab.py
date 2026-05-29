"""Lazy Sidekick adapter for the optional Data Processor tab."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any

from . import design_tokens as theme
from .help_content import DEFAULT_SIDEBAR_TAB_HELP
from .qt_compat import QtWidgets
from .registry import WorkspaceRegistry, WorkspaceVariable

logger = logging.getLogger(__name__)

DATA_PROCESSOR_TAB_ID = "data_processor"
DEFAULT_DATA_PROCESSOR_VARIABLE_NAME = "data_processor_result"


class DataProcessorTabError(ValueError):
    """Structured user-facing error for Sidekick Data Processor actions."""


def build_data_processor_tab(sidebar: Any) -> QtWidgets.QWidget:
    """Build the optional embedded Data Processor tab."""
    try:
        widget = SidekickDataProcessorTab(sidebar)
    except Exception as exc:  # noqa: BLE001 - optional runtime surface
        logger.debug("Data Processor unavailable for Sidekick: %s", exc)
        return _placeholder(
            sidebar,
            "Data Processor",
            "Data Processor is unavailable because optional UI or runtime "
            "dependencies could not be loaded.",
        )
    widget.setToolTip(DEFAULT_SIDEBAR_TAB_HELP[DATA_PROCESSOR_TAB_ID]["summary"])
    return widget


def export_data_processor_frame(
    frame: Any,
    registry: WorkspaceRegistry,
    variable_name: str,
    *,
    selected_columns: list[str] | None = None,
) -> WorkspaceVariable:
    """Export the current Data Processor frame into the shared workspace."""
    name = variable_name.strip()
    if not name:
        raise DataProcessorTabError("Workspace variable name must be non-empty.")
    available = _frame_columns(frame)
    columns = _resolve_selected_columns(available, selected_columns)
    records = _frame_records(frame, columns)
    if len(columns) == 1:
        value: Any = [_normalize_cell(record.get(columns[0])) for record in records]
    else:
        value = [
            {column: _normalize_cell(record.get(column)) for column in columns}
            for record in records
        ]
    try:
        return registry.set(name, value)
    except ValueError as exc:
        raise DataProcessorTabError(str(exc)) from exc


class SidekickDataProcessorTab(QtWidgets.QWidget):
    """Wrap the shared Data Processor widget with Sidekick workspace export."""

    def __init__(self, sidebar: Any) -> None:
        super().__init__(sidebar)
        self.setObjectName("SidekickDataProcessorTab")
        self._sidebar = sidebar
        self._processor_widget = _build_data_processor_widget(self)
        self._build_ui()
        self._connect_optional_signals()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._processor_widget, stretch=1)

        export_group = QtWidgets.QGroupBox("Workspace Export", self)
        export_group.setToolTip(
            "Export the current Data Processor output into the shared "
            "Sidekick workspace."
        )
        export_layout = QtWidgets.QVBoxLayout(export_group)
        export_layout.setContentsMargins(8, 8, 8, 8)
        export_layout.setSpacing(8)

        self._status = QtWidgets.QLabel(
            "Export the current dataset or a selected column into the workspace.",
            export_group,
        )
        self._status.setObjectName("SidekickDataProcessorStatus")
        self._status.setWordWrap(True)
        self._status.setToolTip(
            "Shows workspace-export validation errors and success messages."
        )
        export_layout.addWidget(self._status)

        self._columns_input = QtWidgets.QLineEdit(export_group)
        self._columns_input.setObjectName("SidekickDataProcessorColumns")
        self._columns_input.setPlaceholderText("column_a, column_b (blank = all)")
        self._columns_input.setToolTip(
            "Comma-separated columns to export from the current Data Processor dataset."
        )
        export_layout.addWidget(self._columns_input)

        row = QtWidgets.QHBoxLayout()
        self._variable_input = QtWidgets.QLineEdit(export_group)
        self._variable_input.setObjectName("SidekickDataProcessorVariable")
        self._variable_input.setText(DEFAULT_DATA_PROCESSOR_VARIABLE_NAME)
        self._variable_input.setToolTip(
            "Workspace variable name for the exported Data Processor results."
        )
        row.addWidget(self._variable_input, stretch=1)

        export_button = QtWidgets.QPushButton("Export to Workspace", export_group)
        export_button.setObjectName("SidekickDataProcessorExportWorkspace")
        export_button.setToolTip(
            "Add the current Data Processor result set to the shared "
            "Sidekick workspace."
        )
        export_button.clicked.connect(self.export_to_workspace)
        row.addWidget(export_button)
        export_layout.addLayout(row)

        layout.addWidget(export_group)

    def export_to_workspace(self) -> None:
        """Export the current Data Processor dataset into the shared workspace."""
        try:
            frame = _current_frame(self._processor_widget)
            variable = export_data_processor_frame(
                frame,
                self._sidebar.registry,
                self._variable_input.text(),
                selected_columns=self._selected_columns(),
            )
        except DataProcessorTabError as exc:
            self._status.setText(str(exc))
            return
        self._sidebar.refresh_workspace()
        self._status.setText(
            f"Exported Data Processor results to workspace variable '{variable.name}'."
        )

    def _selected_columns(self) -> list[str] | None:
        raw = self._columns_input.text().strip()
        if not raw:
            return None
        return [column.strip() for column in raw.split(",") if column.strip()]

    def _connect_optional_signals(self) -> None:
        data_loaded = getattr(self._processor_widget, "data_loaded", None)
        if data_loaded is None or not hasattr(data_loaded, "connect"):
            return
        data_loaded.connect(self._suggest_workspace_name)

    def _suggest_workspace_name(self, path: str) -> None:
        if self._variable_input.text().strip() != DEFAULT_DATA_PROCESSOR_VARIABLE_NAME:
            return
        stem = Path(path).stem.strip()
        if stem:
            self._variable_input.setText(f"{stem}_processed")


def _build_data_processor_widget(parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
    module = importlib.import_module("sidekick.ui.widgets.data_processor_widget")
    widget_class = module.DataProcessorWidget
    return widget_class(parent)


def _current_frame(widget: Any) -> Any:
    engine = getattr(widget, "engine", None)
    frame = getattr(engine, "data", None)
    if frame is None:
        raise DataProcessorTabError(
            "Load data in the Data Processor before exporting to the workspace."
        )
    columns = _frame_columns(frame)
    if not columns:
        raise DataProcessorTabError(
            "Load data in the Data Processor before exporting to the workspace."
        )
    return frame


def _frame_columns(frame: Any) -> list[str]:
    columns = getattr(frame, "columns", None)
    if columns is None:
        raise DataProcessorTabError("Current Data Processor results are not tabular.")
    return [str(column) for column in columns]


def _resolve_selected_columns(
    available: list[str],
    selected_columns: list[str] | None,
) -> list[str]:
    if not selected_columns:
        return available
    normalized = [column.strip() for column in selected_columns if column.strip()]
    missing = [column for column in normalized if column not in available]
    if missing:
        raise DataProcessorTabError(
            f"Selected columns are not available in the current dataset: {missing}"
        )
    return normalized


def _frame_records(frame: Any, columns: list[str]) -> list[dict[str, Any]]:
    try:
        selected = frame[columns]
        records = selected.to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001 - normalize workspace export failures
        raise DataProcessorTabError(
            "Current Data Processor results could not be exported to the workspace."
        ) from exc
    return [
        {str(key): _normalize_cell(value) for key, value in record.items()}
        for record in records
    ]


def _normalize_cell(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _placeholder(
    sidebar: Any,
    title: str,
    message: str,
) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget(sidebar)
    widget.setObjectName(theme.SIDEKICK_PLACEHOLDER_OBJECT_NAME)
    layout = QtWidgets.QVBoxLayout(widget)
    label = QtWidgets.QLabel(title, widget)
    label.setObjectName(theme.SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME)
    label.setWordWrap(True)
    layout.addWidget(label)
    detail = QtWidgets.QLabel(message, widget)
    detail.setWordWrap(True)
    layout.addWidget(detail)
    layout.addStretch(1)
    return widget
