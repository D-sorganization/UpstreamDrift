"""Reusable widgets for the Launch Monitor Analytics workbench."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6 import QtCore, QtWidgets

from src.shared.python.launch_monitor import (
    IDENTITY_COLUMNS,
    METRICS,
    PROFILES,
    ColumnMapping,
    ImportOptions,
    detect_profile,
)


class DataFrameTable(QtWidgets.QTableWidget):
    """Read-only, copyable tabular preview for pandas frames."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.setSortingEnabled(False)
        header = self.horizontalHeader()
        assert header is not None  # noqa: S101 - Qt creates the header
        header.setStretchLastSection(False)

    def set_frame(self, frame: pd.DataFrame, *, max_rows: int = 500) -> None:
        """Render at most ``max_rows`` while retaining all columns."""
        preview = frame.head(max_rows)
        self.clear()
        self.setRowCount(len(preview))
        self.setColumnCount(len(preview.columns))
        self.setHorizontalHeaderLabels([str(column) for column in preview.columns])
        for row_position, (_, row) in enumerate(preview.iterrows()):
            for column_position, value in enumerate(row):
                text = "" if pd.isna(value) else str(value)
                self.setItem(
                    row_position, column_position, QtWidgets.QTableWidgetItem(text)
                )
        self.resizeColumnsToContents()


def _read_headers(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, nrows=5, sep=None, engine="python")
    elif suffix in {".tsv", ".txt"}:
        frame = pd.read_csv(path, nrows=5, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, nrows=5)
    elif suffix == ".json":
        frame = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported file extension: {suffix}")
    return [str(column) for column in frame.columns]


class ImportMappingDialog(QtWidgets.QDialog):
    """Preview profile detection and edit column/unit mappings before import."""

    def __init__(self, source: Path, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.source = source
        self.headers = _read_headers(source)
        detection = detect_profile(self.headers)
        self.setWindowTitle("Review Launch Monitor Import")
        self.resize(1240, 600)

        self.profile_combo = QtWidgets.QComboBox()
        for profile_id, profile in PROFILES.items():
            self.profile_combo.addItem(f"{profile.vendor} ({profile_id})", profile_id)
        detected_index = self.profile_combo.findData(detection.profile_id)
        self.profile_combo.setCurrentIndex(max(0, detected_index))

        self.confidence_label = QtWidgets.QLabel(
            f"Detected {detection.profile_id} with {detection.confidence:.0%} "
            "header-fingerprint confidence. Review units before import."
        )
        self.confidence_label.setWordWrap(True)
        self.mapping_table = QtWidgets.QTableWidget(len(self.headers), 5)
        self.mapping_table.setHorizontalHeaderLabels(
            [
                "Source Column",
                "Canonical Target",
                "Source Unit",
                "Direction",
                "Measurement Status",
            ]
        )
        mapping_header = self.mapping_table.horizontalHeader()
        assert mapping_header is not None  # noqa: S101 - Qt creates the header
        mapping_header.setStretchLastSection(True)
        self.profile_combo.currentIndexChanged.connect(self._populate_mappings)
        self._populate_mappings()

        self.player_edit = QtWidgets.QLineEdit()
        self.session_edit = QtWidgets.QLineEdit(source.stem)
        self.model_edit = QtWidgets.QLineEdit()
        self.version_edit = QtWidgets.QLineEdit()
        metadata = QtWidgets.QFormLayout()
        metadata.addRow("Session Name:", self.session_edit)
        metadata.addRow("Player:", self.player_edit)
        metadata.addRow("Monitor Model:", self.model_edit)
        metadata.addRow("Software Version:", self.version_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"Source: {source}"))
        layout.addWidget(self.confidence_label)
        layout.addWidget(self.profile_combo)
        layout.addWidget(self.mapping_table, 1)
        layout.addLayout(metadata)
        layout.addWidget(buttons)

    def _populate_mappings(self) -> None:
        profile_id = str(self.profile_combo.currentData())
        auto = {
            item.source_column: item.target_column
            for item in PROFILES[profile_id].mappings_for(self.headers)
        }
        targets = ["(retain only)", *IDENTITY_COLUMNS, "date", "time", *METRICS]
        for row, header in enumerate(self.headers):
            source_item = QtWidgets.QTableWidgetItem(header)
            source_item.setFlags(
                source_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable
            )
            self.mapping_table.setItem(row, 0, source_item)
            target_combo = QtWidgets.QComboBox()
            target_combo.addItems(targets)
            target_combo.setCurrentText(auto.get(header, "(retain only)"))
            self.mapping_table.setCellWidget(row, 1, target_combo)
            unit_edit = QtWidgets.QLineEdit()
            unit_edit.setPlaceholderText("Infer from header/profile")
            self.mapping_table.setCellWidget(row, 2, unit_edit)
            direction_combo = QtWidgets.QComboBox()
            direction_combo.addItem("As Reported (+1)", 1.0)
            direction_combo.addItem("Invert Sign (-1)", -1.0)
            self.mapping_table.setCellWidget(row, 3, direction_combo)
            status_combo = QtWidgets.QComboBox()
            status_combo.addItems(
                ["reported", "measured", "estimated", "derived", "unknown"]
            )
            self.mapping_table.setCellWidget(row, 4, status_combo)
        self.mapping_table.resizeColumnsToContents()

    def import_options(self) -> ImportOptions:
        """Return the reviewed mapping configuration."""
        mappings: list[ColumnMapping] = []
        for row, header in enumerate(self.headers):
            target_widget = self.mapping_table.cellWidget(row, 1)
            unit_widget = self.mapping_table.cellWidget(row, 2)
            direction_widget = self.mapping_table.cellWidget(row, 3)
            status_widget = self.mapping_table.cellWidget(row, 4)
            if not isinstance(target_widget, QtWidgets.QComboBox):
                continue
            target = target_widget.currentText()
            if target == "(retain only)":
                continue
            unit = (
                unit_widget.text().strip()
                if isinstance(unit_widget, QtWidgets.QLineEdit)
                else ""
            )
            multiplier = (
                float(direction_widget.currentData())
                if isinstance(direction_widget, QtWidgets.QComboBox)
                else 1.0
            )
            status = (
                status_widget.currentText()
                if isinstance(status_widget, QtWidgets.QComboBox)
                else "reported"
            )
            mappings.append(
                ColumnMapping(header, target, unit or None, multiplier, status)
            )
        return ImportOptions(
            profile_id=str(self.profile_combo.currentData()),
            mappings=tuple(mappings),
            session_name=self.session_edit.text().strip() or self.source.stem,
            player=self.player_edit.text().strip() or None,
            monitor_model=self.model_edit.text().strip() or None,
            software_version=self.version_edit.text().strip() or None,
        )
