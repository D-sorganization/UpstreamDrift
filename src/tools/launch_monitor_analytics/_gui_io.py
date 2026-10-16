"""Import, project, export and filter handlers for the workbench."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PyQt6 import QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.launch_monitor_analytics.reporting import (
    build_data_export_record,
    build_reproducibility_manifest,
    format_canonical_csv_export,
)
from src.tools.launch_monitor_analytics.widgets import (
    ImportMappingDialog,
)
from src.tools.launch_monitor_model import (
    FilterRule,
)

logger = get_logger(__name__)


class _ImportExportMixin:
    """File import/export and filter-row handlers for ``MainWidget``."""

    _last_data_export: dict[str, str] | None

    if TYPE_CHECKING:
        # Host attributes/methods live on ``MainWidget`` (composed mixins).
        def __getattr__(self, name: str) -> Any: ...

    def _host_widget(self) -> QtWidgets.QWidget:
        """Return this mixin's host ``MainWidget`` typed as a ``QWidget``."""
        return cast("QtWidgets.QWidget", self)

    def _on_import_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self._host_widget(),
            "Import Launch Monitor Exports",
            "",
            "Data Files (*.csv *.tsv *.txt *.xlsx *.xls *.json);;All Files (*)",
        )
        for raw_path in paths:
            path = Path(raw_path)
            try:
                dialog = ImportMappingDialog(path, self._host_widget())
                if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                    continue
                self.import_file(path, dialog.import_options())
            except (ValueError, OSError) as exc:
                self._show_error(f"Could Not Import {path.name}", exc)

    def _on_open_project(self) -> None:
        if not self._confirm_discard_unsaved("opening another project"):
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._host_widget(),
            "Open Launch Monitor Project",
            "",
            "Launch Monitor Projects (*.lmproject);;All Files (*)",
        )
        if path:
            try:
                self.load_project(path)
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self._show_error("Could Not Open Project", exc)

    def _on_save_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._host_widget(),
            "Save Launch Monitor Project",
            "study.lmproject",
            "Launch Monitor Projects (*.lmproject)",
        )
        if path:
            try:
                self.save_project(path)
            except OSError as exc:
                self._show_error("Could Not Save Project", exc)

    def export_data(self, path: str | Path, *, as_parquet: bool = False) -> Path:
        """Export the analysis view, stamping a shared export ID + hash.

        Mirrors the ``ImportManifest`` provenance pattern: a fresh export
        ID/timestamp is embedded in the artifact (a leading ``#`` comment
        row for CSV, Parquet schema metadata for Parquet) and recorded on
        ``self._last_data_export`` with the file's own SHA-256, so a later
        ``export_manifest`` call can correlate the two on disk.
        """
        destination = Path(path)
        export_id = uuid.uuid4().hex
        exported_at = datetime.now(UTC).isoformat()
        is_parquet = as_parquet or destination.suffix.lower() == ".parquet"
        if is_parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(self.analysis_frame, preserve_index=False)
            metadata = dict(table.schema.metadata or {})
            metadata[b"export_id"] = export_id.encode("utf-8")
            metadata[b"exported_at"] = exported_at.encode("utf-8")
            table = table.replace_schema_metadata(metadata)
            pq.write_table(table, destination)
        else:
            csv_text = format_canonical_csv_export(
                self.analysis_frame, export_id=export_id, exported_at=exported_at
            )
            destination.write_text(csv_text, encoding="utf-8", newline="")
        self._last_data_export = build_data_export_record(
            export_id=export_id,
            exported_at=exported_at,
            data_file=destination.name,
            data_bytes=destination.read_bytes(),
        )
        return destination

    def _on_export_data(self) -> None:
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self._host_widget(),
            "Export Canonical Analysis View",
            "launch_monitor_data.csv",
            "CSV (*.csv);;Parquet (*.parquet)",
        )
        if not path:
            return
        try:
            self.export_data(path, as_parquet="Parquet" in selected_filter)
        except (OSError, ValueError, ImportError) as exc:
            self._show_error("Could Not Export Data", exc)
            return
        export_id = (self._last_data_export or {}).get("export_id", "")
        self.status_label.setText(f"Exported analysis view to {path} (ID {export_id}).")

    def export_manifest(self, path: str | Path) -> Path:
        """Write the reproducibility manifest, correlated to the last export.

        Embeds the last ``export_data`` call's ID, filename, and SHA-256
        under ``data_export`` (``None`` if no export ran yet) so the pair
        stays verifiably linked even once separated on disk.
        """
        destination = Path(path)
        payload = build_reproducibility_manifest(
            project_name=self.project.name,
            sessions=[asdict(session.manifest) for session in self.project.sessions],
            treatment_audit_log=self.project.audit_log,
            frame=self.analysis_frame,
            scientific_boundary=self.scientific_boundary.text(),
            data_export=self._last_data_export,
        )
        destination.write_text(json.dumps(payload, indent=2, default=str), "utf-8")
        return destination

    def _on_export_manifest(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self._host_widget(),
            "Export Reproducibility Manifest",
            "launch_monitor_manifest.json",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            self.export_manifest(path)
        except OSError as exc:
            self._show_error("Could Not Export Manifest", exc)

    def _show_error_message(self, title: str, message: str) -> None:
        """Surface an analysis failure in the log and the status line."""
        logger.warning("%s: %s", title, message)
        self.status_label.setText(f"{title}: {message}")

    def _show_error(self, title: str, exc: Exception) -> None:
        """Surface a synchronous failure with a modal warning."""
        logger.warning("%s: %s", title, exc)
        self.status_label.setText(f"{title}: {exc}")
        QtWidgets.QMessageBox.warning(self._host_widget(), title, str(exc))

    def _add_filter_row(self) -> None:
        row = self.filter_table.rowCount()
        self.filter_table.insertRow(row)
        column_combo = QtWidgets.QComboBox()
        column_combo.addItems([str(column) for column in self.analysis_frame.columns])
        operator_combo = QtWidgets.QComboBox()
        operator_combo.addItems(["eq", "ne", "lt", "le", "gt", "ge", "contains", "in"])
        self.filter_table.setCellWidget(row, 0, column_combo)
        self.filter_table.setCellWidget(row, 1, operator_combo)
        self.filter_table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))

    def _remove_filter_rows(self) -> None:
        rows = sorted(
            {index.row() for index in self.filter_table.selectedIndexes()}, reverse=True
        )
        for row in rows:
            self.filter_table.removeRow(row)

    def _filter_rules(self) -> tuple[FilterRule, ...]:
        rules: list[FilterRule] = []
        for row in range(self.filter_table.rowCount()):
            column_widget = self.filter_table.cellWidget(row, 0)
            operator_widget = self.filter_table.cellWidget(row, 1)
            value_item = self.filter_table.item(row, 2)
            if not isinstance(column_widget, QtWidgets.QComboBox) or not isinstance(
                operator_widget, QtWidgets.QComboBox
            ):
                continue
            rules.append(
                FilterRule(
                    column_widget.currentText(),
                    operator_widget.currentText(),
                    value_item.text() if value_item is not None else "",
                )
            )
        return tuple(rules)
