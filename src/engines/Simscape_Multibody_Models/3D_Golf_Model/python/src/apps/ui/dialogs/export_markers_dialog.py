"""Dialog for selective marker export (CSV / JSON / NPZ)."""

from __future__ import annotations

import re
from typing import Any

from PyQt6 import QtWidgets

from ...core.models import C3DDataModel

_CLUB_MARKER_RE = re.compile(r"^Marker_\d+:\d+:", re.IGNORECASE)


def _is_club_marker(name: str) -> bool:
    return bool(_CLUB_MARKER_RE.match(name))


class ExportMarkersDialog(QtWidgets.QDialog):
    """Modal export-options picker."""

    def __init__(
        self, model: C3DDataModel, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        if model is None:
            raise ValueError("model must be provided")
        self.model = model
        self.setWindowTitle("Export markers")
        self.resize(480, 540)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Marker selection"))

        select_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Select all")
        btn_body = QtWidgets.QPushButton("Body")
        btn_clear = QtWidgets.QPushButton("Clear")
        btn_all.clicked.connect(lambda: self._set_selection(lambda _n: True))
        btn_body.clicked.connect(self._select_body)
        btn_clear.clicked.connect(lambda: self._set_selection(lambda _n: False))
        for b in (btn_all, btn_body, btn_clear):
            select_row.addWidget(b)
        select_row.addStretch()
        layout.addLayout(select_row)

        self.list_markers = QtWidgets.QListWidget()
        self.list_markers.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.MultiSelection
        )
        for name in self.model.marker_names():
            self.list_markers.addItem(name)
        self.list_markers.selectAll()
        layout.addWidget(self.list_markers, 1)

        # Components.
        comp_box = QtWidgets.QGroupBox("Components")
        comp_layout = QtWidgets.QHBoxLayout(comp_box)
        self.radio_x = QtWidgets.QRadioButton("X")
        self.radio_y = QtWidgets.QRadioButton("Y")
        self.radio_z = QtWidgets.QRadioButton("Z")
        self.radio_all = QtWidgets.QRadioButton("All (X, Y, Z)")
        self.radio_all.setChecked(True)
        for r in (self.radio_x, self.radio_y, self.radio_z, self.radio_all):
            comp_layout.addWidget(r)
        layout.addWidget(comp_box)

        # Frame range.
        n_frames = (
            len(self.model.point_time) if self.model.point_time is not None else 0
        )
        range_box = QtWidgets.QGroupBox("Frame range")
        range_layout = QtWidgets.QHBoxLayout(range_box)
        self.spin_start = QtWidgets.QSpinBox()
        self.spin_end = QtWidgets.QSpinBox()
        self.spin_start.setRange(0, max(0, n_frames - 1))
        self.spin_end.setRange(0, max(0, n_frames - 1))
        self.spin_end.setValue(max(0, n_frames - 1))
        range_layout.addWidget(QtWidgets.QLabel("Start"))
        range_layout.addWidget(self.spin_start)
        range_layout.addWidget(QtWidgets.QLabel("End"))
        range_layout.addWidget(self.spin_end)
        layout.addWidget(range_box)

        # Format.
        fmt_box = QtWidgets.QGroupBox("Format")
        fmt_layout = QtWidgets.QHBoxLayout(fmt_box)
        self.radio_csv = QtWidgets.QRadioButton("CSV")
        self.radio_json = QtWidgets.QRadioButton("JSON")
        self.radio_npz = QtWidgets.QRadioButton("NPZ")
        self.radio_csv.setChecked(True)
        for r in (self.radio_csv, self.radio_json, self.radio_npz):
            fmt_layout.addWidget(r)
        layout.addWidget(fmt_box)

        # Options.
        self.check_time = QtWidgets.QCheckBox("Include time column")
        self.check_time.setChecked(True)
        self.check_residual = QtWidgets.QCheckBox("Include residuals")
        layout.addWidget(self.check_time)
        layout.addWidget(self.check_residual)

        # Buttons.
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        ok_btn = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if ok_btn is not None:
            ok_btn.setText("Export…")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._chosen_path: str | None = None

    # -------------------------------------------------------------- helpers

    def _set_selection(self, predicate: Any) -> None:
        self.list_markers.blockSignals(True)
        try:
            for i in range(self.list_markers.count()):
                item = self.list_markers.item(i)
                if item is None:
                    continue
                item.setSelected(bool(predicate(item.text())))
        finally:
            self.list_markers.blockSignals(False)

    def _select_body(self) -> None:
        try:
            from src.shared.python.motion_matching.body_skeleton import (  # type: ignore
                default_body_segments,
            )
        except ImportError:  # pragma: no cover
            from shared.python.motion_matching.body_skeleton import (  # type: ignore
                default_body_segments,
            )
        body = {
            n
            for s in default_body_segments(self.model.marker_names())
            for n in (s.a, s.b)
        }
        self._set_selection(lambda n: n in body)

    def _selected_markers(self) -> list[str]:
        return [item.text() for item in self.list_markers.selectedItems()]

    def _selected_format(self) -> str:
        if self.radio_json.isChecked():
            return "json"
        if self.radio_npz.isChecked():
            return "npz"
        return "csv"

    def _selected_components(self) -> tuple[str, ...]:
        if self.radio_x.isChecked():
            return ("x",)
        if self.radio_y.isChecked():
            return ("y",)
        if self.radio_z.isChecked():
            return ("z",)
        return ("x", "y", "z")

    def _default_extension(self) -> str:
        return {"csv": "csv", "json": "json", "npz": "npz"}[self._selected_format()]

    def _on_accept(self) -> None:
        if not self._selected_markers():
            QtWidgets.QMessageBox.information(
                self, "Export markers", "Please select at least one marker."
            )
            return
        ext = self._default_extension()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export markers",
            f"markers.{ext}",
            f"{ext.upper()} files (*.{ext})",
        )
        if not path:
            return
        self._chosen_path = path
        self.accept()

    def export_params(self) -> dict[str, Any] | None:
        """Return the user's chosen export parameters."""
        if self._chosen_path is None:
            return None
        return {
            "marker_names": self._selected_markers(),
            "components": self._selected_components(),
            "frame_range": (
                int(self.spin_start.value()),
                int(self.spin_end.value()),
            ),
            "fmt": self._selected_format(),
            "path": self._chosen_path,
            "include_time": bool(self.check_time.isChecked()),
            "include_residual": bool(self.check_residual.isChecked()),
        }


__all__ = ["ExportMarkersDialog"]
