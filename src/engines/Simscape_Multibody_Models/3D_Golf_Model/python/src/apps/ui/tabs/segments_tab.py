"""Segments tab — interactively edit user-defined marker-pair segments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6 import QtCore, QtWidgets

from ...core.models import C3DDataModel
from ...services.segment_set_io import (
    DEFAULT_RADIUS_M,
    SegmentSet,
    SegmentSpec,
    default_segment_set_path,
    load_segment_set,
    save_segment_set,
)

try:
    from src.shared.python.motion_matching.body_skeleton import (
        default_body_segments,
    )
except ImportError:  # pragma: no cover
    from shared.python.motion_matching.body_skeleton import (  # type: ignore
        default_body_segments,
    )

_GEOMETRIES = ("line", "cylinder")
_COL_VISIBLE = 0
_COL_A = 1
_COL_B = 2
_COL_GEOMETRY = 3
_COL_GROUP = 4
_COL_DELETE = 5


class _AddSegmentDialog(QtWidgets.QDialog):
    """Modal dialog to add a new segment."""

    def __init__(
        self,
        marker_names: list[str],
        last_group: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add segment")
        layout = QtWidgets.QFormLayout(self)
        self.combo_a = QtWidgets.QComboBox()
        self.combo_b = QtWidgets.QComboBox()
        self.combo_a.addItems(marker_names)
        self.combo_b.addItems(marker_names)
        if len(marker_names) >= 2:
            self.combo_b.setCurrentIndex(1)
        self.combo_geometry = QtWidgets.QComboBox()
        self.combo_geometry.addItems(_GEOMETRIES)
        self.edit_group = QtWidgets.QLineEdit(last_group or "auto")
        layout.addRow("Marker A", self.combo_a)
        layout.addRow("Marker B", self.combo_b)
        layout.addRow("Geometry", self.combo_geometry)
        layout.addRow("Group", self.edit_group)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def selected_spec(self) -> SegmentSpec | None:
        a = self.combo_a.currentText()
        b = self.combo_b.currentText()
        if not a or not b or a == b:
            return None
        return SegmentSpec(
            a=a,
            b=b,
            geometry=self.combo_geometry.currentText(),
            group=self.edit_group.text().strip() or "auto",
            visible=True,
            radius=DEFAULT_RADIUS_M,
        )


class SegmentsTab(QtWidgets.QWidget):
    """Editor for user-defined segments overlaid on the 3D scene."""

    segments_changed = QtCore.pyqtSignal(tuple)  # tuple[SegmentSpec, ...]

    def __init__(self) -> None:
        super().__init__()
        self.model: C3DDataModel | None = None
        self._marker_names: list[str] = []
        self._segments: list[SegmentSpec] = []
        self._last_group: str = "auto"
        self._suspend_signals: bool = False
        self._init_ui()

    # -------------------------------------------------------------------- UI

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        top_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("+ Add segment")
        self.btn_reset = QtWidgets.QPushButton("Reset to default")
        self.btn_add.clicked.connect(self._on_add_clicked)
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        top_row.addWidget(self.btn_add)
        top_row.addWidget(self.btn_reset)
        top_row.addStretch()
        layout.addLayout(top_row)

        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Visible", "Marker A", "Marker B", "Geometry", "Group", ""]
        )
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(
                _COL_GROUP, QtWidgets.QHeaderView.ResizeMode.Stretch
            )
        layout.addWidget(self.table)

        bottom_row = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save segment set…")
        self.btn_load = QtWidgets.QPushButton("Load segment set…")
        self.btn_export = QtWidgets.QPushButton("Export to JSON")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_export.clicked.connect(self._on_export_clicked)
        bottom_row.addWidget(self.btn_save)
        bottom_row.addWidget(self.btn_load)
        bottom_row.addWidget(self.btn_export)
        bottom_row.addStretch()
        layout.addLayout(bottom_row)

    # ------------------------------------------------------------- Public API

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Repopulate when a new model is loaded."""
        self.model = model
        self._marker_names = list(model.marker_names()) if model is not None else []
        if model is None:
            self._set_segments([])
            return
        defaults = tuple(
            SegmentSpec(a=s.a, b=s.b, geometry="line", group=s.group)
            for s in default_body_segments(self._marker_names)
        )
        self._set_segments(list(defaults))

    @property
    def segments(self) -> tuple[SegmentSpec, ...]:
        """Current user segment list."""
        return tuple(self._segments)

    def add_segment(self, spec: SegmentSpec) -> None:
        """Programmatically append a segment (test-friendly)."""
        if not isinstance(spec, SegmentSpec):
            raise TypeError(f"spec must be SegmentSpec, got {type(spec).__name__}")
        self._segments.append(spec)
        self._last_group = spec.group
        self._rebuild_table()
        self._emit_changed()

    def set_segment_geometry(self, index: int, geometry: str) -> None:
        """Programmatically swap the geometry of segment ``index``."""
        if not 0 <= index < len(self._segments):
            raise ValueError(f"index {index} out of range [0, {len(self._segments)})")
        if geometry not in _GEOMETRIES:
            raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
        old = self._segments[index]
        self._segments[index] = SegmentSpec(
            a=old.a,
            b=old.b,
            geometry=geometry,
            group=old.group,
            visible=old.visible,
            radius=old.radius,
        )
        self._rebuild_table()
        self._emit_changed()

    def set_segment_visibility(self, index: int, visible: bool) -> None:
        """Programmatically toggle visibility of segment ``index``."""
        if not 0 <= index < len(self._segments):
            raise ValueError(f"index {index} out of range [0, {len(self._segments)})")
        old = self._segments[index]
        self._segments[index] = SegmentSpec(
            a=old.a,
            b=old.b,
            geometry=old.geometry,
            group=old.group,
            visible=bool(visible),
            radius=old.radius,
        )
        self._rebuild_table()
        self._emit_changed()

    # -------------------------------------------------------------- Internal

    def _set_segments(self, segments: list[SegmentSpec]) -> None:
        self._segments = list(segments)
        self._rebuild_table()
        self._emit_changed()

    def _emit_changed(self) -> None:
        if self._suspend_signals:
            return
        self.segments_changed.emit(tuple(self._segments))

    def _rebuild_table(self) -> None:
        self._suspend_signals = True
        try:
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            for row, spec in enumerate(self._segments):
                self.table.insertRow(row)
                self._populate_row(row, spec)
        finally:
            self.table.blockSignals(False)
            self._suspend_signals = False

    def _populate_row(self, row: int, spec: SegmentSpec) -> None:
        # Visible checkbox
        chk = QtWidgets.QCheckBox()
        chk.setChecked(spec.visible)
        chk.toggled.connect(lambda v, r=row: self._on_visible_toggled(r, v))
        wrapper = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrapper)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.addWidget(chk)
        self.table.setCellWidget(row, _COL_VISIBLE, wrapper)

        combo_a = QtWidgets.QComboBox()
        combo_b = QtWidgets.QComboBox()
        for combo, current in ((combo_a, spec.a), (combo_b, spec.b)):
            combo.addItems(self._marker_names)
            if current in self._marker_names:
                combo.setCurrentText(current)
            else:
                combo.insertItem(0, current)
                combo.setCurrentIndex(0)
        combo_a.currentTextChanged.connect(
            lambda val, r=row: self._on_endpoint_changed(r, "a", val)
        )
        combo_b.currentTextChanged.connect(
            lambda val, r=row: self._on_endpoint_changed(r, "b", val)
        )
        self.table.setCellWidget(row, _COL_A, combo_a)
        self.table.setCellWidget(row, _COL_B, combo_b)

        combo_geom = QtWidgets.QComboBox()
        combo_geom.addItems(_GEOMETRIES)
        combo_geom.setCurrentText(spec.geometry)
        combo_geom.currentTextChanged.connect(
            lambda val, r=row: self._on_geometry_changed(r, val)
        )
        self.table.setCellWidget(row, _COL_GEOMETRY, combo_geom)

        edit_group = QtWidgets.QLineEdit(spec.group)
        edit_group.editingFinished.connect(
            lambda r=row, w=edit_group: self._on_group_changed(r, w.text())
        )
        self.table.setCellWidget(row, _COL_GROUP, edit_group)

        btn_del = QtWidgets.QPushButton("×")
        btn_del.setMaximumWidth(28)
        btn_del.clicked.connect(lambda _checked=False, r=row: self._delete_row(r))
        self.table.setCellWidget(row, _COL_DELETE, btn_del)

    # ---------------------------------------------------------- Row callbacks

    def _replace(self, index: int, **kwargs: Any) -> None:
        old = self._segments[index]
        merged = {
            "a": kwargs.get("a", old.a),
            "b": kwargs.get("b", old.b),
            "geometry": kwargs.get("geometry", old.geometry),
            "group": kwargs.get("group", old.group),
            "visible": kwargs.get("visible", old.visible),
            "radius": kwargs.get("radius", old.radius),
        }
        try:
            self._segments[index] = SegmentSpec(**merged)
        except ValueError:
            # Reject the change silently — table will rebuild from old state.
            self._rebuild_table()
            return
        self._emit_changed()

    def _on_visible_toggled(self, row: int, value: bool) -> None:
        if self._suspend_signals or not 0 <= row < len(self._segments):
            return
        self._replace(row, visible=value)

    def _on_endpoint_changed(self, row: int, which: str, value: str) -> None:
        if self._suspend_signals or not 0 <= row < len(self._segments):
            return
        self._replace(row, **{which: value})

    def _on_geometry_changed(self, row: int, value: str) -> None:
        if self._suspend_signals or not 0 <= row < len(self._segments):
            return
        self._replace(row, geometry=value)

    def _on_group_changed(self, row: int, value: str) -> None:
        if self._suspend_signals or not 0 <= row < len(self._segments):
            return
        text = value.strip() or "auto"
        self._last_group = text
        self._replace(row, group=text)

    def _delete_row(self, row: int) -> None:
        if not 0 <= row < len(self._segments):
            return
        del self._segments[row]
        self._rebuild_table()
        self._emit_changed()

    # ---------------------------------------------------------- Top buttons

    def _on_add_clicked(self) -> None:
        if not self._marker_names:
            QtWidgets.QMessageBox.information(
                self, "Add segment", "No markers available. Load a C3D file first."
            )
            return
        dlg = _AddSegmentDialog(self._marker_names, self._last_group, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            spec = dlg.selected_spec()
            if spec is None:
                QtWidgets.QMessageBox.warning(
                    self, "Add segment", "Marker A and Marker B must differ."
                )
                return
            self.add_segment(spec)

    def _on_reset_clicked(self) -> None:
        if self.model is None:
            return
        defaults = [
            SegmentSpec(a=s.a, b=s.b, geometry="line", group=s.group)
            for s in default_body_segments(self._marker_names)
        ]
        self._set_segments(defaults)

    # ----------------------------------------------------------- Save / Load

    def _on_save_clicked(self) -> None:
        default = default_segment_set_path()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save segment set", str(default), "JSON files (*.json)"
        )
        if path:
            save_segment_set(path, SegmentSet(segments=tuple(self._segments)))

    def _on_load_clicked(self) -> None:
        default = default_segment_set_path()
        start_dir = str(default.parent) if default.parent.exists() else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load segment set", start_dir, "JSON files (*.json)"
        )
        if path:
            try:
                segset = load_segment_set(path)
            except (OSError, ValueError) as e:
                QtWidgets.QMessageBox.warning(
                    self, "Load segment set", f"Could not load:\n{e}"
                )
                return
            self._set_segments(list(segset.segments))

    def _on_export_clicked(self) -> None:
        # Same as save, but force a clear default filename.
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export segment set to JSON",
            "segments.json",
            "JSON files (*.json)",
        )
        if path:
            save_segment_set(path, SegmentSet(segments=tuple(self._segments)))


__all__ = ["SegmentsTab"]
