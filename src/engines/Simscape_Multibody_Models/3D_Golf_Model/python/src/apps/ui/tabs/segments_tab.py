"""Segments tab — interactively edit user-defined marker-pair segments.

Wave 4 of EPIC #4755: the tab now stores its segments as canonical v2
:class:`SegmentVizSpec` instances. The Shape column exposes six options
(Line, Cylinder, Ellipsoid, Capsule, Library shape…, Mesh file…) and two
new buttons let the user import a mesh file or pick from the bundled
shape library.

Back-compat: the legacy v1 :class:`SegmentSpec` API surface
(``add_segment``, ``set_segment_geometry``, ``set_segment_visibility``,
``segments`` property, ``segments_changed`` signal) is preserved as a
filtered v1 view over the v2 store; v2-only shape kinds are dropped from
that view but survive in the underlying store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6 import QtCore, QtWidgets

from src.shared.python.body_part_viz import (
    BindingKind,
    MarkerBinding,
    SegmentVizSet,
    SegmentVizSpec,
    ShapeTheme,
)
from src.shared.python.body_part_viz.asset_library import ShapeLibrary

from ...core.models import C3DDataModel
from ...services.segment_set_io import (
    DEFAULT_RADIUS_M,
    SegmentSpec,
    default_segment_set_path,
    spec_v1_to_v2,
    spec_v2_to_v1,
)

try:
    from src.shared.python.motion_matching.body_skeleton import (
        default_body_segments,
    )
except ImportError:  # pragma: no cover
    from shared.python.motion_matching.body_skeleton import (  # type: ignore
        default_body_segments,
    )

# Shape-picker labels exposed in the table combobox. The first four map
# directly onto v2 shape kinds; the last two are sentinels that open a
# modal chooser instead of selecting a kind in place.
_SHAPE_LABEL_LINE = "Line"
_SHAPE_LABEL_CYLINDER = "Cylinder"
_SHAPE_LABEL_ELLIPSOID = "Ellipsoid"
_SHAPE_LABEL_CAPSULE = "Capsule"
_SHAPE_LABEL_LIBRARY = "Library shape…"
_SHAPE_LABEL_MESH = "Mesh file…"

_SHAPE_LABELS: tuple[str, ...] = (
    _SHAPE_LABEL_LINE,
    _SHAPE_LABEL_CYLINDER,
    _SHAPE_LABEL_ELLIPSOID,
    _SHAPE_LABEL_CAPSULE,
    _SHAPE_LABEL_LIBRARY,
    _SHAPE_LABEL_MESH,
)

_LABEL_TO_KIND = {
    _SHAPE_LABEL_LINE: "line",
    _SHAPE_LABEL_CYLINDER: "cylinder",
    _SHAPE_LABEL_ELLIPSOID: "ellipsoid",
    _SHAPE_LABEL_CAPSULE: "capsule",
}

_MESH_FILTER = (
    "Mesh files (*.stl *.obj *.ply *.glb);;"
    "STL (*.stl);;OBJ (*.obj);;PLY (*.ply);;GLB (*.glb);;All files (*)"
)

_GEOMETRIES = ("line", "cylinder")  # legacy v1 add-dialog options
_COL_VISIBLE = 0
_COL_A = 1
_COL_B = 2
_COL_SHAPE = 3
_COL_GROUP = 4
_COL_DELETE = 5


def _default_shape_params(shape_kind: str) -> dict[str, Any]:
    """Return a sensible default ``shape_params`` for ``shape_kind``."""
    if shape_kind == "line":
        return {"length": 1.0, "radius": DEFAULT_RADIUS_M}
    if shape_kind == "cylinder":
        return {
            "length": 1.0,
            "radius": DEFAULT_RADIUS_M,
            "n_facets": 16,
        }
    if shape_kind == "ellipsoid":
        return {"a": 0.05, "b": 0.05, "c": 0.05}
    if shape_kind == "capsule":
        return {
            "length": 1.0,
            "radius": DEFAULT_RADIUS_M,
            "n_facets": 16,
            "n_lat": 8,
        }
    raise ValueError(f"no default shape_params for shape_kind={shape_kind!r}")


def _shape_label_for_spec(spec: SegmentVizSpec) -> str:
    """Return the picker label that ``spec``'s ``shape_kind`` maps to."""
    kind = spec.shape_kind
    if kind in _LABEL_TO_KIND.values():
        for label, k in _LABEL_TO_KIND.items():
            if k == kind:
                return label
    if kind == "library_shape":
        return _SHAPE_LABEL_LIBRARY
    if kind == "mesh_file":
        return _SHAPE_LABEL_MESH
    return _SHAPE_LABEL_LINE  # safe default


class _AddSegmentDialog(QtWidgets.QDialog):
    """Modal dialog to add a new segment (v1-style line/cylinder)."""

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


class _LibraryShapeDialog(QtWidgets.QDialog):
    """Modal chooser for the bundled :class:`ShapeLibrary` entries."""

    def __init__(
        self,
        names: tuple[str, ...],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick a library shape")
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Library shape:"))
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.addItems(list(names))
        if names:
            self.list_widget.setCurrentRow(0)
        self.list_widget.itemDoubleClicked.connect(lambda *_: self.accept())
        layout.addWidget(self.list_widget)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_name(self) -> str | None:
        item = self.list_widget.currentItem()
        return item.text() if item is not None else None


class SegmentsTab(QtWidgets.QWidget):
    """Editor for user-defined segments overlaid on the 3D scene.

    Stores a list of :class:`SegmentVizSpec` internally; exposes both the
    v2 surface (``viz_segments``, ``viz_segments_changed``) and the v1
    surface (``segments``, ``segments_changed``) for back-compat.
    """

    # v1 signal: tuple[SegmentSpec, ...] — only the v1-shaped subset.
    segments_changed = QtCore.pyqtSignal(tuple)
    # v2 signal: tuple[SegmentVizSpec, ...] — the full canonical store.
    viz_segments_changed = QtCore.pyqtSignal(tuple)

    def __init__(self) -> None:
        super().__init__()
        self.model: C3DDataModel | None = None
        self._marker_names: list[str] = []
        self._viz_segments: list[SegmentVizSpec] = []
        self._last_group: str = "auto"
        self._suspend_signals: bool = False
        self._shape_library: ShapeLibrary | None = None
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
            ["Visible", "Marker A", "Marker B", "Shape", "Group", ""]
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
        self.btn_import_mesh = QtWidgets.QPushButton("Import shape file…")
        self.btn_library = QtWidgets.QPushButton("Library…")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_load.clicked.connect(self._on_load_clicked)
        self.btn_import_mesh.clicked.connect(self._on_import_mesh_clicked)
        self.btn_library.clicked.connect(self._on_library_clicked)
        bottom_row.addWidget(self.btn_save)
        bottom_row.addWidget(self.btn_load)
        bottom_row.addWidget(self.btn_import_mesh)
        bottom_row.addWidget(self.btn_library)
        bottom_row.addStretch()
        layout.addLayout(bottom_row)

    # ------------------------------------------------------------- Public API

    def update_from_model(self, model: C3DDataModel | None) -> None:
        """Repopulate when a new model is loaded."""
        self.model = model
        self._marker_names = list(model.marker_names()) if model is not None else []
        if model is None:
            self._set_viz_segments([])
            return
        defaults = [
            self._make_default_line_spec(s.a, s.b, s.group)
            for s in default_body_segments(self._marker_names)
        ]
        self._set_viz_segments(defaults)

    @property
    def segments(self) -> tuple[SegmentSpec, ...]:
        """Current user segment list as legacy v1 specs (filtered view)."""
        return tuple(
            spec for spec in (spec_v2_to_v1(s) for s in self._viz_segments) if spec
        )

    @property
    def viz_segments(self) -> tuple[SegmentVizSpec, ...]:
        """Current user segment list as canonical v2 specs."""
        return tuple(self._viz_segments)

    def add_segment(self, spec: SegmentSpec) -> None:
        """Programmatically append a v1 segment (back-compat)."""
        if not isinstance(spec, SegmentSpec):
            raise TypeError(f"spec must be SegmentSpec, got {type(spec).__name__}")
        self._viz_segments.append(spec_v1_to_v2(spec))
        self._last_group = spec.group
        self._rebuild_table()
        self._emit_changed()

    def add_viz_segment(self, spec: SegmentVizSpec) -> None:
        """Programmatically append a v2 segment."""
        if not isinstance(spec, SegmentVizSpec):
            raise TypeError(f"spec must be SegmentVizSpec, got {type(spec).__name__}")
        self._viz_segments.append(spec)
        self._last_group = spec.theme.group
        self._rebuild_table()
        self._emit_changed()

    def set_segment_geometry(self, index: int, geometry: str) -> None:
        """Programmatically swap the geometry of segment ``index`` (v1 names)."""
        if not 0 <= index < len(self._viz_segments):
            raise ValueError(
                f"index {index} out of range [0, {len(self._viz_segments)})"
            )
        if geometry not in _GEOMETRIES:
            raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {geometry!r}")
        self._swap_shape_kind(index, geometry)

    def set_segment_visibility(self, index: int, visible: bool) -> None:
        """Programmatically toggle visibility of segment ``index``."""
        if not 0 <= index < len(self._viz_segments):
            raise ValueError(
                f"index {index} out of range [0, {len(self._viz_segments)})"
            )
        old = self._viz_segments[index]
        self._viz_segments[index] = SegmentVizSpec(
            binding=old.binding,
            shape_kind=old.shape_kind,
            shape_params=dict(old.shape_params),
            fitter_kind=old.fitter_kind,
            theme=old.theme,
            visible=bool(visible),
        )
        self._rebuild_table()
        self._emit_changed()

    # -------------------------------------------------------------- Internal

    def _make_default_line_spec(self, a: str, b: str, group: str) -> SegmentVizSpec:
        return SegmentVizSpec(
            binding=MarkerBinding(
                kind=BindingKind.BETWEEN_TWO,
                marker_names=(a, b),
            ),
            shape_kind="line",
            shape_params=_default_shape_params("line"),
            fitter_kind="between_two",
            theme=ShapeTheme(group=group or "auto"),
            visible=True,
        )

    def _set_viz_segments(self, segments: list[SegmentVizSpec]) -> None:
        self._viz_segments = list(segments)
        self._rebuild_table()
        self._emit_changed()

    def _emit_changed(self) -> None:
        if self._suspend_signals:
            return
        self.segments_changed.emit(self.segments)
        self.viz_segments_changed.emit(tuple(self._viz_segments))

    def _swap_shape_kind(self, index: int, shape_kind: str) -> None:
        old = self._viz_segments[index]
        if old.shape_kind == shape_kind:
            return
        # Preserve binding / theme / visibility / fitter; rebuild params.
        new_params = _default_shape_params(shape_kind)
        # Carry over numeric overlap (length, radius) where it makes sense.
        for key in ("length", "radius"):
            if key in old.shape_params and key in new_params:
                new_params[key] = old.shape_params[key]
        self._viz_segments[index] = SegmentVizSpec(
            binding=old.binding,
            shape_kind=shape_kind,
            shape_params=new_params,
            fitter_kind=old.fitter_kind,
            theme=old.theme,
            visible=old.visible,
        )
        self._rebuild_table()
        self._emit_changed()

    def _rebuild_table(self) -> None:
        self._suspend_signals = True
        try:
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            for row, spec in enumerate(self._viz_segments):
                self.table.insertRow(row)
                self._populate_row(row, spec)
        finally:
            self.table.blockSignals(False)
            self._suspend_signals = False

    def _populate_row(self, row: int, spec: SegmentVizSpec) -> None:
        # Visible checkbox
        chk = QtWidgets.QCheckBox()
        chk.setChecked(spec.visible)
        chk.toggled.connect(lambda v, r=row: self._on_visible_toggled(r, v))
        wrapper = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrapper)
        lay.setContentsMargins(4, 0, 4, 0)
        lay.addWidget(chk)
        self.table.setCellWidget(row, _COL_VISIBLE, wrapper)

        # Marker A / B come from the binding marker_names. For non-pairwise
        # bindings (cluster), we show the first two for a stable display
        # but the editing path keeps them read-only.
        marker_a = spec.binding.marker_names[0] if spec.binding.marker_names else ""
        marker_b = (
            spec.binding.marker_names[1] if len(spec.binding.marker_names) >= 2 else ""
        )

        combo_a = QtWidgets.QComboBox()
        combo_b = QtWidgets.QComboBox()
        for combo, current in ((combo_a, marker_a), (combo_b, marker_b)):
            combo.addItems(self._marker_names)
            if current and current in self._marker_names:
                combo.setCurrentText(current)
            elif current:
                combo.insertItem(0, current)
                combo.setCurrentIndex(0)
        combo_a.currentTextChanged.connect(
            lambda val, r=row: self._on_endpoint_changed(r, 0, val)
        )
        combo_b.currentTextChanged.connect(
            lambda val, r=row: self._on_endpoint_changed(r, 1, val)
        )
        self.table.setCellWidget(row, _COL_A, combo_a)
        self.table.setCellWidget(row, _COL_B, combo_b)

        combo_shape = QtWidgets.QComboBox()
        combo_shape.addItems(list(_SHAPE_LABELS))
        combo_shape.setCurrentText(_shape_label_for_spec(spec))
        combo_shape.activated.connect(
            lambda _idx, r=row, w=combo_shape: self._on_shape_picker(r, w.currentText())
        )
        self.table.setCellWidget(row, _COL_SHAPE, combo_shape)

        edit_group = QtWidgets.QLineEdit(spec.theme.group)
        edit_group.editingFinished.connect(
            lambda r=row, w=edit_group: self._on_group_changed(r, w.text())
        )
        self.table.setCellWidget(row, _COL_GROUP, edit_group)

        btn_del = QtWidgets.QPushButton("×")
        btn_del.setMaximumWidth(28)
        btn_del.clicked.connect(lambda _checked=False, r=row: self._delete_row(r))
        self.table.setCellWidget(row, _COL_DELETE, btn_del)

    # ---------------------------------------------------------- Row callbacks

    def _on_visible_toggled(self, row: int, value: bool) -> None:
        if self._suspend_signals or not 0 <= row < len(self._viz_segments):
            return
        self.set_segment_visibility(row, bool(value))

    def _on_endpoint_changed(self, row: int, slot: int | str, value: str) -> None:
        """Mutate one endpoint of segment ``row``.

        ``slot`` accepts either an integer index (new v2 API) or a legacy
        ``"a"`` / ``"b"`` string for back-compat with the v1 callers.
        """
        if self._suspend_signals or not 0 <= row < len(self._viz_segments):
            return
        if isinstance(slot, str):
            if slot == "a":
                slot = 0
            elif slot == "b":
                slot = 1
            else:
                raise ValueError(
                    f"endpoint slot must be 'a', 'b', or an int; got {slot!r}"
                )
        old = self._viz_segments[row]
        names = list(old.binding.marker_names)
        while len(names) <= slot:
            names.append("")
        names[slot] = value
        # DbC: BETWEEN_TWO requires endpoints differ; reject silently.
        if (
            old.binding.kind is BindingKind.BETWEEN_TWO
            and len(names) >= 2
            and names[0] == names[1]
        ):
            self._rebuild_table()
            return
        try:
            new_binding = MarkerBinding(
                kind=old.binding.kind,
                marker_names=tuple(names),
                rest_dimensions=old.binding.rest_dimensions,
                rest_orientation_quat=old.binding.rest_orientation_quat,
            )
        except (TypeError, ValueError):
            self._rebuild_table()
            return
        self._viz_segments[row] = SegmentVizSpec(
            binding=new_binding,
            shape_kind=old.shape_kind,
            shape_params=dict(old.shape_params),
            fitter_kind=old.fitter_kind,
            theme=old.theme,
            visible=old.visible,
        )
        self._emit_changed()

    def _on_shape_picker(self, row: int, label: str) -> None:
        if self._suspend_signals or not 0 <= row < len(self._viz_segments):
            return
        if label in _LABEL_TO_KIND:
            self._swap_shape_kind(row, _LABEL_TO_KIND[label])
            return
        if label == _SHAPE_LABEL_LIBRARY:
            self._pick_library_shape_for_row(row)
            return
        if label == _SHAPE_LABEL_MESH:
            self._pick_mesh_file_for_row(row)
            return

    def _on_geometry_changed(self, row: int, value: str) -> None:
        """Back-compat shim for v1 callers — swap shape kind by v1 name."""
        if self._suspend_signals or not 0 <= row < len(self._viz_segments):
            return
        if value not in _GEOMETRIES:
            raise ValueError(f"geometry must be one of {_GEOMETRIES}, got {value!r}")
        self._swap_shape_kind(row, value)

    def _on_group_changed(self, row: int, value: str) -> None:
        if self._suspend_signals or not 0 <= row < len(self._viz_segments):
            return
        text = value.strip() or "auto"
        self._last_group = text
        old = self._viz_segments[row]
        new_theme = ShapeTheme(
            color=old.theme.color,
            opacity=old.theme.opacity,
            edge_color=old.theme.edge_color,
            edge_width=old.theme.edge_width,
            flat_shaded=old.theme.flat_shaded,
            group=text,
        )
        self._viz_segments[row] = SegmentVizSpec(
            binding=old.binding,
            shape_kind=old.shape_kind,
            shape_params=dict(old.shape_params),
            fitter_kind=old.fitter_kind,
            theme=new_theme,
            visible=old.visible,
        )
        self._emit_changed()

    def _delete_row(self, row: int) -> None:
        if not 0 <= row < len(self._viz_segments):
            return
        del self._viz_segments[row]
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
            self._make_default_line_spec(s.a, s.b, s.group)
            for s in default_body_segments(self._marker_names)
        ]
        self._set_viz_segments(defaults)

    # -------------------------------------------------------- Library / Mesh

    def _resolve_library(self) -> ShapeLibrary | None:
        if self._shape_library is None:
            try:
                self._shape_library = ShapeLibrary.default()
            except (FileNotFoundError, ValueError) as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Library shape",
                    f"Could not load default shape library:\n{exc}",
                )
                return None
        return self._shape_library

    def _pick_library_shape_for_row(self, row: int) -> None:
        lib = self._resolve_library()
        if lib is None:
            self._rebuild_table()
            return
        names = lib.names()
        if not names:
            QtWidgets.QMessageBox.information(
                self, "Library shape", "The default library is empty."
            )
            self._rebuild_table()
            return
        dlg = _LibraryShapeDialog(names, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            self._rebuild_table()
            return
        chosen = dlg.selected_name()
        if chosen is None:
            self._rebuild_table()
            return
        old = self._viz_segments[row]
        self._viz_segments[row] = SegmentVizSpec(
            binding=old.binding,
            shape_kind="library_shape",
            shape_params={"library_name": "default", "shape_id": chosen},
            fitter_kind=old.fitter_kind,
            theme=old.theme,
            visible=old.visible,
        )
        self._rebuild_table()
        self._emit_changed()

    def _pick_mesh_file_for_row(self, row: int) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import shape file",
            str(Path.home()),
            _MESH_FILTER,
        )
        if not path:
            self._rebuild_table()
            return
        old = self._viz_segments[row]
        self._viz_segments[row] = SegmentVizSpec(
            binding=old.binding,
            shape_kind="mesh_file",
            shape_params={"path": str(path), "max_vertices": 5000},
            fitter_kind=old.fitter_kind,
            theme=old.theme,
            visible=old.visible,
        )
        self._rebuild_table()
        self._emit_changed()

    def _on_import_mesh_clicked(self) -> None:
        """Top-level "Import shape file…" — adds a NEW mesh segment."""
        if not self._marker_names or len(self._marker_names) < 2:
            QtWidgets.QMessageBox.information(
                self, "Import shape file", "Load a C3D with at least two markers first."
            )
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import shape file",
            str(Path.home()),
            _MESH_FILTER,
        )
        if not path:
            return
        a, b = self._marker_names[0], self._marker_names[1]
        spec = SegmentVizSpec(
            binding=MarkerBinding(
                kind=BindingKind.BETWEEN_TWO,
                marker_names=(a, b),
            ),
            shape_kind="mesh_file",
            shape_params={"path": str(path), "max_vertices": 5000},
            fitter_kind="between_two",
            theme=ShapeTheme(group=self._last_group),
            visible=True,
        )
        self.add_viz_segment(spec)

    def _on_library_clicked(self) -> None:
        """Top-level "Library…" — adds a NEW library-shape segment."""
        if not self._marker_names or len(self._marker_names) < 2:
            QtWidgets.QMessageBox.information(
                self, "Library", "Load a C3D with at least two markers first."
            )
            return
        lib = self._resolve_library()
        if lib is None:
            return
        names = lib.names()
        if not names:
            QtWidgets.QMessageBox.information(
                self, "Library", "The default library is empty."
            )
            return
        dlg = _LibraryShapeDialog(names, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        chosen = dlg.selected_name()
        if chosen is None:
            return
        a, b = self._marker_names[0], self._marker_names[1]
        spec = SegmentVizSpec(
            binding=MarkerBinding(
                kind=BindingKind.BETWEEN_TWO,
                marker_names=(a, b),
            ),
            shape_kind="library_shape",
            shape_params={"library_name": "default", "shape_id": chosen},
            fitter_kind="between_two",
            theme=ShapeTheme(group=self._last_group),
            visible=True,
        )
        self.add_viz_segment(spec)

    # ----------------------------------------------------------- Save / Load

    def _on_save_clicked(self) -> None:
        default = default_segment_set_path()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save segment set", str(default), "JSON files (*.json)"
        )
        if path:
            viz_set = SegmentVizSet(segments=tuple(self._viz_segments))
            viz_set.save(path)

    def _on_export_clicked(self) -> None:
        """Back-compat shim — same as Save with a clear default filename."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export segment set to JSON",
            "segments.json",
            "JSON files (*.json)",
        )
        if path:
            viz_set = SegmentVizSet(segments=tuple(self._viz_segments))
            viz_set.save(path)

    def _on_load_clicked(self) -> None:
        default = default_segment_set_path()
        start_dir = str(default.parent) if default.parent.exists() else str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load segment set", start_dir, "JSON files (*.json)"
        )
        if not path:
            return
        try:
            viz_set = SegmentVizSet.load(path)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self, "Load segment set", f"Could not load:\n{exc}"
            )
            return
        self._set_viz_segments(list(viz_set.segments))


__all__ = ["SegmentsTab"]
