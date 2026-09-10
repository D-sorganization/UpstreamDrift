"""Reusable metric scene-reference editor for model and video analysis views."""

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.coaching import (
    History,
    ReferenceGeometry,
    ReferencePlane,
    ReferencePoint,
)


class GeometryControls(QWidget):
    """Edit one immutable world-metre document using the shared undo history."""

    changed = pyqtSignal(object)
    pending_changed = pyqtSignal()

    def __init__(self, geometry: ReferenceGeometry) -> None:
        super().__init__()
        if not isinstance(geometry, ReferenceGeometry):
            raise TypeError("GeometryControls requires a ReferenceGeometry document")
        self.history = History(geometry)
        self.selector = QComboBox()
        self.selector.setAccessibleName("Scene Reference")
        self.selector.currentIndexChanged.connect(self._selection_changed)
        self.title = QLineEdit()
        self.visible = QCheckBox("Visible")
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0, 1)
        self.opacity.setSingleStep(0.05)
        self.extent = QDoubleSpinBox()
        self.extent.setDecimals(3)
        self.extent.setRange(0.001, 1000)
        self.fields: dict[str, tuple[QDoubleSpinBox, ...]] = {}
        self._display_values: dict[str, Any] = {}
        self.status = QLabel()
        self.status.setWordWrap(True)
        self._build()
        self._sync()
        self.title.textChanged.connect(self._notify_pending)
        self.visible.toggled.connect(self._notify_pending)
        for field in (
            self.opacity,
            self.extent,
            *(spin for row in self.fields.values() for spin in row),
        ):
            field.valueChanged.connect(self._notify_pending)

    def _notify_pending(self, *_values: object) -> None:
        self.pending_changed.emit()

    @property
    def document(self) -> ReferenceGeometry:
        return self.history.current

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        help_text = QLabel(
            "World coordinates in metres (Y up). Plane anchors define its signed normal; extent changes only its visible area."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        layout.addWidget(self.selector)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.addRow("Name", self.title)
        for name, label in (
            ("origin_m", "Plane Origin"),
            ("along_m", "Along Anchor"),
            ("across_m", "Across Anchor"),
            ("position_m", "Point"),
        ):
            row = QHBoxLayout()
            fields = []
            for axis in "XYZ":
                spin = QDoubleSpinBox()
                spin.setDecimals(4)
                spin.setRange(-10000, 10000)
                spin.setAccessibleName(f"{label} {axis} in metres")
                spin.setMinimumWidth(60)
                spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
                axis_label = QLabel(axis)
                axis_label.setFixedWidth(12)
                row.addWidget(axis_label)
                row.addWidget(spin, 1)
                fields.append(spin)
            self.fields[name] = tuple(fields)
            form.addRow(label, row)
        form.addRow("Half Extent (m)", self.extent)
        form.addRow("Opacity", self.opacity)
        form.addRow(self.visible)
        layout.addLayout(form)
        for entries in (
            (("Add Plane", self.add_plane), ("Add Point", self.add_point)),
            (("Apply", self.apply_selected), ("Delete", self.delete_selected)),
            (("Undo", self.undo), ("Redo", self.redo)),
        ):
            row = QHBoxLayout()
            for label, callback in entries:
                button = QPushButton(label)
                button.clicked.connect(callback)
                row.addWidget(button)
            layout.addLayout(row)
        layout.addWidget(self.status)

    def _selected(self) -> ReferencePlane | ReferencePoint | None:
        identity = self.selector.currentData()
        items: tuple[ReferencePlane | ReferencePoint, ...] = (
            *self.document.planes,
            *self.document.points,
        )
        return next(
            (item for item in items if item.id == identity),
            None,
        )

    def _sync(self, identity: str | None = None) -> None:
        selected = identity or self.selector.currentData()
        with QSignalBlocker(self.selector):
            self.selector.clear()
            for item in (*self.document.planes, *self.document.points):
                self.selector.addItem(item.title, item.id)
            index = self.selector.findData(selected)
            if index >= 0:
                self.selector.setCurrentIndex(index)
        self._selection_changed()

    def _selection_changed(self) -> None:
        item = self._selected()
        if item is None:
            return
        self.title.setText(item.title)
        self.opacity.setValue(item.opacity)
        self.visible.setChecked(item.visible)
        plane = isinstance(item, ReferencePlane)
        self.extent.setEnabled(plane)
        if isinstance(item, ReferencePlane):
            self.extent.setValue(item.half_size_m)
        values = item.model_dump()
        for name, fields in self.fields.items():
            for axis, spin in enumerate(fields):
                spin.setEnabled(name in values)
                spin.setValue(values[name][axis] if name in values else 0)
        self._display_values = self._input_values()
        self.pending_changed.emit()

    def _commit(self, geometry: ReferenceGeometry, identity: str | None = None) -> None:
        self.history.apply(geometry)
        self._sync(identity)
        self.status.clear()
        self.changed.emit(self.document)

    def _replace(self, item: ReferencePlane | ReferencePoint) -> bool:
        field = "planes" if isinstance(item, ReferencePlane) else "points"
        values = self.document.model_dump()
        collection = [value for value in values[field] if value["id"] != item.id]
        values[field] = collection + [item.model_dump()]
        try:
            document = ReferenceGeometry.model_validate(values)
        except ValueError as exc:
            self.status.setText(str(exc))
            return False
        self._commit(document, item.id)
        return True

    def add_plane(self) -> None:
        """Add a world XY reference; users explicitly edit its metric anchors."""
        self._replace(
            ReferencePlane(
                title="Plane", origin_m=(0, 0, 0), along_m=(1, 0, 0), across_m=(0, 1, 0)
            )
        )

    def add_point(self) -> None:
        """Add a named point at the world origin."""
        self._replace(ReferencePoint(title="Point", position_m=(0, 0, 0)))

    @property
    def pending(self) -> bool:
        """Whether numeric fields differ from the selected applied reference."""
        item = self._selected()
        return item is not None and self._input_values() != self._display_values

    def _input_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {
            "title": self.title.text(),
            "opacity": self.opacity.value(),
            "visible": self.visible.isChecked(),
            "half_size_m": self.extent.value(),
        }
        for name, fields in self.fields.items():
            values[name] = tuple(spin.value() for spin in fields)
        return values

    def _field_values(self, item: ReferencePlane | ReferencePoint) -> dict[str, Any]:
        values = item.model_dump()
        for name, value in self._input_values().items():
            if name in values and value != self._display_values.get(name):
                baseline = self._display_values[name]
                if name in self.fields:
                    values[name] = tuple(
                        old if new == displayed else new
                        for old, new, displayed in zip(
                            values[name], value, baseline, strict=True
                        )
                    )
                else:
                    values[name] = value
        return values

    def apply_selected(self) -> bool:
        """Apply validated fields; return false and preserve state on failure."""
        item = self._selected()
        if item is None:
            return True
        try:
            return self._replace(type(item).model_validate(self._field_values(item)))
        except ValueError as exc:
            self.status.setText(str(exc))
            return False

    def delete_selected(self) -> None:
        """Remove the selected reference as one reversible edit."""
        identity = self.selector.currentData()
        values = self.document.model_dump()
        for field in ("planes", "points"):
            values[field] = [item for item in values[field] if item["id"] != identity]
        self._commit(ReferenceGeometry.model_validate(values))

    def undo(self) -> None:
        """Restore the preceding saved edit snapshot."""
        self.history.undo()
        self._sync()
        self.changed.emit(self.document)

    def redo(self) -> None:
        """Restore an undone edit snapshot."""
        self.history.redo()
        self._sync()
        self.changed.emit(self.document)

    def save(self, path: Path) -> None:
        """Persist the currently applied document with atomic replacement."""
        if self.pending and not self.apply_selected():
            raise ValueError(self.status.text())
        self.document.save(path)

    def load(self, path: Path) -> None:
        """Load matching scene geometry as one reversible edit."""
        self._commit(ReferenceGeometry.load(path, scene_id=self.document.scene_id))
