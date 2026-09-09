"""Explicit spatial placement and event synchronization editors for instructors."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QSignalBlocker, pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.reference.registration import ReferenceRegistration, TimeMapping
from src.shared.python.spatial_algebra.pose6dof import (
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
)


def number(
    label: str, bounds: tuple[float, float], step: float = 0.05
) -> QDoubleSpinBox:
    """Consistent keyboard-editable numeric field with an accessible name."""
    field = QDoubleSpinBox()
    field.setRange(*bounds)
    field.setDecimals(5)
    field.setSingleStep(step)
    field.setAccessibleName(label)
    field.setKeyboardTracking(False)
    return field


class SpatialControls(QWidget):
    """Keep 3-D placement and 2-D image adjustment distinct and reversible."""

    changed = pyqtSignal(object)
    pending_changed = pyqtSignal()

    def __init__(
        self,
        registration: ReferenceRegistration,
        kind: str,
        scene_size: tuple[int, int],
        *,
        reference_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.registration = registration
        self.scene_size, self.reference_size = scene_size, reference_size or scene_size
        self.pending = False
        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        layout.addLayout(self.form)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        self.translation = [
            number(f"Scene {axis} Position (m)", (-1000, 1000)) for axis in "XYZ"
        ]
        self.rotation = [
            number(f"Scene {axis} Rotation (deg)", (-180, 180), 1) for axis in "XYZ"
        ]
        self.scale = number("Model Scale", (0.001, 100), 0.05)
        self.image_x = number("Move Image X (px)", (-10000, 10000), 1)
        self.image_y = number("Move Image Y (px)", (-10000, 10000), 1)
        self.image_angle = number("Rotate Image (deg)", (-180, 180), 1)
        self.image_scale = number("Resize Image", (0.01, 100), 0.05)
        if kind == "motion":
            self._motion_form()
        elif kind == "video":
            self._image_form()
        else:
            raise ValueError("Choose motion or video placement")
        layout.addWidget(self.problem)
        layout.addStretch()
        self.set_registration(registration)
        for field in (
            *self.translation,
            *self.rotation,
            self.scale,
            self.image_x,
            self.image_y,
            self.image_angle,
            self.image_scale,
        ):
            field.valueChanged.connect(self._edited)

    def _motion_form(self) -> None:
        hint = QLabel(
            "Place the model in the camera scene. X points toward the target, Y up, Z right. Rotations apply X, then Y, then Z."
        )
        hint.setWordWrap(True)
        self.form.addRow(hint)
        for label, field in zip(
            (
                "X Position (m)",
                "Y Position (m)",
                "Z Position (m)",
                "X Rotation (°)",
                "Y Rotation (°)",
                "Z Rotation (°)",
            ),
            (*self.translation, *self.rotation),
            strict=True,
        ):
            self.form.addRow(label, field)
        self.form.addRow("Model Scale", self.scale)
        button = QPushButton("Apply Model Placement")
        button.clicked.connect(self.apply_placement)
        self.form.addRow(button)

    def _image_form(self) -> None:
        hint = QLabel(
            "Move, rotate and resize the expert image over the player. Adjustments keep any saved perspective alignment. This is a 2-D comparison."
        )
        hint.setWordWrap(True)
        self.form.addRow(hint)
        for label, field in (
            ("Move X (px)", self.image_x),
            ("Move Y (px)", self.image_y),
            ("Rotate (°)", self.image_angle),
            ("Resize", self.image_scale),
        ):
            self.form.addRow(label, field)
        button = QPushButton("Apply Image Adjustment")
        button.clicked.connect(self.apply_image)
        self.form.addRow(button)

    def _edited(self) -> None:
        self.pending = True
        self.pending_changed.emit()

    def set_registration(self, registration: ReferenceRegistration) -> None:
        self.registration = registration
        transform = registration.transform
        angles = np.rad2deg(rotation_matrix_to_euler(np.asarray(transform.rotation)))
        for field, value in zip(
            (*self.translation, *self.rotation, self.scale),
            (*transform.translation_m, *angles, transform.scale),
            strict=True,
        ):
            with QSignalBlocker(field):
                field.setValue(float(value))
        for field, value in (
            (self.image_x, 0),
            (self.image_y, 0),
            (self.image_angle, 0),
            (self.image_scale, 1),
        ):
            with QSignalBlocker(field):
                field.setValue(value)
        self.pending = False
        self.problem.clear()

    def _apply(self, values: dict[str, object]) -> bool:
        try:
            changed = ReferenceRegistration.model_validate(
                self.registration.model_dump() | values
            )
        except ValueError as exc:
            self.problem.setText(str(exc))
            return False
        self.set_registration(changed)
        self.changed.emit(changed)
        return True

    def apply_placement(self) -> bool:
        angles = np.deg2rad([field.value() for field in self.rotation])
        transform = self.registration.transform
        values = transform.model_dump() | {
            "translation_m": tuple(field.value() for field in self.translation),
            "rotation": euler_to_rotation_matrix(angles).tolist(),
            "scale": self.scale.value(),
            "is_calibrated": False,
        }
        return self._apply({"transform": values, "is_calibrated": False})

    def apply_image(self) -> bool:
        width, height = self.scene_size
        reference_width, reference_height = self.reference_size
        fit = min(width / reference_width, height / reference_height)
        base = self.registration.image_transform_2d or (
            (fit, 0, (width - reference_width * fit) / 2),
            (0, fit, (height - reference_height * fit) / 2),
            (0, 0, 1),
        )
        angle = np.deg2rad(self.image_angle.value())
        c, s = (
            np.cos(angle) * self.image_scale.value(),
            np.sin(angle) * self.image_scale.value(),
        )
        cx, cy = width / 2, height / 2
        adjustment = np.array(
            (
                (c, -s, cx - c * cx + s * cy + self.image_x.value()),
                (s, c, cy - s * cx - c * cy + self.image_y.value()),
                (0, 0, 1),
            )
        )
        return self._apply(
            {
                "image_transform_2d": (adjustment @ np.asarray(base)).tolist(),
                "is_calibrated": False,
            }
        )


class TimeControls(QWidget):
    """Choose affine clock mapping or validated pairs of instructor-marked events."""

    changed = pyqtSignal(object)
    pending_changed = pyqtSignal()

    def __init__(self, registration: ReferenceRegistration) -> None:
        super().__init__()
        self.registration = registration
        self.pending = False
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.offset = number("Time Offset (s)", (-86400, 86400))
        self.rate = number("Reference Time Rate", (0.25, 4))
        form.addRow("Offset (s)", self.offset)
        form.addRow("Time Rate", self.rate)
        layout.addLayout(form)
        explanation = QLabel(
            "Paired events replace the offset. Two or more events also set the interval rates. Clear events to use the offset and rate fields."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.affine_button = QPushButton("Use Offset and Rate (Clear Events)")
        self.affine_button.clicked.connect(self.apply_affine)
        layout.addWidget(self.affine_button)
        self.events = QTableWidget(0, 3)
        self.events.setHorizontalHeaderLabels(
            ("Event", "Expert (s)", "Player Scene (s)")
        )
        header = self.events.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.events.setMinimumHeight(110)
        self.events.setAccessibleName("Paired Swing Events")
        layout.addWidget(self.events, 1)
        buttons = QHBoxLayout()
        for label, slot in (
            ("Add Event", self.add_event),
            ("Remove", self.remove_event),
            ("Apply Events", self.apply_events),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.problem = QLabel()
        self.problem.setWordWrap(True)
        layout.addWidget(self.problem)
        self.set_registration(registration)
        self.offset.valueChanged.connect(self._edited)
        self.rate.valueChanged.connect(self._edited)
        self.events.itemChanged.connect(self._edited)

    def _edited(self) -> None:
        self.pending = True
        self.pending_changed.emit()

    def set_registration(self, registration: ReferenceRegistration) -> None:
        self.registration = registration
        mapping = registration.time_mapping
        for field, value in (
            (self.offset, mapping.offset_s),
            (self.rate, mapping.rate_scale),
        ):
            with QSignalBlocker(field):
                field.setValue(value)
        anchors = mapping.event_anchors
        reference = anchors.reference if anchors else {}
        scene = anchors.scene if anchors else {}
        with QSignalBlocker(self.events):
            self.events.setRowCount(len(reference))
            for row, (name, time) in enumerate(reference.items()):
                for column, text in enumerate((name, str(time), str(scene[name]))):
                    self.events.setItem(row, column, QTableWidgetItem(text))
        self.pending = False
        self.problem.clear()

    def add_event(self) -> None:
        if self.events.rowCount() >= 32:
            self.problem.setText("Use at most 32 paired events.")
            return
        self.events.insertRow(self.events.rowCount())
        self._edited()

    def remove_event(self) -> None:
        row = self.events.currentRow()
        if row >= 0:
            self.events.removeRow(row)
            self._edited()

    def _apply(self, mapping: dict[str, object]) -> bool:
        try:
            time = TimeMapping.model_validate(mapping)
            changed = ReferenceRegistration.model_validate(
                self.registration.model_dump() | {"time_mapping": time}
            )
        except ValueError as exc:
            self.problem.setText(str(exc))
            return False
        self.set_registration(changed)
        self.changed.emit(changed)
        return True

    def apply_affine(self) -> bool:
        return self._apply(
            {"offset_s": self.offset.value(), "rate_scale": self.rate.value()}
        )

    def apply_events(self) -> bool:
        if self.events.rowCount() == 0:
            return self.apply_affine()
        reference, scene = {}, {}
        try:
            for row in range(self.events.rowCount()):
                values = [self.events.item(row, column) for column in range(3)]
                if any(value is None for value in values):
                    raise ValueError("Fill the name and both times for every event.")
                name, ref, player = [
                    value.text().strip() for value in values if value is not None
                ]
                if name in reference:
                    raise ValueError("Event names must be unique.")
                reference[name], scene[name] = float(ref), float(player)
        except ValueError as exc:
            self.problem.setText(str(exc))
            return False
        return self._apply(
            {
                "rate_scale": self.rate.value(),
                "event_anchors": {"reference": reference, "scene": scene},
            }
        )
