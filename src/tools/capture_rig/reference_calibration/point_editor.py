"""Mark physical reference points on the existing source-pixel image canvas."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite

import cv2
import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QUndoCommand, QUndoStack
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..annotate_widget import ImageCanvas
from ..styling import reference_marker_colors

Pixel = tuple[float, float]


class _SetPoint(QUndoCommand):
    def __init__(
        self,
        identity: str,
        previous: Pixel | None,
        updated: Pixel | None,
        apply: Callable[[str, Pixel | None], None],
    ) -> None:
        super().__init__(
            f"Mark {identity}" if updated is not None else f"Clear {identity}"
        )
        self.identity, self.previous, self.updated, self.apply = (
            identity,
            previous,
            updated,
            apply,
        )

    def redo(self) -> None:
        self.apply(self.identity, self.updated)

    def undo(self) -> None:
        self.apply(self.identity, self.previous)


class ReferencePointEditor(QDialog):
    """Keyboard and mouse editing with standard Qt undo/redo and explicit Save."""

    def __init__(
        self,
        frame: npt.NDArray[np.uint8],
        point_ids: tuple[str, ...],
        *,
        context: str,
        points: Mapping[str, Pixel] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) == 0:
            raise ValueError("A nonempty original BGR frame is required")
        if not 2 <= len(point_ids) <= 128 or len(set(point_ids)) != len(point_ids):
            raise ValueError(
                "Use distinct physical point identities from the reference"
            )
        self._frame = np.ascontiguousarray(frame, dtype=np.uint8).copy()
        self._point_ids = point_ids
        self._points = dict(points or {})
        if any(
            key not in point_ids or not self._inside(value)
            for key, value in self._points.items()
        ):
            raise ValueError(
                "Existing reference points do not match this original image"
            )
        self.setWindowTitle("Mark Reference Points")
        self.resize(800, 620)
        self.undo_stack = QUndoStack(self)
        self.canvas = ImageCanvas(self)
        self.canvas.setAccessibleName("Reference point image")
        self.point = QComboBox()
        for number, identity in enumerate(point_ids, 1):
            self.point.addItem(
                f"{number}. {identity.replace('-', ' ').title()}", identity
            )
        self.pixel_x, self.pixel_y = QDoubleSpinBox(), QDoubleSpinBox()
        height, width = frame.shape[:2]
        for spin, maximum, label in (
            (self.pixel_x, width - 1, "Horizontal pixel"),
            (self.pixel_y, height - 1, "Vertical pixel"),
        ):
            spin.setRange(0, maximum)
            spin.setDecimals(2)
            spin.setAccessibleName(label)
        self.place = QPushButton("Set Point")
        self.place.setDefault(True)
        self.clear = QPushButton("Clear Point")
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.instructions = QLabel(
            "Mark the same physical corners in every camera: Origin is your marked corner; "
            "Along Arrow follows the marked long edge; Opposite and Across Width follow around the sheet. "
            "For a ruler, use the labelled measurement endpoints. Leave hidden points unmarked. "
            "These are reference observations, not camera calibration. Ctrl+wheel zooms."
        )
        self.instructions.setWordWrap(True)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._layout(context)
        self.canvas.clicked.connect(self.mark)
        self.place.clicked.connect(
            lambda: self.mark(self.pixel_x.value(), self.pixel_y.value())
        )
        self.clear.clicked.connect(self._clear)
        self.point.currentIndexChanged.connect(self._selected)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self._refresh()

    @property
    def points(self) -> dict[str, Pixel]:
        """Return a copy; callers apply it only after an accepted dialog."""
        return dict(self._points)

    def _layout(self, context: str) -> None:
        layout = QVBoxLayout(self)
        heading = QLabel(context)
        heading.setTextFormat(Qt.TextFormat.PlainText)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(self.instructions)
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        layout.addWidget(self.point)
        for label, spin in (("X", self.pixel_x), ("Y", self.pixel_y)):
            controls.addWidget(QLabel(label))
            controls.addWidget(spin)
        controls.addWidget(self.place)
        controls.addWidget(self.clear)
        layout.addLayout(controls)
        history = QHBoxLayout()
        undo = self.undo_stack.createUndoAction(self, "Undo")
        redo = self.undo_stack.createRedoAction(self, "Redo")
        assert undo is not None and redo is not None
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        redo.setShortcut(QKeySequence.StandardKey.Redo)
        for action in (undo, redo):
            self.addAction(action)
            button = QToolButton()
            button.setDefaultAction(action)
            history.addWidget(button)
        history.addWidget(self.status, 1)
        layout.addLayout(history)
        save = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        assert save is not None
        save.setAutoDefault(False)
        layout.addWidget(self.buttons)

    def _inside(self, point: Pixel) -> bool:
        height, width = self._frame.shape[:2]
        return (
            all(isfinite(value) for value in point)
            and 0 <= point[0] < width
            and 0 <= point[1] < height
        )

    def mark(self, x: float, y: float) -> None:
        """Accept an original-image pixel and select the next unmarked identity."""
        pixel = (float(x), float(y))
        if not self._inside(pixel):
            self.status.setText("Choose a point inside the original image.")
            return
        identity = str(self.point.currentData())
        self.undo_stack.push(
            _SetPoint(identity, self._points.get(identity), pixel, self.apply_point)
        )
        for index in range(len(self._point_ids)):
            next_index = (self.point.currentIndex() + index + 1) % len(self._point_ids)
            if self._point_ids[next_index] not in self._points:
                self.point.setCurrentIndex(next_index)
                break

    def apply_point(self, identity: str, pixel: Pixel | None) -> None:
        """Undo-stack callback; retain source coordinates and point identities."""
        if pixel is None:
            self._points.pop(identity, None)
        else:
            self._points[identity] = pixel
        self._refresh()

    def _clear(self) -> None:
        identity = str(self.point.currentData())
        previous = self._points.get(identity)
        if previous is not None:
            self.undo_stack.push(_SetPoint(identity, previous, None, self.apply_point))

    def _selected(self) -> None:
        identity = str(self.point.currentData())
        if identity in self._points:
            self.pixel_x.setValue(self._points[identity][0])
            self.pixel_y.setValue(self._points[identity][1])
        self.clear.setEnabled(identity in self._points)

    def _refresh(self) -> None:
        frame = self._frame.copy()
        radius = max(4, round(frame.shape[1] / 160))
        font_size = max(0.5, frame.shape[1] / 1280)
        foreground, outline = reference_marker_colors()
        for number, identity in enumerate(self._point_ids, 1):
            pixel = self._points.get(identity)
            if pixel is None:
                continue
            location = (round(pixel[0]), round(pixel[1]))
            cv2.circle(frame, location, radius + 2, outline, 3, cv2.LINE_AA)
            cv2.circle(frame, location, radius, foreground, 2, cv2.LINE_AA)
            label = (
                max(0, min(frame.shape[1] - 30, location[0] + radius + 4)),
                max(15, location[1] - radius),
            )
            cv2.putText(
                frame,
                str(number),
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                foreground,
                2,
                cv2.LINE_AA,
            )
        self.canvas.set_image(frame)
        count, total = len(self._points), len(self._point_ids)
        qualifier = (
            "Partial observation; mark more visible points later."
            if count < total
            else "All identities marked; review their physical order."
        )
        self.status.setText(f"{count} of {total} points marked. {qualifier}")
        save = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        assert save is not None
        save.setEnabled(count > 0)
        self._selected()
