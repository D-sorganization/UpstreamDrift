"""Native drawing gestures on the existing letterbox-aware image canvas."""

from __future__ import annotations

from math import copysign, hypot

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QMouseEvent

from src.motion_capture.coaching import Drawing, DrawingLayer, History, render_layer

from .annotate_widget import ImageCanvas


class CoachingCanvas(ImageCanvas):
    changed = pyqtSignal()
    interaction_started = pyqtSignal()

    def __init__(self, layer: DrawingLayer) -> None:
        super().__init__()
        self.history = History(layer)
        self.tool = "select"
        self.colour = "#ffcc33"
        self.stroke = 3
        self.selected: str | None = None
        self.frame = 0
        self._source: npt.NDArray[np.uint8] | None = None
        self._anchor: tuple[float, float] | None = None
        self._original: Drawing | None = None
        self._handle: str | None = None
        self._preview: DrawingLayer | None = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Coaching drawing canvas")
        self.setToolTip(
            "Drag to draw. Select a stroke to move it; drag an endpoint to resize. Arrow keys nudge; Delete removes; Ctrl+Z undoes; Ctrl+wheel zooms."
        )

    @property
    def layer(self) -> DrawingLayer:
        return self.history.current

    def selection(self) -> Drawing | None:
        return next(
            (shape for shape in self.layer.shapes if shape.id == self.selected), None
        )

    def select(self, identity: str | None) -> None:
        self.selected = identity
        self.refresh()
        self.changed.emit()

    def set_frame(self, image: npt.NDArray[np.uint8], frame: int) -> None:
        self._source, self.frame = image, frame
        self.cancel_gesture()
        self.refresh()

    def apply(self, layer: DrawingLayer) -> None:
        self.history.apply(layer)
        self.refresh()
        self.changed.emit()

    def undo(self) -> None:
        self.cancel_gesture()
        self.history.undo()
        self.refresh()
        self.changed.emit()

    def redo(self) -> None:
        self.cancel_gesture()
        self.history.redo()
        self.refresh()
        self.changed.emit()

    def delete(self) -> None:
        if self.selected:
            self.apply(self.layer.without(self.selected))

    def clear(self) -> None:
        self.apply(self.layer.with_shapes(()))

    def refresh(self) -> None:
        if self._source is None:
            return
        import cv2

        document = self._preview or self.layer
        image = render_layer(self._source, document, self.frame)
        shape = next(
            (item for item in document.shapes if item.id == self.selected), None
        )
        if shape and shape.at(self.frame):
            radius = max(1, round(self._tolerance() * 0.6))
            for x, y in (shape.start, shape.end):
                corner = (round(x), round(y))
                cv2.circle(image, corner, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(image, corner, radius, (255, 255, 255), -1, cv2.LINE_AA)
        self.set_image(image)

    def _point(self, event: QMouseEvent) -> tuple[float, float] | None:
        point = self.image_point_from_widget(event.position().toPoint())
        if point is None:
            return None
        layer = self.layer
        return (
            max(0, min(layer.width - 1, point[0])),
            max(0, min(layer.height - 1, point[1])),
        )

    def _tolerance(self) -> float:
        layer = self.layer
        scale = (
            min(self.width() / layer.width, self.height() / layer.height) * self.zoom
        )
        return 7 / max(scale, 0.01)

    def cancel_gesture(self) -> None:
        self._anchor = None
        self._original = None
        self._handle = None
        self._preview = None

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        self.interaction_started.emit()
        point = self._point(event)
        self.cancel_gesture()
        if point is None:
            return
        self._anchor = point
        if self.tool == "select":
            selected = self.selection()
            if selected and selected.at(self.frame):
                for handle in ("start", "end"):
                    endpoint = selected.start if handle == "start" else selected.end
                    if (
                        hypot(point[0] - endpoint[0], point[1] - endpoint[1])
                        <= self._tolerance()
                    ):
                        self._original, self._handle = selected, handle
                        return
            found = next(
                (
                    shape
                    for shape in reversed(self.layer.shapes)
                    if shape.at(self.frame)
                    and shape.hit(point, tolerance=self._tolerance())
                ),
                None,
            )
            self._original = found
            self.select(found.id if found else None)

    def _candidate(self, point: tuple[float, float]) -> Drawing | None:
        assert self._anchor is not None
        if self.tool == "select":
            original = self._original
            if original is None:
                return None
            if self._handle:
                values = {self._handle: point}
                if original.kind == "circle":
                    fixed = original.end if self._handle == "start" else original.start
                    values[self._handle] = self._circle_end(fixed, point)
                return original.changed(**values)
            dx, dy = point[0] - self._anchor[0], point[1] - self._anchor[1]
            layer = self.layer
            dx = max(
                -min(original.start[0], original.end[0]),
                min(dx, layer.width - 1 - max(original.start[0], original.end[0])),
            )
            dy = max(
                -min(original.start[1], original.end[1]),
                min(dy, layer.height - 1 - max(original.start[1], original.end[1])),
            )
            return original.translated(dx, dy)
        if self.tool == "circle":
            point = self._circle_end(self._anchor, point)
        return Drawing.model_validate(
            {
                "kind": self.tool,
                "start": self._anchor,
                "end": point,
                "colour": self.colour,
                "stroke": self.stroke,
            }
        )

    @staticmethod
    def _circle_end(
        start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[float, float]:
        dx, dy = end[0] - start[0], end[1] - start[1]
        size = min(abs(dx), abs(dy))
        return start[0] + copysign(size, dx), start[1] + copysign(size, dy)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is None or self._anchor is None:
            return
        point = self._point(event)
        if point is None:
            return
        try:
            shape = self._candidate(point)
            if shape:
                self._preview = self.layer.with_shape(shape)
                self.selected = shape.id
                self.refresh()
        except ValueError:
            self._preview = None
            self.refresh()

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is None or event.button() != Qt.MouseButton.LeftButton:
            return
        self.mouseMoveEvent(event)
        result = self._preview
        self.cancel_gesture()
        if result:
            self.apply(result)

    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        key, modifiers = event.key(), event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Z:
            self.redo() if modifiers & Qt.KeyboardModifier.ShiftModifier else self.undo()
        elif modifiers & Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Y:
            self.redo()
        elif key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete()
        elif key == Qt.Key.Key_Escape:
            self.cancel_gesture()
            self.refresh()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            shape = self.selection()
            delta = 10 if modifiers & Qt.KeyboardModifier.ShiftModifier else 1
            dx = delta * ((key == Qt.Key.Key_Right) - (key == Qt.Key.Key_Left))
            dy = delta * ((key == Qt.Key.Key_Down) - (key == Qt.Key.Key_Up))
            if shape:
                try:
                    self.apply(self.layer.with_shape(shape.translated(dx, dy)))
                except ValueError:
                    pass  # Boundary nudge leaves the reference in the image.
        else:
            super().keyPressEvent(event)
