"""Saved coaching references, with accessible numeric editing and export."""

from __future__ import annotations

from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.coaching import Drawing

from . import styling
from .coaching_canvas import CoachingCanvas, FrameFinalizer
from .coaching_source import CaptureCoachingSource, CoachingSource
from .flow_layout import FlowLayout
from .session import load_session
from .swing_export_actions import ExportJobSpec, SwingExportActions


class CoachingDialog(QDialog):
    def __init__(
        self,
        root: Path,
        view: str,
        parent: QWidget | None = None,
        *,
        media: CoachingSource | None = None,
        finalize: FrameFinalizer | None = None,
    ) -> None:
        super().__init__(parent)
        self.root, self.view = root, view
        self.media = media or CaptureCoachingSource(root, view)
        self.reader = self.media.reader
        saved = self.media.drawings
        self._saved = saved
        self.canvas = CoachingCanvas(saved, finalize=finalize)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.canvas.interaction_started.connect(self._stop)
        self.setWindowTitle(f"Coaching References · {view}[*]")
        self.resize(900, 720)
        self.tool = QComboBox()
        self.tool.setAccessibleName("Drawing tool")
        self.tool.addItems(
            ["Select", "Line", "Arrow", "Circle", "Ellipse", "Rectangle"]
        )
        self.tool.currentTextChanged.connect(self._tool_changed)
        self.shapes = QComboBox()
        self.shapes.setAccessibleName("Select a saved reference")
        self.shapes.currentIndexChanged.connect(
            lambda: self.canvas.select(self.shapes.currentData())
        )
        self.stroke_width = QSpinBox()
        self.stroke_width.setRange(1, 40)
        self.stroke_width.setValue(3)
        self.stroke_width.setAccessibleName("Stroke width in source pixels")
        self.stroke_width.valueChanged.connect(self._style_changed)
        self.colour = QPushButton("Colour…")
        self.colour.clicked.connect(self._choose_colour)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, saved.frames - 1)
        self.slider.setAccessibleName("Original video frame")
        self.slider.valueChanged.connect(self._show_frame)
        self.play = QPushButton("Play")
        self.play.clicked.connect(self._play)
        self.clock, self.status = QLabel(), QLabel()
        self.status.setWordWrap(True)
        self.geometry_fields: dict[str, QDoubleSpinBox] = {}
        self.visible = QCheckBox("Visible")
        self.first, self.last = QSpinBox(), QSpinBox()
        self.first.setRange(0, saved.frames - 1)
        self.last.setRange(0, saved.frames - 1)
        self.first.setAccessibleName("First visible source frame")
        self.last.setAccessibleName("Last visible source frame")
        self.undo_button, self.redo_button = QPushButton("Undo"), QPushButton("Redo")
        self.undo_button.clicked.connect(self.canvas.undo)
        self.redo_button.clicked.connect(self.canvas.redo)
        self.exporter = SwingExportActions(
            self,
            root,
            view=lambda: view,
            save=self.save,
            status=self.status.setText,
            drawings=lambda: self.canvas.layer,
            job=ExportJobSpec(
                lambda: self.media.export_job(self.canvas.layer),
                "Export Annotated Analysis",
                "analysis.mp4",
            ),
            label="Export Annotated Swing…",
        )
        self._build()
        self.canvas.changed.connect(self._sync)
        styling.apply_theme(self)
        self._show_frame(0)
        self._sync()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        help_text = QLabel(
            "Choose a tool and drag on the image. Select a stroke to move it; drag its white handles to resize. Arrow keys nudge, Delete removes, Ctrl+Z undoes. Drawings are visual references, not measured landmarks."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        tools = QHBoxLayout()
        width_label = QLabel("Width (px)")
        width_label.setBuddy(self.stroke_width)
        for widget in (
            self.tool,
            self.colour,
            width_label,
            self.stroke_width,
            self.undo_button,
            self.redo_button,
        ):
            tools.addWidget(widget)
        layout.addLayout(tools)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.slider)
        playback = QHBoxLayout()
        playback.addWidget(self.play)
        playback.addWidget(self.clock, 1)
        layout.addLayout(playback)
        selection = QHBoxLayout()
        selection.addWidget(QLabel("Reference"))
        selection.addWidget(self.shapes, 1)
        add = QPushButton("Add at Centre")
        add.setToolTip(
            "Create the selected tool at the image centre, then edit its coordinates using the keyboard."
        )
        add.clicked.connect(self.add_center)
        selection.addWidget(add)
        layout.addLayout(selection)
        grid = QGridLayout()
        for column, name in enumerate(("x1", "y1", "x2", "y2")):
            field = QDoubleSpinBox()
            layer = self.canvas.layer
            field.setRange(
                0, (layer.width if name.startswith("x") else layer.height) - 1
            )
            field.setDecimals(1)
            field.setAccessibleName(f"Reference {name} in source pixels")
            self.geometry_fields[name] = field
            label = QLabel(name.upper())
            label.setBuddy(field)
            grid.addWidget(label, 0, column * 2)
            grid.addWidget(field, 0, column * 2 + 1)
        grid.addWidget(self.visible, 1, 0, 1, 2)
        grid.addWidget(QLabel("From Frame"), 1, 2)
        grid.addWidget(self.first, 1, 3)
        grid.addWidget(QLabel("Through Frame"), 1, 4)
        grid.addWidget(self.last, 1, 5)
        apply = QPushButton("Apply")
        apply.clicked.connect(self.apply_properties)
        grid.addWidget(apply, 1, 6, 1, 2)
        layout.addLayout(grid)
        buttons = FlowLayout(spacing=6)
        for title, callback in (
            ("Delete Selected", self.canvas.delete),
            ("Clear (Undoable)", self.canvas.clear),
            ("Export Still…", self._choose_still),
            ("Save References", self.save),
            ("Close", self.close),
        ):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.add_widget(button)
        buttons.add_widget(self.exporter.button)
        layout.addLayout(buttons)
        layout.addWidget(self.status)

    def _tool_changed(self, text: str) -> None:
        self.canvas.cancel_gesture()
        self.canvas.tool = text.lower()
        self.canvas.refresh()

    def _choose_colour(self) -> None:
        colour = QColorDialog.getColor(
            QColor(self.canvas.colour), self, "Reference Colour"
        )
        if colour.isValid():
            self.canvas.colour = colour.name()
            self._style_changed()

    def _style_changed(self) -> None:
        self.canvas.stroke = self.stroke_width.value()
        self.colour.setText(f"Colour {self.canvas.colour}…")
        selected = self.canvas.selection()
        if selected:
            self.canvas.apply(
                self.canvas.layer.with_shape(
                    selected.changed(
                        colour=self.canvas.colour, stroke=self.canvas.stroke
                    )
                )
            )

    def _sync(self) -> None:
        layer = self.canvas.layer
        self.shapes.blockSignals(True)
        self.shapes.clear()
        self.shapes.addItem("No Selection", None)
        for index, shape in enumerate(layer.shapes, 1):
            self.shapes.addItem(
                f"{index} · {shape.kind.title()}"
                + (" · Hidden Here" if not shape.at(self.slider.value()) else ""),
                shape.id,
            )
        self.shapes.setCurrentIndex(max(0, self.shapes.findData(self.canvas.selected)))
        self.shapes.blockSignals(False)
        history = self.canvas.history
        self.undo_button.setEnabled(history.can_undo)
        self.redo_button.setEnabled(history.can_redo)
        selected = self.canvas.selection()
        for field in self.geometry_fields.values():
            field.setEnabled(selected is not None)
        if selected:
            for name, value in zip(
                ("x1", "y1", "x2", "y2"), (*selected.start, *selected.end), strict=True
            ):
                self.geometry_fields[name].setValue(value)
            self.visible.setChecked(selected.visible)
            self.first.setValue(selected.first)
            self.last.setValue(
                selected.last if selected.last is not None else layer.frames - 1
            )
            self.canvas.colour = selected.colour
            self.stroke_width.blockSignals(True)
            self.stroke_width.setValue(selected.stroke)
            self.stroke_width.blockSignals(False)
        self.colour.setText(f"Colour {self.canvas.colour}…")
        self.setWindowModified(self._has_unsaved())

    def _has_unsaved(self) -> bool:
        return self.canvas.layer != self._saved or self.media.dirty

    def add_center(self) -> None:
        kind = self.tool.currentText().lower()
        if kind == "select":
            self.status.setText("Choose a drawing tool before adding a reference.")
            return
        layer = self.canvas.layer
        size = min(layer.width, layer.height) / 4
        x, y = layer.width / 2, layer.height / 2
        shape = Drawing.model_validate(
            {
                "kind": kind,
                "start": (x - size / 2, y - size / 2),
                "end": (x + size / 2, y + size / 2),
                "colour": self.canvas.colour,
                "stroke": self.stroke_width.value(),
            }
        )
        self.canvas.selected = shape.id
        self.canvas.apply(layer.with_shape(shape))

    def apply_properties(self) -> None:
        shape = self.canvas.selection()
        if shape is None:
            return
        values = {name: field.value() for name, field in self.geometry_fields.items()}
        try:
            updated = shape.changed(
                start=(values["x1"], values["y1"]),
                end=(values["x2"], values["y2"]),
                first=self.first.value(),
                last=self.last.value(),
                visible=self.visible.isChecked(),
            )
            self.canvas.apply(self.canvas.layer.with_shape(updated))
            self.status.setText("Reference Updated")
        except ValueError as exc:
            self.status.setText(str(exc))

    def _show_frame(self, frame: int) -> None:
        image = self.reader.read(frame)
        if image is None:
            self._stop()
            self.status.setText(f"Could not decode frame {frame}.")
            return
        self.canvas.set_frame(image, frame)
        self.clock.setText(
            f"Source Frame {frame} / {self.reader.frame_count - 1} · {self.media.time_at(frame):.3f} s"
        )
        self._sync()

    def _stop(self) -> None:
        self._timer.stop()
        self.play.setText("Play")

    def _play(self) -> None:
        if self._timer.isActive():
            self._stop()
        else:
            if self.slider.value() == self.slider.maximum():
                self.slider.setValue(0)
            self._timer.start(max(1, round(1000 / (self.reader.fps or 30))))
            self.play.setText("Pause")

    def _advance(self) -> None:
        if self.slider.value() >= self.slider.maximum():
            self._stop()
        else:
            self.slider.setValue(self.slider.value() + 1)

    def save(self) -> bool:
        try:
            self.media.save(self.canvas.layer)
            self._saved = self.canvas.layer
            self.status.setText(
                "References Saved · Original Video and Pose Data Preserved"
            )
            self._sync()
            return True
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))
            return False

    def _choose_still(self) -> None:
        self._stop()
        name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Reference Still",
            str(self.root / "reference.png"),
            "PNG Image (*.png)",
        )
        if name and self.save():
            try:
                self.media.still(self.canvas.layer, self.slider.value(), Path(name))
                self.status.setText(f"Reference Image and Sidecar Saved: {name}")
            except (ValueError, OSError, cv2.error) as exc:
                self.status.setText(str(exc))

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        self._stop()
        if not self.exporter.can_close():
            event.ignore()
            return
        if self._has_unsaved():
            answer = QMessageBox.question(
                self,
                "Unsaved References",
                "Save changes to coaching references?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel or (
                answer == QMessageBox.StandardButton.Save and not self.save()
            ):
                event.ignore()
                return
        self.reader.close()
        event.accept()

    def reject(self) -> None:
        """Escape follows the same unsaved/export guard as the window close button."""
        self.close()


def show_coaching(root: Path, parent: QWidget, view: str | None = None) -> None:
    """Shared entry from Capture Library and the pre-ingestion editor."""
    try:
        if view is None:
            media = load_session(root)
            views = [item.view for item in media.views if item.recording is not None]
            if not views:
                raise ValueError("This capture has no original recordings")
            view, ok = QInputDialog.getItem(
                parent, "Coaching References", "Camera View", views, editable=False
            )
            if not ok:
                return
        CoachingDialog(root, view, parent).exec()
    except (ValueError, OSError, cv2.error) as exc:
        QMessageBox.warning(parent, "Coaching References", str(exc))
