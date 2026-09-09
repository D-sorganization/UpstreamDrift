"""Frame marks and source-pixel crop before pose ingestion (#9860)."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QMouseEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.rig.bundle import load_bundle
from src.motion_capture.rig.edits import (
    CropRect,
    SessionEdits,
    ViewEdit,
    load_edits,
    save_edits,
)

from .annotate_widget import ImageCanvas
from . import styling
from .player import VideoReader
from .swing_export_actions import SwingExportActions
from .coaching_dialog import show_coaching
from .flow_layout import FlowLayout


class CropCanvas(ImageCanvas):
    """Reuse the annotation canvas' exact letterbox/zoom coordinate mapping."""

    cropped = pyqtSignal(object)

    def __init__(self) -> None:
        super().__init__()
        self._start: tuple[float, float] | None = None
        self.setToolTip(
            "Drag between two corners to crop. Ctrl+wheel zooms. Reset crop restores the image."
        )

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self._start = self.image_point_from_widget(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is not None and self._start is not None:
            end = self.image_point_from_widget(event.position().toPoint())
            if end is not None:
                x0, x1 = sorted((max(0, round(self._start[0])), max(0, round(end[0]))))
                y0, y1 = sorted((max(0, round(self._start[1])), max(0, round(end[1]))))
                if x1 > x0 and y1 > y0:
                    self.cropped.emit(
                        CropRect(x=x0, y=y0, width=x1 - x0, height=y1 - y0)
                    )
        self._start = None
        super().mouseReleaseEvent(event)


class SwingEditor(QDialog):
    """Save a reversible per-view recipe; video bytes are never written here."""

    def __init__(self, root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.root = root
        self.setWindowTitle("Edit swing · trim and crop")
        self.resize(850, 650)
        self._saved = load_edits(root)
        self._edits = dict(self._saved.views)
        self._entries = {e.view: e for e in load_bundle(root)[1].recordings if e.ok}
        if not self._entries:
            raise ValueError("This capture has no usable recordings")
        self._reader: VideoReader | None = None
        self._current = ""
        self._loading = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.view = QComboBox()
        self.view.addItems(list(self._entries))
        self.canvas = CropCanvas()
        self.canvas.cropped.connect(self.set_crop)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("Source frame")
        self.slider.valueChanged.connect(self._show_frame)
        self.first, self.last = QSpinBox(), QSpinBox()
        self.first.setAccessibleName("First included source frame")
        self.last.setAccessibleName("Last included source frame")
        self.mark_in, self.mark_out = QPushButton("Mark &in"), QPushButton("Mark &out")
        self.mark_in.clicked.connect(lambda: self.first.setValue(self.slider.value()))
        self.mark_out.clicked.connect(lambda: self.last.setValue(self.slider.value()))
        self.play = QPushButton("&Play selection")
        self.play.clicked.connect(self._toggle_play)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.exporter = SwingExportActions(
            self,
            root,
            view=lambda: self._current,
            save=self.save,
            status=self.status.setText,
        )
        self.clock = QLabel()
        self.crop_fields = {name: QSpinBox() for name in ("x", "y", "width", "height")}
        for name, field in self.crop_fields.items():
            field.setAccessibleName(f"Crop {name} in source pixels")
            field.valueChanged.connect(self._redraw)
        self._build()
        self.view.currentTextChanged.connect(self._switch_view)
        self._switch_view(self.view.currentText())
        # Explicit defaults are now the baseline; opening/closing is not an edit.
        self._baseline = self._recipe()
        styling.apply_theme(self)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        help_text = QLabel(
            "Keep just the swing: scrub, mark in/out, then drag a crop around the player. "
            "Frames are inclusive and numbered from 0. Originals and camera timing are preserved."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)
        layout.addWidget(self.view)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.slider)
        layout.addWidget(self.clock)
        marks = QHBoxLayout()
        for widget in (self.play, self.mark_in, self.first, self.mark_out, self.last):
            marks.addWidget(widget)
        layout.addLayout(marks)
        crop_row = QHBoxLayout()
        for name, field in self.crop_fields.items():
            label = QLabel(name.capitalize())
            label.setBuddy(field)
            crop_row.addWidget(label)
            crop_row.addWidget(field)
        reset_crop = QPushButton("Reset crop")
        reset_crop.clicked.connect(lambda: self.set_crop(None))
        crop_row.addWidget(reset_crop)
        layout.addLayout(crop_row)
        buttons = FlowLayout(spacing=6)
        reset = QPushButton("Reset this view")
        reset.clicked.connect(self.reset_view)
        save = QPushButton("&Save selection")
        save.setDefault(True)
        save.clicked.connect(self.save)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.add_widget(reset)
        references = QPushButton("Draw References…")
        references.clicked.connect(self._draw_references)
        buttons.add_widget(references)
        buttons.add_widget(self.exporter.button)
        buttons.add_widget(save)
        buttons.add_widget(close)
        layout.addWidget(self.status)
        layout.addLayout(buttons)

    def _draw_references(self) -> None:
        self._stop()
        if self.save():
            show_coaching(self.root, self, self._current)

    def current_edit(self) -> ViewEdit:
        crop = CropRect(
            **{name: field.value() for name, field in self.crop_fields.items()}
        )
        assert self._reader is not None
        full_image = (crop.x, crop.y, crop.width, crop.height) == (
            0,
            0,
            self._reader.width,
            self._reader.height,
        )
        edit = ViewEdit(
            first=self.first.value(),
            last=self.last.value(),
            crop=None if full_image else crop,
        )
        edit.validate_recording(self._entries[self._current])
        return edit

    def _recipe(self) -> SessionEdits:
        edits = dict(self._edits)
        if self._current:
            edits[self._current] = self.current_edit()
        return SessionEdits(views=edits)

    def _switch_view(self, name: str) -> None:
        self._stop()
        try:
            if self._current:
                self._edits[self._current] = self.current_edit()
            entry = self._entries[name]
            reader = VideoReader(
                self.root / entry.file
            )  # originals, never resized proxies
            if reader.frame_count <= 0:
                reader.close()
                raise ValueError("The recording has no decodable frames")
        except (ValueError, OSError) as exc:
            self.status.setText(str(exc))
            self.view.blockSignals(True)
            self.view.setCurrentText(self._current)
            self.view.blockSignals(False)
            if not self._current:
                raise
            return
        if self._reader:
            self._reader.close()
        self._reader, self._current = reader, name
        self._loading = True
        self.slider.setRange(0, reader.frame_count - 1)
        for field in (self.first, self.last):
            field.setRange(0, reader.frame_count - 1)
        for key, field in self.crop_fields.items():
            field.setRange(
                0 if key in ("x", "y") else 1,
                reader.width if key in ("x", "width") else reader.height,
            )
        edit = self._edits.get(name, ViewEdit())
        self.first.setValue(edit.first)
        self.last.setValue(
            edit.last if edit.last is not None else reader.frame_count - 1
        )
        self.set_crop(edit.crop)
        self._loading = False
        self.slider.setValue(edit.first)
        self._show_frame(self.slider.value())

    def set_crop(self, crop: CropRect | None) -> None:
        if self._reader is None:
            return
        crop = crop or CropRect(
            x=0, y=0, width=self._reader.width, height=self._reader.height
        )
        crop.validate_size(self._reader.width, self._reader.height)
        for name, field in self.crop_fields.items():
            field.blockSignals(True)
            field.setValue(getattr(crop, name))
            field.blockSignals(False)
        self._redraw()

    def reset_view(self) -> None:
        self.first.setValue(0)
        self.last.setValue(self.slider.maximum())
        self.set_crop(None)
        self.slider.setValue(0)

    def _redraw(self) -> None:
        if not self._loading:
            self._show_frame(self.slider.value())

    def _show_frame(self, index: int) -> None:
        if self._reader is None or self._loading:
            return
        import cv2

        frame = self._reader.read(index)
        if frame is None:
            self._stop()
            self.status.setText(f"Could not decode source frame {index}")
            return
        # A copied preview frame is marked; the inference image remains untouched.
        preview = frame.copy()
        x, y, w, h = (
            self.crop_fields[n].value() for n in ("x", "y", "width", "height")
        )
        colour = QColor(styling.signal_colors().ok)
        cv2.rectangle(
            preview,
            (x, y),
            (x + w - 1, y + h - 1),
            (colour.blue(), colour.green(), colour.red()),
            1,
        )
        self.canvas.set_image(preview)
        rate = self._entries[self._current].achieved_fps or self._reader.fps
        seconds = index / rate if rate > 0 else 0
        self.clock.setText(
            f"Source frame {index} / {self.slider.maximum()} · {seconds:.3f} s"
        )

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._stop()
            return
        try:
            edit = self.current_edit()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        assert self._reader is not None
        self.slider.setValue(edit.first)
        self._timer.start(max(1, round(1000 / (self._reader.fps or 30))))
        self.play.setText("Pause")

    def _advance(self) -> None:
        if self.slider.value() >= self.last.value():
            self._stop()
        else:
            self.slider.setValue(self.slider.value() + 1)

    def _stop(self) -> None:
        self._timer.stop()
        self.play.setText("&Play selection")

    def save(self) -> bool:
        try:
            recipe = self._recipe()
            save_edits(self.root, recipe)
        except (ValueError, OSError) as exc:
            self.status.setText(f"Selection not saved: {exc}")
            return False
        self._edits = dict(recipe.views)
        self._baseline = recipe
        self.status.setText(
            "Selection saved. Ingest will process these frames and crops."
        )
        return True

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        if not self.exporter.can_close():
            event.ignore()
            return
        try:
            dirty = self._recipe() != self._baseline
        except ValueError:
            dirty = True
        if dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved swing selection",
                "Save your selection before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if answer == QMessageBox.StandardButton.Cancel or (
                answer == QMessageBox.StandardButton.Save and not self.save()
            ):
                event.ignore()
                return
        self._stop()
        if self._reader:
            self._reader.close()
        event.accept()

    def reject(self) -> None:
        self.close()  # Escape follows the same unsaved-change policy.
