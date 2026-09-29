"""Guided annotation dialog: click the joint the banner asks for (#9800, #9803).

:class:`ImageCanvas` shows one frame at a zoom and maps a click back to
image pixels exactly (letterboxing and zoom inverted). :class:`AnnotateDialog`
drives a :class:`Guide` over a :class:`AnnotationSet`: the banner says which
frame and joint, a click accepts, ``S`` skips (occluded), ``B`` goes back,
``N`` moves to the next frame, ``J`` jumps, ``A`` accepts the whole frame
(edit mode), ``Q`` finishes. In **edit mode** the dialog is opened over an
observation set: the detector's points are drawn, and a click replaces one
while a skip rejects it; the corrections are saved as a sparse layer that
``rig annotations-to-observations --merge-with`` applies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.annotate import AnnotationSet, Guide, annotation_path
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from src.shared.python.core.contracts import require

from .overlay import PoseTrack, draw_pose
from .player import VideoReader, clamp_index

CORRECTION_COLOUR = (60, 255, 60)
INTERPOLATED_COLOUR = (200, 200, 60)
CURRENT_COLOUR = (255, 120, 255)
ZOOM_STEP = 1.25
MAX_ZOOM = 8.0


def bgr_to_pixmap(frame_bgr: npt.NDArray[np.uint8]) -> QPixmap:
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    image = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image)


class ImageCanvas(QLabel):
    """A frame at a zoom; clicks come back in image pixels."""

    clicked = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 200)
        self._image: npt.NDArray[np.uint8] | None = None
        self._zoom = 1.0
        self._drawn: tuple[int, int] = (0, 0)  # drawn pixmap size
        self._source_pixmap = QPixmap()
        self._pan = QPointF()
        self._pan_mode = False
        self._drag: QPointF | None = None

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_image(self, frame_bgr: npt.NDArray[np.uint8]) -> None:
        if self._image is None or self._image.shape != frame_bgr.shape:
            self._pan = QPointF()
        self._image = np.ascontiguousarray(frame_bgr, dtype=np.uint8)
        self._source_pixmap = bgr_to_pixmap(self._image)
        self._render()

    def set_pan_mode(self, active: bool) -> None:
        """Use left-drag navigation instead of marking; middle-drag also pans."""
        self._pan_mode = active
        self._drag = None
        self.setCursor(
            Qt.CursorShape.OpenHandCursor if active else Qt.CursorShape.ArrowCursor
        )

    def pan_by(self, dx: float, dy: float) -> None:
        """Move the image in viewport pixels, bounded by its visible edges."""
        self._pan += QPointF(dx, dy)
        self._clamp_pan()
        self.update()

    def _clamp_pan(self) -> None:
        dw, dh = self._drawn
        max_x, max_y = (
            max(0.0, (dw - self.width()) / 2),
            max(0.0, (dh - self.height()) / 2),
        )
        self._pan = QPointF(
            max(-max_x, min(max_x, self._pan.x())),
            max(-max_y, min(max_y, self._pan.y())),
        )

    def set_zoom(self, factor: float) -> None:
        """Precondition: ``0 < factor <= MAX_ZOOM``; 1.0 fits the widget."""
        require(0 < factor <= MAX_ZOOM, "zoom out of range", factor)
        self._zoom = float(factor)
        self._render()

    def _fit_size(self) -> tuple[int, int]:
        """Pixmap size at zoom 1 (fit inside the widget, aspect kept)."""
        assert self._image is not None
        h, w = self._image.shape[:2]
        scale = min(self.width() / w, self.height() / h)
        return max(1, int(w * scale)), max(1, int(h * scale))

    def _render(self) -> None:
        if self._image is None:
            return
        fit_w, fit_h = self._fit_size()
        w, h = int(fit_w * self._zoom), int(fit_h * self._zoom)
        self._drawn = (w, h)
        self._clamp_pan()
        self.update()

    def paintEvent(self, event: QPaintEvent | None) -> None:  # noqa: N802 - Qt
        super().paintEvent(event)
        if self._source_pixmap.isNull():
            return
        dw, dh = self._drawn
        target = QRectF(
            (self.width() - dw) / 2 + self._pan.x(),
            (self.height() - dh) / 2 + self._pan.y(),
            dw,
            dh,
        )
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # Qt clips to the viewport; zoom never allocates an enlarged bitmap.
        painter.drawPixmap(
            target, self._source_pixmap, QRectF(self._source_pixmap.rect())
        )
        painter.end()

    def image_point_from_widget(self, pos: QPoint) -> tuple[float, float] | None:
        """Image pixel under a widget position, or ``None`` outside the image."""
        if self._image is None or self._drawn == (0, 0):
            return None
        dw, dh = self._drawn
        x0 = (self.width() - dw) / 2 + self._pan.x()
        y0 = (self.height() - dh) / 2 + self._pan.y()
        u, v = pos.x() - x0, pos.y() - y0
        if not (0 <= u < dw and 0 <= v < dh):
            return None
        h, w = self._image.shape[:2]
        return (u + 0.5) * w / dw - 0.5, (v + 0.5) * h / dh - 0.5

    def mousePressEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802 - Qt
        if event is not None and (
            event.button() == Qt.MouseButton.MiddleButton
            or (self._pan_mode and event.button() == Qt.MouseButton.LeftButton)
        ):
            self._drag = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            point = self.image_point_from_widget(event.position().toPoint())
            if point is not None:
                self.clicked.emit(*point)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802 - Qt
        if event is not None and self._drag is not None:
            delta = event.position() - self._drag
            self.pan_by(delta.x(), delta.y())
            self._drag = event.position()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802 - Qt
        if self._drag is not None:
            self.set_pan_mode(self._pan_mode)
            if event is not None:
                event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent | None) -> None:  # noqa: N802 - Qt
        if event is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            step = ZOOM_STEP if event.angleDelta().y() > 0 else 1 / ZOOM_STEP
            self.set_zoom(min(MAX_ZOOM, max(1.0, self._zoom * step)))
            event.accept()
            return
        super().wheelEvent(event)

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        self._render()
        super().resizeEvent(event)  # type: ignore[arg-type]


@dataclass(frozen=True)
class AnnotateSettings:
    """Which joints to ask for, over which frames, at which stride, by whom."""

    joints: Sequence[str] = JOINT_NAMES
    frame_range: tuple[int, int] | None = None
    stride: int = 1
    annotator: str = "unknown"


@dataclass(frozen=True)
class BaseSet:
    """An observation set to correct: its name and the view's file (#9803)."""

    name: str
    file: Path


class AnnotateDialog(QDialog):
    """Frame-by-frame, joint-by-joint clicking over one view."""

    def __init__(
        self,
        session: Path,
        view: str,
        video: Path,
        *,
        settings: AnnotateSettings | None = None,
        base: BaseSet | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        settings = settings or AnnotateSettings()
        base_set = base.name if base else None
        self.session, self.view = session, view
        self.reader = VideoReader(video)
        self.path = annotation_path(session, view)
        self.store = (
            AnnotationSet.load(self.path)
            if self.path.is_file()
            else AnnotationSet(
                view,
                self.reader.width,
                self.reader.height,
                self.reader.fps or 30.0,
                annotator=settings.annotator,
                base_set=base_set,
            )
        )
        if base_set and self.store.base_set is None:
            self.store.base_set = base_set
        self.base: PoseTrack | None = PoseTrack.load(base.file) if base else None
        last = max(self.reader.frame_count - 1, 0)
        first, stop = settings.frame_range or (0, last)
        self.guide = Guide(
            self.store,
            tuple(settings.joints),
            (first, min(stop, last)),
            settings.stride,
            only_missing=self.base is None,
        )
        self.dirty = False
        self.setWindowTitle(
            f"Annotate {view}" + (f" · editing {base_set}" if base_set else "")
        )
        self._build()
        self._refresh()

    # -- layout -----------------------------------------------------------------
    def _build(self) -> None:
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.canvas = ImageCanvas()
        self.canvas.clicked.connect(self._on_click)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, max(self.reader.frame_count - 1, 0))
        self.slider.valueChanged.connect(self._on_slider)
        self.joint_list = QListWidget()
        self.progress = QLabel()
        buttons = QHBoxLayout()
        self.buttons: dict[str, QPushButton] = {}
        for key, label in (
            ("skip", "Skip (S)"),
            ("back", "Back (B)"),
            ("next_frame", "Next frame (N)"),
            ("accept_frame", "Accept frame (A)"),
            ("jump", "Jump… (J)"),
            ("finish", "Finish (Q)"),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, k=key: self.act(k))
            buttons.addWidget(button)
            self.buttons[key] = button
        self.buttons["accept_frame"].setEnabled(self.base is not None)
        left = QVBoxLayout()
        left.addWidget(self.banner)
        left.addWidget(self.canvas, 1)
        left.addWidget(self.slider)
        left.addLayout(buttons)
        right = QVBoxLayout()
        right.addWidget(QLabel("Coverage"))
        right.addWidget(self.joint_list, 1)
        right.addWidget(self.progress)
        layout = QHBoxLayout(self)
        layout.addLayout(left, 4)
        layout.addLayout(right, 1)
        self.resize(1200, 760)

    # -- actions ------------------------------------------------------------------
    def act(self, key: str) -> None:
        """Dispatch a button/key action; ``jump`` asks for a frame."""
        if key == "jump":
            frame, ok = QInputDialog.getInt(
                self, "Jump", "Frame", self.guide.frame, 0, self.reader.frame_count - 1
            )
            if ok:
                self.guide.jump(frame)
        elif key == "accept_frame":
            self._accept_frame()
        elif key == "finish":
            self.finish()
            return
        elif not self.guide.finished:
            getattr(self.guide, key)()
            self.dirty = True
        else:
            return
        self._refresh()

    def _on_click(self, x: float, y: float) -> None:
        if self.guide.finished:
            return
        self.guide.accept(x, y)
        self.dirty = True
        self._refresh()

    def _accept_frame(self) -> None:
        """Edit mode: keep the detector's points for the rest of this frame."""
        if self.guide.finished:
            return
        frame = self.guide.frame
        while not self.guide.finished and self.guide.frame == frame:
            self.guide.next_frame()

    def _on_slider(self, value: int) -> None:
        if value != self.guide.frame and not self.guide.finished:
            self.guide.jump(value)
            self._refresh()

    def keyPressEvent(self, event: QKeyEvent | None) -> None:  # noqa: N802 - Qt
        if event is None:
            return
        text = event.text().lower()
        if text == "j":
            self.act("jump")
        elif text == "a":
            self.act("accept_frame")
        elif text == "q":
            self.act("finish")
        elif text and self.guide.handle_key(text):
            self.dirty = True
            self._refresh()
        else:
            super().keyPressEvent(event)

    def finish(self) -> None:
        self.save()
        self.accept()

    def save(self) -> Path:
        self.store.save(self.path, base=self.session)
        self.dirty = False
        return self.path

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        if self.dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved annotations",
                "Save before closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.save()
        self.reader.close()
        super().closeEvent(event)  # type: ignore[arg-type]

    # -- rendering ----------------------------------------------------------------
    def _refresh(self) -> None:
        prompt = self.guide.prompt()
        frame_index = self.guide.frame if prompt else self.slider.value()
        frame_index = clamp_index(frame_index, self.reader.frame_count)
        if prompt is not None:
            text = prompt.text
            if self.base is not None:
                text = text.replace("click the joint", "click to correct").replace(
                    "S skip (occluded)", "S reject, A accept frame"
                )
                conf = self._base_confidence(frame_index, prompt.joint)
                if conf is not None:
                    text = text.replace(" — ", f" (detector {conf:.2f}) — ", 1)
            self.banner.setText(text)
        else:
            self.banner.setText("Finished — press Finish to save and close.")
        image = self.reader.read(frame_index)
        if image is not None:
            self.canvas.set_image(self._decorated(image, frame_index, prompt))
        self.slider.blockSignals(True)
        self.slider.setValue(frame_index)
        self.slider.blockSignals(False)
        done, total = self.guide.progress()
        self.progress.setText(f"{done}/{total} prompts · {self.store.count()} points")
        self.joint_list.clear()
        for joint, cov in self.store.coverage().items():
            self.joint_list.addItem(
                f"{joint}: {cov['annotated']} pts, {cov['skipped']} skipped"
            )

    def _base_confidence(self, frame: int, joint: str) -> float | None:
        if self.base is None or joint not in self.base.names:
            return None
        pose = self.base.at(frame)
        names = self.base.names
        return None if pose is None else float(pose[1][names.index(joint)])

    def _decorated(
        self, image: npt.NDArray[np.uint8], frame: int, prompt: object
    ) -> npt.NDArray[np.uint8]:
        out = image
        if self.base is not None:
            pose = self.base.at(frame)
            if pose is not None:
                out = draw_pose(
                    out, pose[0], pose[1], self.base.edges, min_confidence=0.5
                )
        points, conf = self._layer_points(frame)
        if points.size:
            out = draw_pose(out, points, conf, (), point_colour=CORRECTION_COLOUR)
        current = getattr(prompt, "joint", None)
        guess = self.store.interpolate(frame, current) if current else None
        if guess is not None and guess.interpolated:
            out = draw_pose(
                out,
                np.array([[guess.x_px, guess.y_px]]),
                np.ones(1),
                point_colour=INTERPOLATED_COLOUR,
                thickness=1,
            )
        return out

    def _layer_points(
        self, frame: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        points = self.store.points_at(frame)
        if not points:
            return np.zeros((0, 2)), np.zeros(0)
        arr = np.array([[p.x_px, p.y_px] for p in points.values()])
        return arr, np.ones(len(arr))
