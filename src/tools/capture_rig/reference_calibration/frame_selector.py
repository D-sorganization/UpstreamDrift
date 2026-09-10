"""Choose a calibration frame using familiar video transport and full-screen controls."""

from pathlib import Path

from PyQt6.QtCore import QByteArray, QElapsedTimer, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..annotate_widget import ImageCanvas
from ..view_windows import window_shortcuts
from .video_preview import PreviewFrame, VideoPreview


class _PreviewCanvas(ImageCanvas):
    fullscreen_requested = pyqtSignal()

    def mouseDoubleClickEvent(self, event: QMouseEvent | None) -> None:  # noqa: N802
        if event is not None and event.button() == Qt.MouseButton.LeftButton:
            self.fullscreen_requested.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


class ReferenceFrameSelector(QDialog):
    """Preview original video; only explicit acceptance selects evidence to archive."""

    def __init__(
        self,
        path: Path,
        *,
        context: str,
        initial_index: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.selected_frame: int | None = None
        self.displayed_index: int | None = None
        self._requested = initial_index
        self._frame_count = 0
        self._fps = 30.0
        self._playing = False
        self._selectable = False
        self._play_clock = QElapsedTimer()
        self._play_start = 0
        self._closed = False
        self._geometry = QByteArray()
        self.setWindowTitle("Choose a Reference Frame")
        self.resize(850, 680)
        self._build(context)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)
        self.loader = VideoPreview(path, self)
        self.loader.ready.connect(self._display)
        self.loader.failed.connect(self._failed)
        self.finished.connect(self._finish)
        window_shortcuts(self, self.toggle_fullscreen, self._escape)
        self._seek(initial_index)

    def _build(self, context: str) -> None:
        layout = QVBoxLayout(self)
        heading = QLabel(context)
        heading.setTextFormat(Qt.TextFormat.PlainText)
        heading.setWordWrap(True)
        layout.addWidget(heading)
        hint = QLabel(
            "Choose a clear, stationary reference. The next screen marks original pixels. Double-click video or press F11 for full screen."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.canvas = _PreviewCanvas()
        self.canvas.fullscreen_requested.connect(self.toggle_fullscreen)
        layout.addWidget(self.canvas, 1)
        self.clock = QLabel("Loading Original Video…")
        layout.addWidget(self.clock)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("Original Video Timeline")
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._scrub)
        layout.addWidget(self.slider)
        row = QHBoxLayout()
        previous, following = QPushButton("Previous"), QPushButton("Next")
        previous.clicked.connect(lambda: self._scrub(max(0, self._requested - 1)))
        following.clicked.connect(
            lambda: self._scrub(min(self._frame_count - 1, self._requested + 1))
        )
        self.play = QPushButton("Play")
        self.play.setEnabled(False)
        self.play.clicked.connect(self._toggle_play)
        self.frame = QSpinBox()
        self.frame.setRange(0, 2_147_483_647)
        self.frame.setAccessibleName("Original Frame Number")
        self.frame.setKeyboardTracking(False)
        self.frame.valueChanged.connect(self._scrub)
        for widget in (previous, self.play, following, QLabel("Frame"), self.frame):
            row.addWidget(widget)
        layout.addLayout(row)
        self.status = QLabel("Reading the original recording…")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        footer = QHBoxLayout()
        fullscreen = QPushButton("Full Screen")
        fullscreen.clicked.connect(self.toggle_fullscreen)
        self.use = QPushButton("Use This Frame")
        self.use.setEnabled(False)
        self.use.clicked.connect(self._choose)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        for button in (fullscreen, self.use, cancel):
            button.setAutoDefault(False)
            footer.addWidget(button)
        layout.addLayout(footer)

    def _seek(self, index: int) -> None:
        self._requested = max(0, index)
        self._selectable = False
        self.use.setEnabled(False)
        self.status.setText(f"Loading original frame {self._requested}…")
        self.loader.request(self._requested)

    def _scrub(self, index: int) -> None:
        self._pause()
        self._seek(index)

    def _display(self, packet: PreviewFrame) -> None:
        if self._closed:
            return
        self.displayed_index = packet.index
        self._frame_count, self._fps = packet.frame_count, packet.fps
        self.canvas.set_image(packet.image)
        self.clock.setText(
            f"Frame {packet.index} of {packet.frame_count - 1} · {packet.index / packet.fps:.3f} s · {packet.fps:g} fps"
        )
        for control in (self.slider, self.frame):
            control.blockSignals(True)
            control.setRange(0, packet.frame_count - 1)
            control.setValue(self._requested)
            control.blockSignals(False)
        self.slider.setEnabled(True)
        self.play.setEnabled(True)
        current = packet.index == self._requested
        self._selectable = current
        self.use.setEnabled(current and not self._playing)
        if current:
            self.status.setText(
                "Choose Use This Frame to mark the reference, or continue scrubbing."
            )
        if self._playing and current:
            self._timer.start(max(33, round(1000 / self._fps)))

    def _failed(self, message: str) -> None:
        if not self._closed:
            self._pause()
            self._selectable = False
            self.use.setEnabled(False)
            self.status.setText(
                f"{message}. Choose another frame, or cancel and restore the original file in Capture Library."
            )

    def _pause(self) -> None:
        self._playing = False
        self._timer.stop()
        self.play.setText("Play")

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause()
            self.use.setEnabled(self._selectable)
        else:
            self._playing = True
            self.play.setText("Pause")
            self.use.setEnabled(False)
            self._play_start = (
                self._requested if self._requested < self._frame_count - 1 else 0
            )
            self._play_clock.start()
            if self._requested >= self._frame_count - 1:
                self._seek(0)
            else:
                self._advance()

    def _advance(self) -> None:
        if self._requested >= self._frame_count - 1:
            self._pause()
            self.use.setEnabled(self._selectable)
        else:
            source_index = self._play_start + int(
                self._play_clock.elapsed() * self._fps / 1000
            )
            self._seek(
                min(self._frame_count - 1, max(self._requested + 1, source_index))
            )

    def _choose(self) -> None:
        if self.use.isEnabled() and self.displayed_index == self._requested:
            self.selected_frame = self.displayed_index
            self.accept()

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.leave_fullscreen()
        else:
            self._geometry = self.saveGeometry()
            self.showFullScreen()

    def leave_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.restoreGeometry(self._geometry)

    def _escape(self) -> None:
        if self.isFullScreen():
            self.leave_fullscreen()
        else:
            self.reject()

    def _finish(self, _result: int) -> None:
        self._closed = True
        self._pause()
        self.loader.close()
