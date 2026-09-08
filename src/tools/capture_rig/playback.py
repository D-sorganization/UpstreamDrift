"""Frame-accurate playback of one view with the pose and model overlays.

:class:`PlaybackPanel` shows one view and one observation set at a time,
draws the detector's pose (:mod:`.overlay`) and the selected variants'
projected models (:mod:`.overlay_render`) on each frame, and steps through
the recording at its own frame rate. Moved out of :mod:`.gui` (#9816) so the
tile module stays within the file-size budget; ``gui.PlaybackPanel`` remains
importable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.theme.layout_metrics import LayoutMetrics

from .overlay import PoseTrack, draw_pose
from .overlay_box import VariantOverlayBox
from .overlay_render import render_frame
from .player import VideoReader, clamp_index
from .session import SessionMedia, ViewMedia

PLAYBACK_MIN_SIZE = (320, 200)


class PlaybackPanel(QWidget):
    """One view and one observation set at a time, frame-accurate, with overlay."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._reader: VideoReader | None = None
        self._track: PoseTrack | None = None
        self._index = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.step)
        self.view_combo = QComboBox()
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        self.set_combo = QComboBox()
        self.set_combo.currentIndexChanged.connect(self._on_set_changed)
        self.overlay_check = QCheckBox("pose overlay")
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(lambda _: self.show_frame(self._index))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.valueChanged.connect(
            lambda _: self.show_frame(self._index)
        )
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.variants = VariantOverlayBox()
        self.variants.changed.connect(lambda: self.show_frame(self._index))
        self.image = QLabel("no session loaded")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(*PLAYBACK_MIN_SIZE)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        self.status = QLabel("")
        top = QHBoxLayout()
        top.setSpacing(LayoutMetrics.SPACING_SM)
        top.addWidget(QLabel("View"))
        top.addWidget(self.view_combo, 1)
        top.addWidget(QLabel("Set"))
        top.addWidget(self.set_combo, 1)
        top.addWidget(self.overlay_check)
        top.addWidget(QLabel("min conf"))
        top.addWidget(self.confidence_spin)
        top.addWidget(self.play_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LayoutMetrics.SPACING_SM)
        layout.addLayout(top)
        layout.addWidget(self.variants)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.slider)
        layout.addWidget(self.status)

    @property
    def frame_index(self) -> int:
        return self._index

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def load(self, media: SessionMedia) -> None:
        """Offer every playable view; select the first."""
        self.close_media()
        self.variants.load(media)
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        for view in media.views:
            if view.playable is not None:
                self.view_combo.addItem(view.view, view)
        self.view_combo.blockSignals(False)
        if self.view_combo.count():
            self._on_view_changed(0)
        else:
            self.image.setText("no playable recording in this session")

    def _current_view(self) -> ViewMedia | None:
        return self.view_combo.currentData()

    def current_view_name(self) -> str | None:
        view = self._current_view()
        return None if view is None else view.view

    def current_set_name(self) -> str | None:
        text = self.set_combo.currentText()
        return text or None

    def _on_view_changed(self, index: int) -> None:
        view: ViewMedia | None = self.view_combo.itemData(index)
        if view is None or view.playable is None:
            return
        self.close_media()
        self._reader = VideoReader(view.playable)
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for name, path in (view.observation_sets or {}).items():
            self.set_combo.addItem(name, path)
        if not self.set_combo.count() and view.observations is not None:
            self.set_combo.addItem("observations", view.observations)
        self.set_combo.blockSignals(False)
        self._load_track()
        self.slider.setRange(0, max(self._reader.frame_count - 1, 0))
        rate = self._reader.fps or view.fps or 30.0
        self._timer.setInterval(max(int(1000.0 / rate), 1))
        self.show_frame(0)

    def _on_set_changed(self, _index: int) -> None:
        self._load_track()
        self.show_frame(self._index)

    def _load_track(self) -> None:
        path: Path | None = self.set_combo.currentData()
        self._track = PoseTrack.load(path) if path is not None else None

    def close_media(self) -> None:
        self._timer.stop()
        self.play_button.setText("Play")
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        self._track = None

    def toggle_play(self) -> None:
        if self._reader is None:
            return
        if self.playing:
            self._timer.stop()
            self.play_button.setText("Play")
        else:
            self._timer.start()
            self.play_button.setText("Pause")

    def step(self) -> None:
        if self._reader is None:
            return
        nxt = self._index + 1
        if nxt >= self._reader.frame_count:
            self.toggle_play()
            return
        self.slider.setValue(nxt)

    def show_frame(self, index: int) -> None:
        """Draw frame ``index`` (clamped) with the overlay when one exists."""
        if self._reader is None:
            return
        self._index = clamp_index(index, self._reader.frame_count)
        frame = self._reader.read(self._index)
        if frame is None:
            return
        pose = self._track.at(self._index) if self._track else None
        if pose is not None and self.overlay_check.isChecked():
            frame = draw_pose(
                frame,
                pose[0],
                pose[1],
                self._track.edges if self._track else (),
                min_confidence=float(self.confidence_spin.value()),
            )
        view_name = self.current_view_name()
        tracks = self.variants.tracks_for(view_name) if view_name else ()
        if tracks:
            frame = render_frame(frame, tracks, self._index)
        self._blit(frame)
        self.slider.blockSignals(True)
        self.slider.setValue(self._index)
        self.slider.blockSignals(False)
        detected = "pose" if pose is not None else "no pose"
        total = self._reader.frame_count
        extra = f" · {self.variants.error}" if self.variants.error else ""
        self.status.setText(f"frame {self._index + 1}/{total} · {detected}{extra}")

    def _blit(self, frame_bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        image = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.image.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image.setPixmap(pixmap)
