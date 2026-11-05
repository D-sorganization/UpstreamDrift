"""Playback controls for the Putting Green Simulator (#14473, epic #14452).

Play/pause, the scrubber, the phase jumps and the camera view presets, on two
lines so the phone width still fits. The bar only holds the controls; the
simulator owns the timer and the camera and listens to the bar's signals.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.putting_dynamics import Mode
from src.tools.putting_green_gui._camera import CAMERA_PRESETS
from src.tools.putting_green_gui._scene_builder import PuttScene

_VIEW_LABELS = {
    "orbit": "Orbit",
    "down_the_line": "Down the line",
    "face_on": "Face on",
    "overhead": "Overhead",
}
_PHASE_LABELS = {
    Mode.AIRBORNE: "Hop",
    Mode.SLIDE: "Skid",
    Mode.ROLL: "Roll",
    Mode.REST: "Rest",
}


class PlaybackBar(QWidget):
    """View presets, play/pause, the scrubber and the phase jumps.

    ``play_toggled`` fires on Play/Pause, ``scrubbed`` with the sample a
    player scrubbed or jumped to, and ``view_changed`` on a new view.
    """

    play_toggled = pyqtSignal()
    scrubbed = pyqtSignal(int)
    view_changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        transport, choices = QHBoxLayout(), QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.setToolTip("Play or pause the putt")
        self.play_btn.clicked.connect(self.play_toggled.emit)
        self.scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self.scrub_slider.setRange(0, 0)
        self.scrub_slider.setToolTip("Scrub through the putt")
        self.scrub_slider.valueChanged.connect(self.scrubbed.emit)
        self.phase_combo = QComboBox()
        self.phase_combo.setToolTip("Jump to where the hop, skid, roll or rest starts")
        self.phase_combo.activated.connect(self._jump_to_phase)
        self.view_combo = QComboBox()
        self.view_combo.setToolTip("Camera view")
        for preset in CAMERA_PRESETS:
            self.view_combo.addItem(_VIEW_LABELS[preset], preset)
        self.view_combo.currentIndexChanged.connect(self.view_changed.emit)
        transport.addWidget(self.play_btn)
        transport.addWidget(self.scrub_slider, stretch=1)
        choices.addWidget(self.phase_combo, stretch=1)
        choices.addWidget(self.view_combo, stretch=1)
        for combo in (self.phase_combo, self.view_combo):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(3)
        layout.addLayout(transport)
        layout.addLayout(choices)

    def load(self, scene: PuttScene) -> None:
        """Reset the scrubber to the start of ``scene`` and list its phases."""
        self.follow(0)
        self.scrub_slider.blockSignals(True)
        self.scrub_slider.setRange(0, scene.trajectory_xyz.shape[0] - 1)
        self.scrub_slider.blockSignals(False)
        self.phase_combo.clear()
        for mark in scene.phases:
            label = f"{_PHASE_LABELS[mark.mode]} {mark.t_s:.2f} s"
            self.phase_combo.addItem(label, mark.index)

    def follow(self, index: int) -> None:
        """Move the scrubber with the playback; the echo is not a scrub."""
        self.scrub_slider.blockSignals(True)
        self.scrub_slider.setValue(index)
        self.scrub_slider.blockSignals(False)

    def set_playing(self, playing: bool) -> None:
        self.play_btn.setText("Pause" if playing else "Play")

    def view(self) -> str:
        """The chosen camera preset (a key of ``CAMERA_PRESETS``)."""
        return str(self.view_combo.currentData())

    def _jump_to_phase(self, combo_index: int) -> None:
        index = int(self.phase_combo.itemData(combo_index))
        self.follow(index)
        self.scrubbed.emit(index)


__all__ = ["PlaybackBar"]
