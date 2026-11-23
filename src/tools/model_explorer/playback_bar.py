"""Playback transport bar for Model Explorer results playback (#12996).

Provides an interactive playback control surface:
- Play / Pause button with background timer,
- Horizontal timeline slider mapped to the results frame count,
- Current simulation time readout and frame counter,
- Signal emission on user or automated scrub actions,
- An impact-zone slow-motion toggle (#15524) that stays off and disabled
  until the caller reports the run has an impact window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from src.tools.model_explorer.results_session import ResultsSession

logger = get_logger(__name__)

__all__ = ["PlaybackBar"]

PLAYBACK_INTERVAL_MS = 33  # ~30 FPS


class PlaybackBar(QWidget):
    """Transport controls for loaded results in the Model Explorer."""

    scrubbed = pyqtSignal(int)
    #: Emitted with the new state whenever impact-zone slow motion changes.
    impact_zone_slow_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._session: ResultsSession | None = None
        self._impact_available = False
        self._build_ui()

        # Playback timer parented so it terminates when the bar is destroyed.
        self._timer = QTimer(self)
        self._timer.setInterval(PLAYBACK_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)

        self.setEnabled(False)

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("PlaybackPlayButton")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play_toggled)
        layout.addWidget(self.play_button)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("PlaybackSlider")
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.slider, 1)

        self.impact_slow_button = QPushButton("Slow impact")
        self.impact_slow_button.setObjectName("PlaybackImpactSlowButton")
        self.impact_slow_button.setCheckable(True)
        self.impact_slow_button.toggled.connect(self.impact_zone_slow_toggled)
        layout.addWidget(self.impact_slow_button)
        self._refresh_impact_button()

        self.time_label = QLabel("t = 0.000 s (0/0)")
        self.time_label.setObjectName("PlaybackTimeLabel")
        self.time_label.setMinimumWidth(150)
        layout.addWidget(self.time_label)

    def set_impact_zone_available(self, available: bool) -> None:
        """Declare whether the loaded run has an impact window to slow down.

        Invariant: the toggle is checked only while it is enabled. Losing
        availability while it is on unchecks it and emits ``False`` once, so
        the consumer's clock never stays slow. Idempotent.
        """
        self._impact_available = bool(available)
        self._refresh_impact_button()

    def _refresh_impact_button(self) -> None:
        playable = self._session is not None and self._session.frame_count >= 2
        usable = self._impact_available and playable
        button = self.impact_slow_button
        button.setEnabled(usable)
        button.setToolTip(
            "Slow playback around the ball impact"
            if usable
            else "No qualified impact window in this run"
        )
        if not usable and button.isChecked():
            button.setChecked(False)  # emits impact_zone_slow_toggled(False)

    def set_session(self, session: ResultsSession | None) -> None:
        """Configure transport controls for ``session``, or reset when ``None``."""
        self._session = session
        self._refresh_impact_button()
        if session is None or session.frame_count < 2:
            self.stop()
            self.slider.setRange(0, 0)
            self.time_label.setText("t = 0.000 s (0/0)")
            self.setEnabled(False)
            return

        self.setEnabled(True)
        self.slider.blockSignals(True)
        self.slider.setRange(0, session.frame_count - 1)
        self.slider.setValue(session.current_frame)
        self.slider.blockSignals(False)
        self.update_time_label(session.current_frame)

    def show_frame(self, index: int) -> None:
        """Update slider and time readout to reflect playback at ``index``."""
        if self._session is None:
            return
        if not (0 <= index < self._session.frame_count):
            raise ValueError(
                f"frame {index} is outside range [0, {self._session.frame_count - 1}]"
            )
        if self.slider.value() != index:
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)
        self.update_time_label(index)

    def update_time_label(self, index: int) -> None:
        """Format and update the current time and frame index text."""
        if self._session is None:
            return
        time_s = self._session.current_time_s
        self.time_label.setText(
            f"t = {time_s:.3f} s  ({index + 1}/{self._session.frame_count})"
        )

    def _on_slider_value_changed(self, value: int) -> None:
        self.scrubbed.emit(value)

    def _on_play_toggled(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self._timer.start()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        """Step playback forward by one frame, looping at the boundary."""
        if self._session is None or self._session.frame_count <= 0:
            self.stop()
            return
        next_frame = (self._session.current_frame + 1) % self._session.frame_count
        self.scrubbed.emit(next_frame)

    def stop(self) -> None:
        """Halt playback and reset play button state. Idempotent."""
        self._timer.stop()
        if self.play_button.isChecked():
            self.play_button.blockSignals(True)
            self.play_button.setChecked(False)
            self.play_button.setText("Play")
            self.play_button.blockSignals(False)
