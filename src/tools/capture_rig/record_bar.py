"""Transport-style recording controls for the Capture Rig tile.

:class:`RecordingClock` is the Qt-free state machine: idle → countdown →
recording → idle, driven by ``tick(now)``. :class:`RecordBar` renders it as
a video-interface control strip: one big Record/Stop button, one-click
duration presets plus a custom spinner, a countdown selector, a blinking
REC readout with elapsed/remaining time and a progress bar. The bar emits
``start_requested`` when the countdown ends (the tile then launches the
recorder and calls :meth:`RecordBar.recording_started`) and
``stop_requested`` when the operator ends a take early. ``badge_changed``
carries the short text the preview tiles stamp on their frames.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.core.contracts import require
from src.shared.python.theme.layout_metrics import LayoutMetrics

from . import styling
from .flow_layout import FlowLayout

DURATION_PRESETS_S: tuple[float, ...] = (5.0, 10.0, 15.0, 30.0)
COUNTDOWN_CHOICES_S: tuple[float, ...] = (0.0, 3.0, 5.0, 10.0)
TICK_MS = 100
BLINK_PERIOD_S = 1.0


class Phase(str, Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"


def clock_text(seconds: float) -> str:
    """``MM:SS`` for a non-negative number of seconds (floor)."""
    require(seconds >= 0, "seconds must be non-negative", seconds)
    whole = int(seconds)
    return f"{whole // 60:02d}:{whole % 60:02d}"


@dataclass
class RecordingClock:
    """Countdown then recording, advanced by :meth:`tick` with a monotonic time.

    Invariants: ``duration_s > 0``; ``countdown_s >= 0``; ``elapsed`` and
    ``remaining`` are non-negative; ``progress`` is within ``[0, 1]``.
    """

    duration_s: float = 10.0
    countdown_s: float = 0.0
    phase: Phase = Phase.IDLE
    _started: float = field(default=0.0, repr=False)
    _now: float = field(default=0.0, repr=False)

    def arm(self, duration_s: float, countdown_s: float, now: float) -> Phase:
        """Start the countdown (or, with none, wait for :meth:`recording_started`)."""
        require(self.phase is Phase.IDLE, "clock must be idle to arm", self.phase)
        require(duration_s > 0, "duration must be positive", duration_s)
        require(countdown_s >= 0, "countdown must be non-negative", countdown_s)
        self.duration_s, self.countdown_s = duration_s, countdown_s
        self._started = self._now = now
        self.phase = Phase.COUNTDOWN
        return self.phase

    def tick(self, now: float) -> Phase:
        """Advance to ``now``; the countdown ends by itself, recording does not."""
        require(now >= self._now, "time must not run backwards", (now, self._now))
        self._now = now
        return self.phase

    def countdown_done(self) -> bool:
        """True once the armed countdown has fully elapsed."""
        return (
            self.phase is Phase.COUNTDOWN
            and self._now - self._started >= self.countdown_s
        )

    def recording_started(self, now: float) -> None:
        """The recorder is running; the take clock starts here."""
        require(self.phase is Phase.COUNTDOWN, "arm before recording", self.phase)
        self._started = self._now = now
        self.phase = Phase.RECORDING

    def reset(self) -> None:
        self.phase = Phase.IDLE

    # -- readings ---------------------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return (
            max(0.0, self._now - self._started) if self.phase is not Phase.IDLE else 0.0
        )

    @property
    def remaining_countdown(self) -> float:
        if self.phase is not Phase.COUNTDOWN:
            return 0.0
        return max(0.0, self.countdown_s - self.elapsed)

    @property
    def remaining(self) -> float:
        if self.phase is not Phase.RECORDING:
            return self.duration_s
        return max(0.0, self.duration_s - self.elapsed)

    @property
    def progress(self) -> float:
        if self.phase is not Phase.RECORDING:
            return 0.0
        return min(1.0, self.elapsed / self.duration_s)

    def badge(self) -> str:
        """Short text for the preview tiles ('' when idle)."""
        if self.phase is Phase.COUNTDOWN:
            left = self.remaining_countdown
            return f"Starting in {math.ceil(left)}" if left > 0 else "Starting…"
        if self.phase is Phase.RECORDING:
            return f"● REC {clock_text(self.elapsed)} / {clock_text(self.duration_s)}"
        return ""


class RecordBar(QWidget):
    """Record/Stop, duration presets, countdown, REC readout and progress."""

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    badge_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        now: type(time.monotonic) | None = None,  # type: ignore[valid-type]
    ) -> None:
        super().__init__(parent)
        self._now = now or time.monotonic
        self.clock = RecordingClock()
        self._badge = ""
        self.record_button = QPushButton("●  Record")
        self.record_button.setMinimumHeight(LayoutMetrics.TRANSPORT_BUTTON_HEIGHT)
        self.record_button.setToolTip(
            "Start a take of the chosen duration (after the countdown, if any). "
            "While recording this button stops the take early."
        )
        self.record_button.clicked.connect(self.toggle)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.5, 600.0)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setValue(10.0)
        self.duration_spin.setToolTip("Take length in seconds")
        self.preset_buttons: dict[float, QPushButton] = {}
        for seconds in DURATION_PRESETS_S:
            button = QPushButton(f"{seconds:g} s")
            button.setToolTip(f"Set the take length to {seconds:g} seconds")
            button.clicked.connect(lambda _=False, s=seconds: self.set_duration(s))
            self.preset_buttons[seconds] = button
        self.countdown_combo = QComboBox()
        for seconds in COUNTDOWN_CHOICES_S:
            label = "no countdown" if seconds == 0 else f"{seconds:g} s countdown"
            self.countdown_combo.addItem(label, seconds)
        self.countdown_combo.setCurrentIndex(1)
        self.countdown_combo.setToolTip(
            "Delay between pressing Record and the recorder starting, so you can "
            "walk to address"
        )
        self.indicator = QLabel("")
        self.indicator.setMinimumWidth(LayoutMetrics.READOUT_MIN_WIDTH)
        self.indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setMaximumHeight(LayoutMetrics.PROGRESS_BAR_HEIGHT)
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self.tick)
        self._build()
        self._render()

    def _build(self) -> None:
        """The transport wraps (#9846): it sits under the video, which must
        stay the widest thing in the tile, so the strip reflows onto a second
        row instead of setting a floor under the whole window."""
        row = FlowLayout(spacing=LayoutMetrics.SPACING_SM)
        row.add_widget(self.record_button)
        row.add_widget(self.indicator)
        for button in self.preset_buttons.values():
            row.add_widget(button)
        row.add_widget(self.duration_spin)
        row.add_widget(self.countdown_combo)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, LayoutMetrics.SPACING_SM, 0, 0)
        layout.setSpacing(LayoutMetrics.SPACING_SM)
        layout.addLayout(row)
        layout.addWidget(self.progress)

    # -- settings -------------------------------------------------------------------
    def duration_s(self) -> float:
        return float(self.duration_spin.value())

    def set_duration(self, seconds: float) -> None:
        require(seconds > 0, "duration must be positive", seconds)
        self.duration_spin.setValue(seconds)

    def countdown_s(self) -> float:
        return float(self.countdown_combo.currentData())

    @property
    def phase(self) -> Phase:
        return self.clock.phase

    @property
    def badge(self) -> str:
        return self._badge

    # -- transport ------------------------------------------------------------------
    def toggle(self) -> None:
        """Record when idle; cancel the countdown or stop the take otherwise."""
        if self.clock.phase is Phase.IDLE:
            self.begin()
        elif self.clock.phase is Phase.COUNTDOWN:
            self.cancel()
        else:
            self.stop_requested.emit()

    def begin(self) -> None:
        """Arm the countdown; ``start_requested`` fires when it ends."""
        self.clock.arm(self.duration_s(), self.countdown_s(), self._now())
        self._timer.start()
        self.tick()

    def cancel(self) -> None:
        """Abandon an armed countdown without recording."""
        self._timer.stop()
        self.clock.reset()
        self._render()

    def recording_started(self) -> None:
        """The tile launched the recorder; the take clock runs from now."""
        self.clock.recording_started(self._now())
        if not self._timer.isActive():
            self._timer.start()
        self._render()

    def recording_finished(self) -> None:
        """The recorder exited (normally, early or failed)."""
        self._timer.stop()
        self.clock.reset()
        self._render()

    def tick(self) -> None:
        """Advance the clock; fire ``start_requested`` once the countdown ends."""
        self.clock.tick(self._now())
        if self.clock.countdown_done():
            self._timer.stop()  # restarted by recording_started
            self.start_requested.emit()
        self._render()

    # -- rendering ------------------------------------------------------------------
    def restyle(self) -> None:
        """Button and readout colours follow a theme change."""
        self._render()

    def _render(self) -> None:
        phase = self.clock.phase
        recording = phase is Phase.RECORDING
        self.record_button.setText(
            "■  Stop"
            if recording
            else ("Cancel" if phase is Phase.COUNTDOWN else "●  Record")
        )
        self.record_button.setStyleSheet(styling.record_button_style(recording))
        for widget in (
            self.duration_spin,
            self.countdown_combo,
            *self.preset_buttons.values(),
        ):
            widget.setEnabled(phase is Phase.IDLE)
        self.progress.setValue(int(round(self.clock.progress * 1000)))
        blink_on = int(self.clock.elapsed / (BLINK_PERIOD_S / 2)) % 2 == 0
        if recording:
            dot = "●" if blink_on else "○"
            text = f"{dot} REC {clock_text(self.clock.elapsed)} / {clock_text(self.clock.duration_s)}"
        elif phase is Phase.COUNTDOWN:
            text = self.clock.badge()
        else:
            text = "ready"
        self.indicator.setStyleSheet(
            styling.readout_style(recording, phase is Phase.COUNTDOWN)
        )
        self.indicator.setText(text)
        badge = self.clock.badge()
        if badge != self._badge:
            self._badge = badge
            self.badge_changed.emit(badge)
