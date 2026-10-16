"""Results playback panel of the Model Explorer (#12915, epic #12878).

Shows a joint-space run on the shared-appearance golfer: a 3D view rendered
by :class:`GolferPlayback`, a timeline with play/pause, a camera-view chooser,
a time-series plot (club-head speed and every joint angle) with a cursor at
the current time, and an ``Export MP4...`` button. All state lives in the
Qt-free :class:`ResultsSession`; this widget only draws it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.golf_view_presets.presets import VIEW_ORDER
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.ui.qt.plotting import MplCanvas
from src.tools.model_explorer.golfer_playback import (
    CLUB_HEAD_SPEED,
    TimeSeries,
    results_time_series,
    spec_summary,
)
from src.tools.model_explorer.playback_clock import SPEEDS, PlaybackClock
from src.tools.model_explorer.results_session import ResultsSession

logger = get_logger(__name__)

__all__ = ["ResultsPlaybackPanel"]

PLAYBACK_INTERVAL_MS = 33  # ~30 fps timeline playback tick (frames skip if late)


def _speed_text(speed: float) -> str:
    """``0.25`` -> ``"0.25x"``, ``1.0`` -> ``"1x"``."""
    return f"{speed:g}x"


class _ScaledPixmapLabel(QLabel):
    """Label that keeps its pixmap scaled (aspect kept) to its own size.

    ``Ignored`` size policy: the pixmap never dictates the label's size, so the
    image grows and shrinks with the dock instead of staying at its first size.
    """

    def __init__(self) -> None:
        super().__init__()
        self._source: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_source(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self._rescale()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source is None:
            return
        self.setPixmap(
            self._source.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class ResultsPlaybackPanel(QWidget):
    """3D golfer view, timeline, plots and export for one loaded run.

    Precondition: ``session.pose_source`` is not ``None`` (``ValueError``).
    """

    def __init__(self, session: ResultsSession, parent: QWidget | None = None):
        super().__init__(parent)
        pose = session.pose_source
        if pose is None:
            raise ValueError("the playback panel needs a joint-space run")
        self.session = session
        self._pose = pose
        self._playback = session.playback()
        self.series: dict[str, TimeSeries] = results_time_series(pose)
        self._clock = PlaybackClock(pose.time_s)
        self._now: Callable[[], float] = time.monotonic
        self._timer = QTimer(self)
        self._timer.setInterval(PLAYBACK_INTERVAL_MS)
        self._timer.timeout.connect(self._advance)
        self._build_ui()
        self.show_frame(0)

    # ---- construction -----------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.title_label = QLabel(self.session.title)
        self.title_label.setObjectName("ResultsPlaybackTitle")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        self.source_label = QLabel(
            f"{self._pose.source_tag}: {spec_summary(self._pose)}"
        )
        self.source_label.setWordWrap(True)
        self.source_label.setToolTip(str(self._pose.path))
        layout.addWidget(self.source_label)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.view_label = _ScaledPixmapLabel()
        self.view_label.setObjectName("ResultsPlaybackView")
        splitter.addWidget(self.view_label)
        splitter.addWidget(self._build_plot())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        layout.addLayout(self._build_timeline())

    def _build_plot(self) -> QWidget:
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("Plot:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(list(self.series))
        self.channel_combo.currentTextChanged.connect(self._draw_plot)
        row.addWidget(self.channel_combo, 1)
        column.addLayout(row)
        self.canvas = MplCanvas(width=6, height=2.4, dpi=90)
        # Constrained layout re-fits labels on every draw, so a resized dock
        # never clips the axis labels (a one-off tight_layout did).
        self.canvas.fig.set_layout_engine("constrained")
        self.canvas.setMinimumHeight(140)
        self._axes = self.canvas.fig.add_subplot(111)
        self._cursor: Any = None
        column.addWidget(self.canvas, 1)
        self._draw_plot(self.channel_combo.currentText())
        return box

    def _build_timeline(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.toggled.connect(self._on_play_toggled)
        row.addWidget(self.play_button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("ResultsTimeline")
        self.slider.setRange(0, self._pose.frame_count - 1)
        self.slider.valueChanged.connect(self._on_scrub)
        row.addWidget(self.slider, 1)
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(150)
        row.addWidget(self.time_label)
        row.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(list(VIEW_ORDER))
        self.view_combo.currentTextChanged.connect(self.set_view)
        row.addWidget(self.view_combo)
        row.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems([_speed_text(s) for s in SPEEDS])
        self.speed_combo.setCurrentText(_speed_text(1.0))
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        row.addWidget(self.speed_combo)
        self.export_button = QPushButton("Export MP4...")
        self.export_button.setToolTip("Render this run on the golfer to an MP4")
        self.export_button.clicked.connect(self.export_mp4)
        row.addWidget(self.export_button)
        return row

    # ---- playback ---------------------------------------------------------

    @property
    def current_frame(self) -> int:
        return self.session.current_frame

    def current_image(self) -> np.ndarray:
        """RGB array of the frame on screen (re-rendered, for audits)."""
        return np.asarray(self._playback.render(self.current_frame))

    def show_frame(self, index: int) -> None:
        """Scrub to ``index``: pose the golfer, move the plot cursor."""
        self.session.scrub(index)
        if self.slider.value() != index:
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)
        rgb = np.ascontiguousarray(self._playback.render(index))
        height, width, _ = rgb.shape
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image.copy())
        self.view_label.set_source(self._pixmap)
        time_s = self.session.current_time_s
        self.time_label.setText(
            f"t = {time_s:.3f} s  ({index + 1}/{self._pose.frame_count})"
        )
        if self._cursor is not None:
            self._cursor.set_xdata([time_s, time_s])
            self.canvas.draw_idle()

    def set_view(self, name: str) -> None:
        """Switch the camera preset and redraw the current frame."""
        self._playback.view = name
        self.show_frame(self.current_frame)

    @property
    def speed(self) -> float:
        """Playback speed relative to real time (one of ``SPEEDS``)."""
        return self._clock.speed

    def _on_scrub(self, index: int) -> None:
        """Slider moved by the user: show it and, if playing, play on from it."""
        self.show_frame(index)
        if self.play_button.isChecked():
            self._clock.start(index, self._now())

    def _on_speed_changed(self, position: int) -> None:
        self._clock.set_speed(SPEEDS[position], self._now())

    def _on_play_toggled(self, playing: bool) -> None:
        self.play_button.setText("Pause" if playing else "Play")
        if playing:
            self._clock.start(self.current_frame, self._now())
            self._timer.start()
        else:
            self._timer.stop()

    def _advance(self) -> None:
        """Timer tick: show the frame the wall clock has reached (may skip)."""
        index = self._clock.frame_at(self._now())
        if index != self.current_frame:
            self.show_frame(index)

    # ---- plots ------------------------------------------------------------

    def _draw_plot(self, name: str) -> None:
        series = self.series.get(name)
        axes = self._axes
        axes.clear()
        if series is None:
            self.canvas.draw_idle()
            return
        axes.plot(self._pose.time_s, series.values, color="#1f77b4", linewidth=1.4)
        axes.set_xlabel("Time (s)")
        axes.set_ylabel(f"{series.name}\n({series.unit})", fontsize=9)
        if name == CLUB_HEAD_SPEED and series.values.size:
            peak = int(np.argmax(series.values))
            axes.set_title(
                f"Peak {series.values[peak]:.1f} mph at "
                f"t = {self._pose.time_s[peak]:.3f} s",
                fontsize=9,
            )
        axes.grid(True, alpha=0.3)
        self._cursor = axes.axvline(
            self.session.current_time_s, color="#d62728", linewidth=1.0
        )
        self.canvas.draw_idle()

    def select_channel(self, name: str) -> None:
        """Plot channel ``name``; ``ValueError`` when the run has no such channel."""
        if name not in self.series:
            raise ValueError(f"no channel {name!r}; have {sorted(self.series)}")
        self.channel_combo.setCurrentText(name)

    # ---- export -----------------------------------------------------------

    def export_mp4(self) -> Path | None:
        """Ask for a destination and film the run on the golfer."""
        self.play_button.setChecked(False)
        default = f"{self._pose.engine}_{self._pose.run_name}.mp4"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export MP4", default, "MP4 Video (*.mp4)"
        )
        if not file_path:
            return None
        target = Path(file_path)
        if target.suffix.lower() != ".mp4":
            target = target.with_suffix(".mp4")
        try:
            out = self.session.export_video(target)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Results video export failed")
            QMessageBox.critical(self, "Export MP4", f"Could not export video:\n{exc}")
            return None
        QMessageBox.information(self, "Export MP4", f"Exported video to:\n{out}")
        return out

    def stop(self) -> None:
        """Stop playback (called before the panel is discarded)."""
        self._timer.stop()
