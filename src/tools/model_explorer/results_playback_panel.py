"""Results playback panel of the Model Explorer (#12915, epic #12878).

Shows a joint-space run on the shared-appearance golfer: a 3D view rendered
by :class:`GolferPlayback`, a timeline with play/pause, a camera-view chooser,
a time-series plot (club-head speed and every joint angle) with a cursor at
the current time, an ``Export MP4...`` button and the export's ball, ball
flight, shot shape and finish aim line (:mod:`playback_ball_overlay`). All
state lives in the
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
    QCheckBox,
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
from src.tools.model_explorer.playback_ball_overlay import (
    SHAPES,
    BallOverlay,
    ShotLayout,
    flight_unavailable_reason,
    load_overlay_settings,
)
from src.tools.model_explorer.playback_clock import SPEEDS, PlaybackClock
from src.tools.model_explorer.playback_view import load_layout_settings
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
        self._ball_settings = load_overlay_settings()
        self.ball_overlay = BallOverlay(
            pose, self._ball_settings, flight_unavailable_reason()
        )
        self._playback.scene_hook = self._decorate_scene
        self._ball_poll = QTimer(self)
        self._ball_poll.setInterval(self._ball_settings.poll_interval_ms)
        self._ball_poll.timeout.connect(self._poll_ball)
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
        cfg = load_layout_settings()
        splitter.setStretchFactor(0, cfg.render_stretch)
        splitter.setStretchFactor(1, cfg.plot_stretch)
        self._split_ratio = [cfg.render_stretch, cfg.plot_stretch]
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter
        layout.addWidget(splitter, 1)
        layout.addLayout(self._build_timeline())
        layout.addLayout(self._build_ball_row())

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        """First show: split the height by the settings ratio (size hints
        would otherwise give the plot half of it)."""
        super().showEvent(event)
        if self._split_ratio:
            QTimer.singleShot(0, self._apply_split_ratio)

    def _apply_split_ratio(self) -> None:
        total = self.splitter.height()
        weights = self._split_ratio
        if weights and total > 0:
            self.splitter.setSizes([total * w // sum(weights) for w in weights])
            self._split_ratio = []

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
        self.slider.setMinimumWidth(load_layout_settings().slider_min_width_px)
        self.slider.setRange(0, self._pose.frame_count - 1)
        self.slider.valueChanged.connect(self._on_scrub)
        row.addWidget(self.slider, 1)
        self.time_label = QLabel()
        self.time_label.setMinimumWidth(150)
        row.addWidget(self.time_label)
        row.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(list(VIEW_ORDER))
        self.view_combo.setCurrentText(self._playback.view)
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

    def _build_ball_row(self) -> QHBoxLayout:
        row, defaults = QHBoxLayout(), self._ball_settings
        self.ball_check = QCheckBox("Ball")
        self.ball_check.setChecked(defaults.show_ball)
        self.flight_check = QCheckBox("Ball flight")
        self.aim_check = QCheckBox("Aim line")
        self.shape_combo = QComboBox()
        self.shape_combo.addItems(list(SHAPES))
        self.shape_combo.setCurrentText(defaults.shot_shape)
        self.shape_combo.setToolTip("Shot shape of a right-handed golfer")
        reason = self.ball_overlay.flight_reason
        self.flight_check.setChecked(defaults.show_flight and reason is None)
        self.aim_check.setChecked(defaults.show_aim_line and reason is None)
        for box in (self.flight_check, self.aim_check, self.shape_combo):
            box.setEnabled(reason is None)
            if reason is not None:
                box.setToolTip(
                    defaults.flight_unavailable_tooltip.format(reason=reason)
                )
        for box in (self.ball_check, self.flight_check, self.aim_check):
            box.toggled.connect(self._redraw)
            row.addWidget(box)
        if reason is None:
            self.aim_check.setToolTip("Ends where the ball finishes; needs Ball flight")
        self.flight_check.toggled.connect(self._sync_aim)
        self._sync_aim()
        row.addWidget(QLabel("Shot:"))
        self.shape_combo.currentTextChanged.connect(self._redraw)
        row.addWidget(self.shape_combo)
        self.ball_status = QLabel()
        self.ball_status.setWordWrap(True)
        row.addWidget(self.ball_status, 1)
        return row

    def set_heading(self, name: str) -> None:
        """Prefix the panel title with what was opened (e.g. a demo label)."""
        self.title_label.setText(f"{name}: {self.session.title}")

    # ---- playback ---------------------------------------------------------

    @property
    def current_frame(self) -> int:
        return self.session.current_frame

    def current_image(self) -> np.ndarray:
        """RGB array of the frame on screen (re-rendered, for audits)."""
        return self._frame_rgb(self.current_frame)

    def _frame_rgb(self, index: int) -> np.ndarray:
        """The golfer at ``index`` with the ball, flight and aim line asked for."""
        rgb = np.asarray(self._playback.render(index))
        flight, aim = self.flight_check.isChecked(), self.aim_check.isChecked()
        layout = self._shot_layout()
        if layout is None or layout.flight is None or not (flight or aim):
            return rgb
        depth = self._playback.scene_depth()
        time_s = float(self._pose.time_s[index])
        view = self._playback.view
        paint = self.ball_overlay.paint
        return paint(rgb, view, time_s, layout, flight=flight, aim=aim, depth_m=depth)

    def _sync_aim(self) -> None:
        """The aim line ends where the ball finishes: it needs the flight."""
        flying = self.flight_check.isChecked()
        self.aim_check.setEnabled(flying)
        if not flying:
            self.aim_check.setChecked(False)

    def _shot_layout(self) -> ShotLayout | None:
        """The finished layout asked for; polls until it is ready.

        The flight (and the shape's club-roll solve) is computed only while
        Ball flight is checked; otherwise only the ball at address is placed.
        """
        flight = self.flight_check.isChecked()
        layout = self.ball_overlay.layout(self.shape_combo.currentText(), flight)
        if layout is not None:
            self.ball_status.setText(layout.note)
        elif not self._ball_poll.isActive():
            what = "ball flight" if flight else "ball"
            self.ball_status.setText(f"Computing {what}...")
            self._ball_poll.start()
        return layout

    def _decorate_scene(self, scene: Any, index: int) -> None:
        if self.ball_check.isChecked():
            time_s = float(self._pose.time_s[index])
            self.ball_overlay.decorate_scene(scene, time_s, self._shot_layout())

    def _poll_ball(self) -> None:
        shape, flight = self.shape_combo.currentText(), self.flight_check.isChecked()
        if self.ball_overlay.layout(shape, flight) is not None:
            self._ball_poll.stop()
            self._redraw()

    def _redraw(self) -> None:
        self.show_frame(self.current_frame)

    def set_ball_visible(self, visible: bool) -> None:
        """Show or hide the ball at address."""
        self.ball_check.setChecked(bool(visible))

    def set_ball_flight_visible(self, visible: bool) -> None:
        """Show or hide the flight trace; stays off when the model cannot run."""
        self.flight_check.setChecked(bool(visible) and self.flight_check.isEnabled())

    def set_aim_line_visible(self, visible: bool) -> None:
        """Show or hide the finish aim line; stays off without the flight."""
        self.aim_check.setChecked(bool(visible) and self.aim_check.isEnabled())

    def set_shot_shape(self, shape: str) -> None:
        """Choose the shot shape (``straight``, ``draw`` or ``fade``)."""
        if shape not in SHAPES:
            raise ValueError(f"shot shape must be one of {SHAPES}, got {shape!r}")
        self.shape_combo.setCurrentText(shape)

    def wait_for_shot(self, timeout_s: float) -> ShotLayout:
        """Block until the layout asked for is computed, then redraw (audits)."""
        shape, flight = self.shape_combo.currentText(), self.flight_check.isChecked()
        future = self.ball_overlay.request(shape, flight)
        future.exception(timeout=timeout_s)
        self._poll_ball()
        return future.result()

    def show_frame(self, index: int) -> None:
        """Scrub to ``index``: pose the golfer, move the plot cursor."""
        self.session.scrub(index)
        if self.slider.value() != index:
            self.slider.blockSignals(True)
            self.slider.setValue(index)
            self.slider.blockSignals(False)
        rgb = np.ascontiguousarray(self._frame_rgb(index))
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
        axes.plot(self._pose.time_s, series.values, color="tab:blue", linewidth=1.4)
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
            self.session.current_time_s, color="tab:red", linewidth=1.0
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
        self._ball_poll.stop()
        self._playback.scene_hook = None
        self.ball_overlay.close()
