"""Video Analyzer GUI component.

Replaces the former static "GUI placeholder" label (issue #8883): the
window now picks a video file, runs :meth:`SwingAnalyzer.analyze_video`
against it, and shows the resulting head-stability score. Analysis runs
off the GUI thread via :mod:`src.tools.async_action` (the same pattern
used by ``simulation_backends_launcher``) so MediaPipe decoding a whole
video does not freeze the window. A failure (bad file, MediaPipe not
installed) is surfaced as an honest status/report message instead of a
crash or a silently blank screen.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.launchers.help_menu import build_help_menu
from src.tools.async_action import AsyncActionBar, WorkerContext
from src.tools.window_theme import apply_theme_best_effort
from src.tools.video_analyzer.analyzer import SwingAnalyzer
from src.tools.video_analyzer.types import PostureMetrics

logger = logging.getLogger(__name__)

__all__ = ["MainWidget", "VideoAnalyzerWindow", "get_dockable_ui"]

#: Errors ``analyze_video`` raises for a caller to display, rather than
#: letting them crash the event loop.
_ANALYSIS_ERRORS = (FileNotFoundError, RuntimeError, ValueError)


class MainWidget(QWidget):
    """Central widget: pick a video, analyze it, show the result.

    Every action a test cares about is a synchronous, dialog-free public
    method (:meth:`set_video_path`, :meth:`run_analysis`); the button
    handlers wrap those methods, matching the convention documented in
    ``simulation_backends_launcher/gui.py``.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._video_path: Path | None = None

        self._build_widgets()
        self._build_layout()
        self._wire_signals()
        apply_theme_best_effort(self)

        self.status_label.setText("Choose a video to analyze.")

    # ---- construction ----------------------------------------------

    def _build_widgets(self) -> None:
        self.path_label = QLabel("No video selected.")
        self.path_label.setWordWrap(True)

        self.choose_button = QPushButton("Choose Video...")
        self.choose_button.setToolTip(
            "Pick a video file to run MediaPipe pose-based swing analysis on."
        )

        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setEnabled(False)
        self.analyze_button.setToolTip(
            "Run head-stability analysis on the selected video."
        )

        self.action_bar = AsyncActionBar()
        self.action_bar.set_trigger_buttons(self.analyze_button)

        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setPlaceholderText("Analysis results appear here.")

        self.status_label = QLabel()
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)

    def _build_layout(self) -> None:
        file_row = QHBoxLayout()
        file_row.addWidget(self.choose_button)
        file_row.addWidget(self.path_label, stretch=1)

        outer = QVBoxLayout(self)
        outer.addLayout(file_row)
        outer.addWidget(self.analyze_button)
        outer.addWidget(self.action_bar)
        outer.addWidget(self.report_text, stretch=1)
        outer.addWidget(self.status_label)

    def _wire_signals(self) -> None:
        self.choose_button.clicked.connect(self._on_choose_clicked)
        self.analyze_button.clicked.connect(self.run_analysis_async)

    # ---- testable core -----------------------------------------------

    def set_video_path(self, path: str | Path) -> None:
        """Select a video file for analysis.

        Args:
            path: Filesystem path to the video.

        Raises:
            ValueError: If ``path`` does not exist.
        """
        resolved = Path(path)
        if not resolved.exists():
            raise ValueError(f"video file not found: {resolved}")
        self._video_path = resolved
        self.path_label.setText(str(resolved))
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Ready to analyze.")

    def run_analysis(self) -> PostureMetrics | None:
        """Run analysis synchronously and render the result.

        Returns:
            The computed :class:`PostureMetrics`, or ``None`` if no video
            is selected yet or analysis failed (the failure is rendered
            into the status/report panes rather than raised).
        """
        if self._video_path is None:
            self.status_label.setText("Choose a video first.")
            return None
        try:
            metrics = SwingAnalyzer().analyze_video(self._video_path)
        except _ANALYSIS_ERRORS as exc:
            self._report_failure("Analysis failed", exc)
            return None
        self._present_metrics(metrics)
        return metrics

    def run_analysis_async(self) -> None:
        """Run analysis off the GUI thread, with progress and cancel."""
        if self._video_path is None:
            self.status_label.setText("Choose a video first.")
            return
        path = self._video_path

        def _work(ctx: WorkerContext) -> PostureMetrics:
            ctx.report(None, f"analyzing {path.name}")
            return SwingAnalyzer().analyze_video(path)

        self.action_bar.start(
            "Analyze",
            _work,
            on_finished=self._present_metrics,
            on_failed=lambda message: self._report_failure_text(
                "Analysis failed", message
            ),
        )

    def cleanup(self) -> None:
        """Cancel and join any running analysis (launcher tab close)."""
        self.action_bar.shutdown()

    # ---- internal helpers ----------------------------------------------

    def _present_metrics(self, metrics: PostureMetrics) -> None:
        """Render a completed analysis. GUI-thread only."""
        self.report_text.setPlainText(
            "Video analysis\n"
            "==============\n"
            f"head stability: {metrics.head_stability:.1f} / 100\n\n"
            "100 = no head movement detected across the video; lower "
            "scores indicate more head drift during the swing."
        )
        self.status_label.setText("Analysis complete.")

    def _report_failure(self, headline: str, exc: Exception) -> None:
        """Surface an analysis failure in the status/report panes and log it."""
        logger.exception(headline)
        self._report_failure_text(headline, str(exc))

    def _report_failure_text(self, headline: str, detail: str) -> None:
        """Surface a failure whose traceback the worker already logged."""
        message = f"{headline}: {detail}"
        self.status_label.setText(message)
        self.report_text.setPlainText(message)

    def _on_choose_clicked(self) -> None:
        """Button handler: pick a video via dialog, then select it."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose video",
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)",
        )
        if not path:
            return
        try:
            self.set_video_path(path)
        except ValueError as exc:
            self.status_label.setText(str(exc))


class VideoAnalyzerWindow(QMainWindow):
    """Standalone window for Video Analyzer."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Video Analyzer")
        self.setMinimumSize(800, 600)
        self._main_widget = MainWidget(self)
        self.setCentralWidget(self._main_widget)
        menubar = self.menuBar()
        assert menubar is not None
        build_help_menu(
            menubar,
            self,
            doc_target=(
                "Video Analysis Tutorial",
                "docs/tutorials/content/04_video_analysis.md",
            ),
        )

    @property
    def main_widget(self) -> MainWidget:
        """Return the embedded :class:`MainWidget`."""
        return self._main_widget


def get_dockable_ui(parent: Any = None) -> VideoAnalyzerWindow:
    """Return the main window instance for docking in the unified launcher."""
    return VideoAnalyzerWindow(parent)
