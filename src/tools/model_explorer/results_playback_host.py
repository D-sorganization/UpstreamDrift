"""Results Playback hosting for the Model Explorer main widget.

Owns the swing-datastore picker and the dock that shows the Results Playback
panel, so ``gui.py`` stays under the monolith line limit.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QInputDialog,
    QMessageBox,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

from .default_match import MatchScenario, default_match, match_scenarios
from .results_session import ResultsSession

logger = get_logger(__name__)


class ResultsPlaybackHostMixin:
    """File > Open Swing Result and the playback dock for ``MainWidget``."""

    playback_panel: Any | None
    playback_dock: QDockWidget | None
    playback_bar: Any | None
    shot_dock: QDockWidget | None = None
    _results: ResultsSession | None
    _inner: Any
    visualization_dock: Any
    visualization_widget: Any
    status_bar: Any
    export_video_action: Any
    matched_swing_menu: Any

    # Provided by MainWidget (the File > Open Swing Result handler).
    open_results: Callable[[], None]
    scrub_results: Callable[[int], None]

    def _dialog_parent(self) -> QWidget | None:
        """The host widget (``MainWidget``) as a dialog parent, if it is one."""
        return self if isinstance(self, QWidget) else None

    def load_results(self, path: str | Path, engine: str | None = None) -> bool:
        """Load a results archive for playback and export.

        A joint-space run (fit record, closed-loop rollout or same-input
        bundle) opens the Results Playback panel on the shared-appearance
        golfer; a marker archive is loaded for marker-overlay export.
        Returns False, with a message naming the file and the problem, when
        the archive is missing, fails schema validation or cannot be shown.
        """
        from src.shared.python.motion_matching.video_frame_source import (
            VideoSourceError,
        )

        results_path = Path(path)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            session = ResultsSession.load(results_path, engine)
            # Release the previous run's GL renderer before building the next:
            # closing it afterwards would tear down the new one's context.
            self._discard_results()
            panel = self._playback_panel_for(session)
        except (OSError, ImportError, RuntimeError, VideoSourceError) as exc:
            logger.warning("Could not load results %s: %s", results_path, exc)
            self._discard_results()
            self.export_video_action.setEnabled(False)
            QMessageBox.warning(
                self._dialog_parent(),
                "Results Load Error",
                f"Cannot play '{results_path.name}':\n\n{exc}",
            )
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._discard_results()
        self._results = session
        if hasattr(self, "playback_bar") and self.playback_bar is not None:
            self.playback_bar.set_session(session)
        self.scrub_results(0)
        self._show_playback_panel(panel)
        self._show_shot_card(session, panel)
        self.export_video_action.setEnabled(True)
        self.status_bar.showMessage(
            f"{session.title}: {results_path.name}, {session.frame_count} frames"
        )
        return True

    def open_swing_result(self) -> None:
        """Pick a run from the swing datastore (File > Open Swing Result...).

        Lists every playable run under the datastore (``UD_SWING_DATASTORE``
        or ``D:/FleetData/swing-datastore``); without one, falls back to the
        plain ``Load Results...`` file dialog.
        """
        from .pose_results import default_datastore_root, discover_result_runs

        root = default_datastore_root()
        runs = discover_result_runs(root) if root is not None else []
        if not runs:
            self.status_bar.showMessage("No swing datastore found; choose a file")
            self.open_results()
            return
        labels = [run.label for run in runs]
        choice, ok = QInputDialog.getItem(
            self._dialog_parent(),
            "Open Swing Result",
            f"Runs in {root}:",
            labels,
            0,
            False,
        )
        if ok and choice:
            run = runs[labels.index(choice)]
            self.load_results(run.path, run.engine)

    def load_match_scenario(self, scenario: MatchScenario) -> bool:
        """Open a matched swing and name it (and whether it is a demo) on screen."""
        if not self.load_results(scenario.path):
            return False
        if self.playback_panel is not None:
            self.playback_panel.set_heading(scenario.label)
        frames = 0 if self._results is None else self._results.frame_count
        self.status_bar.showMessage(f"Matched swing: {scenario.label}, {frames} frames")
        return True

    def load_default_match(self) -> bool:
        """Open the default matched swing; False (with a status hint) if none."""
        scenario = default_match()
        if scenario is None:
            self.status_bar.showMessage("No matched swing: see UD_DEFAULT_MATCH_BUNDLE")
        return scenario is not None and self.load_match_scenario(scenario)

    def add_matched_swing_menu(self, file_menu: Any) -> None:
        """File > Open Matched Swing, listing the configured swings on open."""
        self.matched_swing_menu = menu = file_menu.addMenu("Open &Matched Swing")
        menu.aboutToShow.connect(self.refresh_matched_swing_menu)
        self.refresh_matched_swing_menu()

    def refresh_matched_swing_menu(self) -> None:
        """Rebuild the matched-swing menu from the current configuration."""
        menu = self.matched_swing_menu
        menu.clear()
        for scenario in match_scenarios():
            action = menu.addAction(scenario.label)
            action.setToolTip(str(scenario.path))
            action.triggered.connect(
                lambda _=False, s=scenario: self.load_match_scenario(s)
            )

    @staticmethod
    def _playback_panel_for(session: ResultsSession) -> Any | None:
        """A Results Playback panel for a joint-space run, else ``None``."""
        if session.pose_source is None:
            return None
        from .results_playback_panel import ResultsPlaybackPanel

        try:
            return ResultsPlaybackPanel(session)
        except BaseException:
            session.close()
            raise

    def _show_playback_panel(self, panel: Any | None) -> None:
        self.playback_panel = panel
        if panel is None:
            return
        if self.playback_dock is None:
            self.playback_dock = QDockWidget("Results Playback", self._inner)
            self.playback_dock.setObjectName("ResultsPlaybackDock")
            self._inner.tabifyDockWidget(self.visualization_dock, self.playback_dock)
        self.playback_dock.setWidget(panel)
        self.playback_dock.show()
        self.playback_dock.raise_()

    def _show_shot_card(self, session: ResultsSession, panel: Any | None) -> None:
        """Dock the Shot card (impact factors, ball flight) beside the playback."""
        if panel is None:
            return
        from .playback_flight_panel import attach_shot_dock, summary_for_path

        summary = summary_for_path(session.path)
        self.shot_dock = attach_shot_dock(self._inner, self.shot_dock, summary)

    def _discard_results(self) -> None:
        """Drop the loaded run, its panel and its renderer; idempotent."""
        if self.shot_dock is not None:
            self.shot_dock.hide()
        panel, self.playback_panel = self.playback_panel, None
        if panel is not None:
            # The C++ widget may already be gone during teardown.
            with contextlib.suppress(RuntimeError):
                panel.stop()
                panel.deleteLater()
        if hasattr(self, "playback_bar") and self.playback_bar is not None:
            self.playback_bar.set_session(None)
        if hasattr(self, "visualization_widget") and hasattr(
            self.visualization_widget, "set_markers"
        ):
            self.visualization_widget.set_markers(None)
        session, self._results = self._results, None
        if session is not None:
            session.close()
