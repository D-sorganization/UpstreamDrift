"""Results Playback hosting for the Model Explorer main widget.

Owns the swing-datastore picker and the dock that shows the Results Playback
panel, so ``gui.py`` stays under the monolith line limit.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEventLoop
from PyQt6.QtWidgets import (
    QDockWidget,
    QInputDialog,
    QMessageBox,
    QProgressBar,
    QWidget,
)

from src.shared.python.logging_pkg.logging_config import get_logger

from .default_match import MatchScenario, default_match, match_scenarios
from .results_loader_worker import LOAD_ERRORS, ResultsLoader, ResultsLoadSettings
from .playback_view import load_layout_settings
from .results_session import ResultsSession

logger = get_logger(__name__)


class ResultsPlaybackHostMixin:
    """File > Open Swing Result and the playback dock for ``MainWidget``."""

    playback_panel: Any | None
    playback_dock: QDockWidget | None
    playback_bar: Any | None
    segment_dock: QDockWidget
    properties_dock: QDockWidget
    _hidden_editor_docks: tuple[QDockWidget, ...] = ()
    _results: ResultsSession | None
    _inner: Any
    visualization_dock: Any
    visualization_widget: Any
    status_bar: Any
    export_video_action: Any
    matched_swing_menu: Any
    results_load_settings = ResultsLoadSettings()
    _results_loader: ResultsLoader | None = None
    _results_message: str | None = None

    # Provided by MainWidget (the File > Open Swing Result handler).
    open_results: Callable[[], None]
    scrub_results: Callable[[int], None]

    def _dialog_parent(self) -> QWidget | None:
        """The host widget (``MainWidget``) as a dialog parent, if it is one."""
        return self if isinstance(self, QWidget) else None

    def load_results(
        self,
        path: str | Path,
        engine: str | None = None,
        *,
        wait: bool = True,
        heading: str | None = None,
    ) -> bool:
        """Load a results archive for playback and export.

        A joint-space run (fit record, closed-loop rollout or same-input
        bundle) opens the Results Playback panel on the shared-appearance
        golfer; a marker archive is loaded for marker-overlay export.
        A :class:`ResultsLoader` thread does the slow work and supersedes a
        load in flight. ``wait=False`` returns True once started;
        otherwise, still processing events, it waits for the whole load.
        Returns False, with a message naming the file and the problem, when
        the archive is missing, fails schema validation or cannot be shown.
        ``heading`` (a matched-swing label) titles the panel.
        """
        results_path = Path(path)
        loader = ResultsLoader(results_path, engine)
        self._results_loader = loader
        self._live_loaders().add(loader)
        loader.loaded.connect(
            lambda s, ld=loader: self._deliver(self._on_loaded, ld, s, heading)
        )
        loader.failed.connect(
            lambda text, ld=loader: self._deliver(self._on_load_failed, ld, text)
        )
        loader.finished.connect(
            lambda ld=loader: self._deliver(self._on_loader_finished, ld)
        )
        self._results_message = None
        self.status_bar.showMessage(f"Loading {heading or results_path.name}\u2026")
        self._load_progress_bar().show()
        loader.start()
        return self.wait_for_results_load() if wait else True

    @property
    def is_loading_results(self) -> bool:
        """True while a results archive is loading in the background."""
        return self._results_loader is not None

    def wait_for_results_load(self, timeout_s: float | None = None) -> bool:
        """Process events until the load ends (``timeout_s``, default from
        ``results_load_settings``, must be positive); True if a run is in."""
        limit = timeout_s or self.results_load_settings.wait_timeout_s
        if limit <= 0:
            raise ValueError(f"timeout_s must be positive, got {limit}")
        # Poll the state, not ``finished``: it may already have been emitted,
        # and a nested QEventLoop would then sit out the whole timeout.
        deadline = time.monotonic() + limit
        while self.is_loading_results and time.monotonic() < deadline:
            QCoreApplication.processEvents(
                QEventLoop.ProcessEventsFlag.WaitForMoreEvents,
                self.results_load_settings.poll_interval_ms,
            )
        return self._results is not None and not self.is_loading_results

    def cancel_results_loads(self) -> None:
        """Abandon every background load and wait for its thread (teardown)."""
        self._results_loader = None
        loaders = self._live_loaders()
        for loader in list(loaders):
            if not sip.isdeleted(loader):  # Qt may have retired it already
                loader.wait()
        loaders.clear()
        QCoreApplication.sendPostedEvents()  # superseded sessions close now

    def _deliver(self, handler: Callable[..., None], *args: Any) -> None:
        """Run ``handler`` unless this widget is deleted (PyQt6 would abort)."""
        if isinstance(self, QWidget) and sip.isdeleted(self):
            for arg in args:  # forget the loader, release an unshown session
                if isinstance(arg, ResultsLoader):
                    self._live_loaders().discard(arg)
                elif hasattr(arg, "close"):
                    arg.close()
            return
        handler(*args)

    def _live_loaders(self) -> set[ResultsLoader]:
        # Instance dict: ``cleanup`` runs from ``destroyed`` (Qt lookups raise).
        loaders: set[ResultsLoader] = self.__dict__.setdefault("_loaders", set())
        return loaders

    def _load_progress_bar(self) -> QProgressBar:
        if getattr(self, "_results_progress", None) is None:
            self._results_progress = bar = QProgressBar()
            bar.setRange(0, 0)  # busy indicator: the load has no known length
            bar.setMaximumWidth(self.results_load_settings.progress_width_px)
            self.status_bar.addPermanentWidget(bar)
        return self._results_progress

    def _on_loaded(
        self, loader: ResultsLoader, session: ResultsSession, heading: str | None
    ) -> None:
        if loader is not self._results_loader:
            session.close()  # superseded while it loaded
            return
        try:
            # Release the previous run's GL renderer before building the next:
            # closing it afterwards would tear down the new one's context.
            self._discard_results()
            panel = self._playback_panel_for(session)
        except LOAD_ERRORS as exc:
            self._on_load_failed(loader, str(exc))
            return
        self._results = session
        if hasattr(self, "playback_bar") and self.playback_bar is not None:
            self.playback_bar.set_session(session)
        self.scrub_results(0)
        self._show_playback_panel(panel)
        if panel is not None and heading:
            panel.set_heading(heading)
        self.export_video_action.setEnabled(True)
        frames = f"{session.frame_count} frames"
        self._results_message = (
            f"Matched swing: {heading}, {frames}"
            if heading
            else f"{session.title}: {loader.path.name}, {frames}"
        )
        self.status_bar.showMessage(self._results_message)

    def _on_load_failed(self, loader: ResultsLoader, message: str) -> None:
        if loader is not self._results_loader:
            return
        logger.warning("Could not load results %s: %s", loader.path, message)
        self._discard_results()
        self._results_message = None
        self.export_video_action.setEnabled(False)
        self.status_bar.showMessage(f"Could not load {loader.path.name}")
        QMessageBox.warning(
            self._dialog_parent(),
            "Results Load Error",
            f"Cannot play '{loader.path.name}':\n\n{message}",
        )

    def _on_loader_finished(self, loader: ResultsLoader) -> None:
        if not sip.isdeleted(loader):  # Qt may have retired it already
            loader.wait()
        self._live_loaders().discard(loader)  # the worker retires the thread
        if loader is self._results_loader:
            self._results_loader = None
            self._load_progress_bar().hide()
            if self._results_message:
                self.status_bar.showMessage(self._results_message)

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
            self.load_results(run.path, run.engine, wait=False)

    def load_match_scenario(self, scenario: MatchScenario) -> bool:
        """Start opening a matched swing, named (demo or not) on screen.

        Returns once the background load has started (see ``load_results``).
        """
        return self.load_results(scenario.path, wait=False, heading=scenario.label)

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
            self._inner.tabifiedDockWidgetActivated.connect(self._on_dock_tab_shown)
        self.playback_dock.setWidget(panel)
        self.playback_dock.show()
        self.playback_dock.raise_()
        self._on_dock_tab_shown(self.playback_dock)

    def _on_dock_tab_shown(self, dock: QDockWidget) -> None:
        """Playback tab on top: the swing gets the window, the editor docks
        step aside; the viewport tab brings back the ones this hid."""
        if dock is self.playback_dock:
            if load_layout_settings().hide_editor_docks:
                shown = tuple(
                    d
                    for d in (self.segment_dock, self.properties_dock)
                    if not d.isHidden()
                )
                self._hidden_editor_docks += shown
                for hidden in shown:
                    hidden.hide()
        elif dock is self.visualization_dock:
            for restored in self._hidden_editor_docks:
                restored.show()
            self._hidden_editor_docks = ()

    def _discard_results(self) -> None:
        """Drop the loaded run, its panel and its renderer; idempotent."""
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
