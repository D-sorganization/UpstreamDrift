"""PyQt6 Launch Monitor Analytics workbench."""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.tools.async_action import WorkerContext
from src.tools.launch_monitor_analytics._gui_analysis_compare import (
    _CompareAnalysisMixin,
)
from src.tools.launch_monitor_analytics._gui_analysis_core import _CoreAnalysisMixin
from src.tools.launch_monitor_analytics._gui_io import _ImportExportMixin
from src.tools.launch_monitor_analytics._gui_params import (  # noqa: F401
    _ELLIPSE_POINTS,
    _ComparisonParams,
    _DispersionData,
    _DispersionParams,
    _DispersionPlotData,
    _ModelParams,
    _RelationshipParams,
    _TrendParams,
)
from src.tools.launch_monitor_analytics._gui_tabs import _MainWidgetTabsMixin
from src.tools.launch_monitor_analytics.destructive_guards import (
    DestructiveActionGuards,
)
from src.tools.launch_monitor_analytics.plot_canvas import PlotCanvas
from src.tools.launch_monitor_analytics.reporting import (
    build_project_report,
)
from src.tools.launch_monitor_analytics.widgets import (
    populate_combo as _populate_combo,
)
from src.tools.launch_monitor_model import (
    ImportedSession,
    ImportManifest,
    ImportOptions,
    LaunchMonitorProject,
    import_session,
    load_private_corpus,
    numeric_metric_columns,
)

logger = get_logger(__name__)

__all__ = ["LaunchMonitorAnalyticsWindow", "MainWidget", "PlotCanvas", "main"]


class MainWidget(
    _MainWidgetTabsMixin,
    _CoreAnalysisMixin,
    _CompareAnalysisMixin,
    _ImportExportMixin,
    DestructiveActionGuards,
    QtWidgets.QWidget,
):
    """Embeddable launch-monitor data-management and analysis workspace."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Launch Monitor Analytics")
        self.project = LaunchMonitorProject("Untitled Launch Monitor Study")
        self.analysis_frame = pd.DataFrame()
        self._dirty = False
        self._last_data_export: dict[str, str] | None = None
        self._build_ui()
        self._wire_actions()
        self._apply_theme_best_effort()
        self._refresh_all()

    def is_dirty(self) -> bool:
        return self._dirty

    def import_file(self, path: str | Path, options: ImportOptions | None = None):  # noqa: ANN201
        """Import and append a session, then refresh every workspace."""
        session = import_session(path, options)
        self.project.add_session(session)
        self.analysis_frame = self.project.combined_shots()
        self._dirty = True
        self._refresh_all()
        self.status_label.setText(
            f"Imported {session.name}: {len(session.shots)} shots from "
            f"{session.manifest.vendor}. Project now contains "
            f"{len(self.analysis_frame)} shots."
        )
        return session

    def load_private_corpus_sessions(self, root: str | Path | None = None) -> int:
        """Load the private shot corpus as one session per corpus source.

        Returns the number of sessions added. Sources already present in the
        project are skipped rather than raising, so the action is repeatable.
        Raises ``FileNotFoundError`` when no authorized checkout or corpus
        manifest is available, ``ValueError`` when the corpus disagrees with
        the manifest that describes it (the ADR-0048 D30 gate), and
        ``ImportError`` when the Parquet reader is not installed; callers in
        the UI surface those as dialogs.
        """
        frame = load_private_corpus(root)
        existing = {session.session_id for session in self.project.sessions}
        imported_at = datetime.now(UTC).isoformat()
        added = 0
        for session_id, group in frame.groupby("session_id", sort=True, observed=True):
            identifier = str(session_id)
            if identifier in existing:
                continue
            shots = group.reset_index(drop=True)
            vendor = str(shots["monitor_vendor"].iloc[0])
            manifest = ImportManifest(
                source_path=f"corpus://{identifier}",
                file_sha256="private-authority-parquet",
                profile_id="private_corpus",
                vendor=vendor,
                imported_at=imported_at,
                row_count=len(shots),
                source_columns=tuple(str(column) for column in shots.columns),
                metric_sources={},
                source_units={},
                unit_evidence={},
            )
            self.project.add_session(
                ImportedSession(
                    session_id=identifier,
                    name=identifier,
                    shots=shots,
                    manifest=manifest,
                )
            )
            added += 1
        if added:
            self.analysis_frame = self.project.combined_shots()
            self._dirty = True
            self._refresh_all()
        return added

    def _on_load_private_corpus(self) -> None:
        previous = QtWidgets.QApplication.overrideCursor()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            added = self.load_private_corpus_sessions()
        except (FileNotFoundError, ImportError, ValueError) as error:
            logger.info("Private corpus load unavailable: %s", error)
            QtWidgets.QMessageBox.warning(
                self, "Private corpus unavailable", str(error)
            )
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            if previous is not None:
                QtWidgets.QApplication.setOverrideCursor(previous)
        if not added:
            self.status_label.setText(
                "Private corpus already loaded; no new sources were added."
            )
            return
        self.status_label.setText(
            f"Loaded the private corpus: {added} sources, "
            f"{len(self.analysis_frame)} shots now in the project."
        )

    def save_project(self, path: str | Path) -> Path:
        saved = self.project.save(path)
        self._dirty = False
        self.status_label.setText(f"Saved project to {saved}.")
        return saved

    def load_project(self, path: str | Path) -> None:
        self.project = LaunchMonitorProject.load(path)
        self.analysis_frame = self.project.combined_shots()
        self._dirty = False
        self._refresh_all()
        self.status_label.setText(
            f"Loaded {self.project.name}: {len(self.analysis_frame)} shots."
        )

    def clear_project(self) -> None:
        self.project = LaunchMonitorProject("Untitled Launch Monitor Study")
        self.analysis_frame = pd.DataFrame()
        self._dirty = False
        self._last_data_export = None
        self._refresh_all()
        self.status_label.setText("New empty project. Import launch-monitor exports.")

    def _refresh_all(self) -> None:
        self._refresh_session_tree()
        self.data_table.set_frame(self.analysis_frame)
        self.flexible_analysis.set_frame(self.analysis_frame)
        self._reset_analysis_canvases()
        metrics = numeric_metric_columns(self.analysis_frame)
        for widget in (
            self.relationship_metrics,
            self.relationship_controls,
            self.model_features,
        ):
            widget.clear()
            widget.addItems(metrics)
        for combo in (
            self.model_target,
            self.comparison_metric,
            self.dispersion_forward,
            self.dispersion_lateral,
            self.trend_metric,
        ):
            _populate_combo(combo, metrics)
        columns = [str(column) for column in self.analysis_frame.columns]
        _populate_combo(self.comparison_match, ["(unmatched)", *columns])
        monitors = (
            sorted(self.analysis_frame["monitor_vendor"].dropna().astype(str).unique())
            if "monitor_vendor" in self.analysis_frame
            else []
        )
        _populate_combo(self.comparison_reference, monitors)
        time_columns = [
            column
            for column in columns
            if "time" in column or "date" in column or column == "captured_at"
        ]
        _populate_combo(self.trend_time, time_columns)
        self._set_preferred_combo(self.dispersion_forward, "carry_distance")
        self._set_preferred_combo(self.dispersion_lateral, "lateral_carry")
        self._refresh_report()

    @staticmethod
    def _set_preferred_combo(combo: QtWidgets.QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _reset_analysis_canvases(self) -> None:
        """Clear every analysis canvas so stale charts never outlive their data.

        Called from ``_refresh_all``, the single choke point reached by
        ``clear_project``, ``import_file``, ``load_private_corpus_sessions``,
        ``load_project``, ``_remove_selected_sessions``, and
        ``_run_treatment_ui``. Any of those changes the underlying
        project/session set, so a chart built against the previous data
        must not remain visible until (or unless) the user reruns that
        specific analysis against the new data.
        """
        message = f"{self.project.name}: run an analysis to populate this chart."
        for canvas in (
            self.relationship_plot,
            self.model_plot,
            self.comparison_plot,
            self.dispersion_plot,
            self.trend_plot,
        ):
            canvas.empty(message)

    def _title_with_project(self, text: str) -> str:
        """Append the current project name so a chart's data identity is visible."""
        return f"{text} — {self.project.name}"

    def _refresh_session_tree(self) -> None:
        self.session_tree.clear()
        for session in self.project.sessions:
            item = QtWidgets.QTreeWidgetItem(
                [
                    session.name,
                    session.manifest.vendor,
                    str(len(session.shots)),
                    session.manifest.profile_id,
                ]
            )
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, session.session_id)
            item.setToolTip(0, session.manifest.source_path)
            self.session_tree.addTopLevelItem(item)
        self.session_tree.resizeColumnToContents(0)

    def _refresh_report(self) -> None:
        warnings = [
            warning
            for session in self.project.sessions
            for warning in session.manifest.warnings
        ]
        self.report_text.setPlainText(
            build_project_report(
                project_name=self.project.name,
                session_count=len(self.project.sessions),
                frame=self.analysis_frame,
                import_warning_count=len(warnings),
                treatment_action_count=len(self.project.audit_log),
            )
        )

    # ---- shared action-bar plumbing (#9470) -----------------------------

    def _start_action(
        self,
        name: str,
        work: Callable[[WorkerContext], Any],
        present: Callable[[Any], None],
        error_title: str,
    ) -> None:
        """Start one analysis on the shared action bar (#9470).

        ``work`` runs on the worker thread and must not touch widgets;
        ``present`` and the failure dialog run on the GUI thread through the
        bar's queued signal connections.
        """
        self.action_bar.start(
            name,
            work,
            on_finished=present,
            on_failed=lambda message: self._show_error_message(error_title, message),
        )

    def cleanup(self) -> None:
        """Cancel and join any running action (launcher tab close, #9470)."""
        self.action_bar.shutdown()

    # ---- relationship analysis ------------------------------------------


class LaunchMonitorAnalyticsWindow(QtWidgets.QMainWindow):
    """Standalone window wrapper for :class:`MainWidget`."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Launch Monitor Analytics")
        self.resize(1440, 900)
        self.main_widget = MainWidget(self)
        self.setCentralWidget(self.main_widget)
        self._build_menu()

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        assert menu_bar is not None
        file_menu = menu_bar.addMenu("&File")
        assert file_menu is not None
        load_corpus_action = QtGui.QAction("Load &Private Corpus", self)
        load_corpus_action.setStatusTip(
            "Load every source in the authorized private shot corpus."
        )
        load_corpus_action.triggered.connect(
            self.main_widget._on_load_private_corpus  # noqa: SLF001 - same tool
        )
        file_menu.addAction(load_corpus_action)
        file_menu.addSeparator()
        quit_action = QtGui.QAction("&Quit", self)
        quit_action.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        from src.launchers.help_menu import build_help_menu

        build_help_menu(
            menu_bar,
            self,
            doc_target=(
                "User Manual",
                "docs/user_guide/user_manual.md",
            ),
        )

    def _show_about(self) -> None:
        QtWidgets.QMessageBox.about(
            self,
            "About Launch Monitor Analytics",
            "<b>Launch Monitor Analytics</b><br><br>"
            "A vendor-neutral workbench for auditable session aggregation, "
            "impact-parameter interdependency mapping, predictive modeling, "
            "monitor agreement, dispersion, and longitudinal trend analysis.<br><br>"
            "Association and predictive performance do not establish causality.",
        )


def main(argv: list[str] | None = None) -> int:
    """Run the standalone workbench and return the Qt exit code."""
    arguments = argv if argv is not None else sys.argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(arguments)
    window = LaunchMonitorAnalyticsWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
