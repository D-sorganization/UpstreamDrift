#!/usr/bin/env python
"""
C3D Motion Analysis GUI

Features:
- Load C3D files (via C3DDataReader)
- Inspect metadata, markers, analog channels
- 2D plots of marker/analog time-series
- 3D marker trajectory viewer
- Basic kinematic analysis: speed, path length, extrema
- Consolidated loading path and MVC architecture
"""

import os
import sys
from pathlib import Path

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt

from .core.models import C3DDataModel
from .services.loader_thread import C3DLoaderThread
from .ui.tabs.analog_plot_tab import AnalogPlotTab
from .ui.tabs.analysis_tab import AnalysisTab
from .ui.tabs.marker_plot_tab import MarkerPlotTab
from .ui.tabs.overview_tab import OverviewTab
from .ui.tabs.viewer_3d_tab import Viewer3DTab

# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------


class C3DViewerMainWindow(QtWidgets.QMainWindow):
    """Main window for the C3D motion analysis viewer application."""

    def __init__(self) -> None:
        """Initialize the main window and create UI components."""
        super().__init__()

        self.setWindowTitle("C3D Motion Analysis Viewer")
        self.resize(1400, 900)
        self.setAcceptDrops(True)

        self.model: C3DDataModel | None = None
        self._loader_thread: C3DLoaderThread | None = None

        self._create_actions()
        self._create_menus()
        self._create_central_widget()
        self._update_ui_state(False)

        if (sb := self.statusBar()) is not None:
            sb.showMessage("Ready")

    # ----------------------------- UI setup --------------------------------

    def _create_actions(self) -> None:
        """Create menu actions for the application."""
        self.action_open = QtGui.QAction("Open &C3D…", self)
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.setStatusTip("Open a C3D file for analysis")
        self.action_open.triggered.connect(self.open_c3d_file)

        self.action_exit = QtGui.QAction("E&xit", self)
        self.action_exit.setShortcut("Ctrl+Q")
        self.action_exit.triggered.connect(self.close)

        self.action_about = QtGui.QAction("&About", self)
        self.action_about.triggered.connect(self.show_about_dialog)

    def _create_menus(self) -> None:
        """Create menu bar and menus."""
        menubar = self.menuBar()
        if menubar is None:
            return

        file_menu = menubar.addMenu("&File")
        if file_menu is not None:
            file_menu.addAction(self.action_open)
            file_menu.addSeparator()
            file_menu.addAction(self.action_exit)

        help_menu = menubar.addMenu("&Help")
        if help_menu is not None:
            help_menu.addAction(self.action_about)

    def _create_central_widget(self) -> None:
        """Create the central tab widget with all tabs."""
        self.tabs = QtWidgets.QTabWidget()

        self.overview_tab = OverviewTab()
        self.marker_plot_tab = MarkerPlotTab()
        self.analog_plot_tab = AnalogPlotTab()
        self.viewer3d_tab = Viewer3DTab()
        self.analysis_tab = AnalysisTab()

        self.tabs.addTab(self.overview_tab, "Overview")
        self.tabs.setTabToolTip(0, "Metadata and file information")
        self.tabs.addTab(self.marker_plot_tab, "Markers (2D)")
        self.tabs.setTabToolTip(1, "2D plots of marker trajectories")
        self.tabs.addTab(self.analog_plot_tab, "Analog")
        self.tabs.setTabToolTip(2, "Analog data visualization")
        self.tabs.addTab(self.viewer3d_tab, "3D Viewer")
        self.tabs.setTabToolTip(3, "3D interactive view of markers")
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.setTabToolTip(4, "Kinematic analysis and calculations")

        self.setCentralWidget(self.tabs)

    # ---------------------- UI state management ----------------------------

    def _update_ui_state(self, enabled: bool) -> None:
        """Update the enabled state of UI widgets after loading a model."""
        widgets = [
            self.tabs,
        ]
        for w in widgets:
            w.setEnabled(enabled)

    def show_about_dialog(self) -> None:
        """Show the about dialog."""
        QtWidgets.QMessageBox.about(
            self,
            "About C3D Viewer",
            "C3D Viewer\n\nPart of the Golf Modeling Suite.\n"
            "Uses the consolidated C3DDataReader for consistent ingestion.",
        )

    # --------------------------- File I/O ----------------------------------

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                path = urls[0].toLocalFile()
                if path.lower().endswith(".c3d"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        """Handle drop event."""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.load_c3d_file_from_path(path)

    def load_c3d_file_from_path(self, path: str) -> None:
        """Load a C3D file from the given path."""
        # Security validation (F-004)
        # Allow User Home and Project Root
        # shared module import must be available
        from shared.python.security_utils import validate_path

        suite_root = Path(__file__).parents[6]
        allowed = [
            Path.home(),
            suite_root,
        ]
        try:
            # We use strict=False to allow checking but log/warn if outside,
            # or strict=True if we want to block. Review suggested blocking.
            path = str(validate_path(path, allowed, strict=True))
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Security Warning", str(e))
            return

        if (sb := self.statusBar()) is not None:
            sb.showMessage(f"Loading {os.path.basename(path)}... (Async)")

        # Ensure single cursor override
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._update_ui_state(False)  # Disable UI during load

        # Start async worker
        # Keep a reference to prevent garbage collection
        self._loader_thread = C3DLoaderThread(path)
        self._loader_thread.loaded.connect(self._on_load_success)
        self._loader_thread.failed.connect(self._on_load_failure)
        # Ensure we cleanup reference when done
        self._loader_thread.finished.connect(self._on_load_finished)
        self._loader_thread.start()

    def open_c3d_file(self) -> None:
        """Open a file dialog to load a C3D file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open C3D file",
            "",
            "C3D files (*.c3d);;All files (*.*)",
        )
        if path:
            self.load_c3d_file_from_path(path)

    def _on_load_success(self, model: C3DDataModel) -> None:
        """Handle successful model load."""
        self.model = model
        self._populate_ui_with_model()
        self._update_ui_state(True)
        if (sb := self.statusBar()) is not None:
            sb.showMessage(f"Loaded {os.path.basename(model.filepath)} successfully.")

    def _on_load_failure(self, error_msg: str) -> None:
        """Handle load failure."""
        if (sb := self.statusBar()) is not None:
            sb.showMessage("Error loading file.")

        QtWidgets.QMessageBox.critical(
            self,
            "Error loading C3D",
            f"Failed to load file.\n\nError:\n{error_msg}",
        )
        self._update_ui_state(True)  # Re-enable UI (at least menus)

    def _on_load_finished(self) -> None:
        """Cleanup after thread finish."""
        QtWidgets.QApplication.restoreOverrideCursor()
        self._loader_thread = None

    # --------------------- Populate UI from model --------------------------

    def _populate_ui_with_model(self) -> None:
        """Populate UI components with data from the loaded model."""
        if self.model is None:
            return

        self.overview_tab.update_from_model(self.model)
        self.marker_plot_tab.update_from_model(self.model)
        self.analog_plot_tab.update_from_model(self.model)
        self.viewer3d_tab.update_from_model(self.model)
        self.analysis_tab.update_from_model(self.model)


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = C3DViewerMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
